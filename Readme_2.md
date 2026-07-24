# FAA SDR Intelligence Assistant — GenAI Layer

A free, fully local multi-agent GenAI system built on top of the FAA
Service Difficulty Report (SDR) capstone (Tasks 1–5: data prep, EDA,
NLP/root-cause, critical-issue classification, early warning system) —
now extended with a Neo4j-backed knowledge graph for relational
reasoning over parts, aircraft, and systems.

**Stack — no API keys, everything runs locally except the graph DB:**
- **Ollama** — local LLM + local embeddings
- **LangChain** — tools, prompts
- **LangGraph** — multi-agent orchestration (supervisor pattern)
- **Chroma** — local vector store for retrieval-augmented root-cause analysis
- **Neo4j Aura** — hosted graph database for relational KG queries
- **Streamlit** — chat UI

---

## Architecture — 6 agents

```
                 ┌───────────────┐
   question ───► │  Supervisor   │  (routes to 1-4 specialists)
                 └───────┬───────┘
        ┌────────────────┼────────────────┬─────────────┐
        ▼                ▼                ▼             ▼
 📊 Data Analytics  🔎 Root-Cause  ⚠️ Risk / Early-    🕸️ Knowledge
    Agent              (RAG) Agent    Warning Agent       Graph Agent
        │                │                │             │
        └────────────────┴────────┬───────┴─────────────┘
                                   ▼
                          📝 Report Composer
                                   │
                                   ▼
                            final answer
```

| Agent | Role | Tools it uses |
|---|---|---|
| **Supervisor / Router** | Decides which specialist(s) a question needs | — (LLM classification only) |
| **Data Analytics** | Counts, trends, top parts/aircraft | `get_dataset_summary`, `get_top_parts`, `get_monthly_trend` |
| **Root-Cause (RAG)** | Retrieves similar historical reports, explains likely causes | `search_similar_reports` (Chroma vector search) |
| **Risk / Early-Warning** | Current alerts, control-chart status, scores new/hypothetical reports | `get_current_alerts`, `get_fleet_anomaly_status`, `score_discrepancy` |
| **Knowledge Graph** | Relational questions — co-occurring parts, JASC system rollups, part→condition patterns, critical-report clustering | `find_parts_for_aircraft`, `get_cooccurring_parts`, `get_common_conditions_for_part`, `get_system_summary`, `get_critical_parts_for_aircraft` (Neo4j Cypher queries) |
| **Report Composer** | Weaves specialist outputs into one coherent answer | — (LLM synthesis only) |

Why a Knowledge Graph agent alongside RAG: RAG answers semantic "why"
questions by matching narrative text; it can't answer relational
questions like "which parts fail together on the same aircraft" or
"what's linked to this JASC system" — that requires actual graph
traversal over structured entities (Aircraft, Part, System, Condition,
Discrepancy), which is what Neo4j is built for.

---

## Folder structure

```
genai_project/
├── data/
│   ├── raw/               # put the original FAA SDR CSVs here
│   ├── processed/         # cleaned + labeled dataset, Task 4/5 outputs
│   └── vector_store/      # persisted Chroma DB (built by scripts/build_vectorstore.py)
├── models/
│   └── critical_issue_model.joblib   # trained Task 4 classifier + vectorizer/encoder
├── src/
│   ├── data_pipeline/
│   │   └── clean_data.py             # Task 1
│   ├── ml/
│   │   ├── train_classifier.py       # Task 4
│   │   └── early_warning.py          # Task 5
│   └── genai/
│       ├── llm.py                    # Ollama LLM + embeddings factory
│       ├── vectorstore.py            # Chroma build/load for the RAG agent
│       ├── graph_db.py               # Neo4j driver + query helper for the Graph agent
│       ├── kg_tools.py               # LangChain tools that query the knowledge graph
│       ├── tools.py                  # LangChain tools used by the other agents
│       ├── state.py                  # shared LangGraph state
│       ├── graph.py                  # wires the 6 agents into a LangGraph graph
│       └── agents/
│           ├── supervisor.py
│           ├── data_agent.py
│           ├── rag_agent.py
│           ├── risk_agent.py
│           ├── graph_agent.py        # Knowledge Graph specialist
│           ├── report_agent.py
│           └── _specialist_base.py   # shared tool-agent runner
├── app/
│   └── streamlit_app.py              # chat UI + alerts/control-chart quick views
├── scripts/
│   ├── run_pipeline.py               # runs Task 4 + Task 5 end-to-end
│   ├── build_vectorstore.py          # (re)builds the Chroma store
│   └── build_kg.py                   # (re)builds the Neo4j knowledge graph
├── notebooks/                        # optional, for your own EDA
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### 1. Install Ollama and pull the two free local models
```bash
# https://ollama.com/download
ollama pull llama3.1:8b        # chat model (swap for a smaller one, e.g. llama3.2:3b, if your machine is limited)
ollama pull nomic-embed-text   # embedding model
ollama serve                    # usually starts automatically after install
```

### 2. Python environment
```bash
cd genai_project
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass  # if got error in opening the .venv then use this
# python -m venv .venv && source .venv/bin/activate         # Windows: .venv\Scripts\activate        
.\.venv\Scripts\Activate.ps1                                # for virtual environment.
pip install -r requirements.txt
cp .env.example .env                                        # defaults already work, edit if you used different model names
```

### 3. Set up Neo4j Aura (free tier)
1. Create a free instance at https://console.neo4j.io if you haven't already.
2. Download the instance's credentials file (Aura gives you this once, on creation).
3. Add these four values to your `.env`:
   ```
   NEO4J_URI=neo4j+s://<your-instance-id>.databases.neo4j.io
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=<your-password>
   NEO4J_DATABASE=neo4j
   ```
   Note: Aura's actual auth username is always `neo4j`, and the default
   database name is also `neo4j` — the instance ID (e.g. `7a331d30`) is
   just the instance's display name, not the database name or username.

### 4. Add your data
Copy your two source CSVs into `data/raw/`:
- `Concated_Final_Cleaned_Dataset.csv`
- `Cleaned_Output_FAA-SDR-2026.csv`

### 5. Run the pipeline (Task 4 + Task 5 → populates `data/processed/` and `models/`)
```bash
python scripts/run_pipeline.py
```

### 6. Build the vector store (Task 3-style retrieval, powers the RAG agent)
```bash
python scripts/build_vectorstore.py
```
This embeds a capped, deduplicated sample of discrepancy narratives
(15,000 by default — see `VECTORSTORE_MAX_DOCS` in `.env`) using the
local `nomic-embed-text` model. On CPU-only machines this can take a
while; lower the cap if you want a faster first run.

### 7. Build the knowledge graph (powers the Graph agent)
```bash
python scripts/build_kg.py
```
Loads `data/processed/faa_sdr_labeled_dataset.csv` into Neo4j Aura as
`Aircraft`, `Part`, `System`, `Condition`, `Operator`, and `Discrepancy`
nodes with relationships between them. Safe to re-run — every write
uses `MERGE`, so re-running refreshes data rather than duplicating it.
On Aura's free tier this can take a few minutes for large datasets
(runs in batches of 500 rows).

### 8. Launch the app
```bash
streamlit run app/streamlit_app.py
```

---

## Example questions to try
- "How many reports were filed for landing gear in 2024?" → Data Analytics agent
- "Why do battery packs keep failing on the 737?" → Root-Cause (RAG) agent
- "Is anything trending worse right now that I should worry about?" → Risk agent
- "Which parts tend to fail together on the same aircraft?" → Knowledge Graph agent
- "What parts are most linked to critical reports on the 737?" → Knowledge Graph agent
- "Why is the skin corrosion issue on the 717 getting worse, and how risky is it?" → RAG + Risk agents, woven together by the Report Composer

---

## Notes / things to tune
- **Model size vs. machine:** `llama3.1:8b` needs ~8GB RAM free. If that's
  tight, use `llama3.2:3b` or `qwen2.5:3b` instead — just update
  `OLLAMA_CHAT_MODEL` in `.env`.
- **Vector store size:** local CPU embedding is the slowest step. Start
  small (`VECTORSTORE_MAX_DOCS=3000`) to confirm everything works, then
  scale up.
- **Alert/anomaly thresholds** live in `src/ml/early_warning.py` (control
  chart: rolling window + 2σ; alerts: +5pp lift, >15% recent rate,
  ≥40 total / ≥15 recent reports) — adjust to your risk tolerance.
- **Routing quality** depends on the chat model's instruction-following.
  If a small local model routes questions oddly, either upgrade the
  model or tighten the few-shot examples in
  `src/genai/agents/supervisor.py`.
- **Neo4j Aura free tier** pauses after a period of inactivity — the
  first query after a pause can take a few extra seconds while it
  wakes back up. This is expected, not a bug.
cd
