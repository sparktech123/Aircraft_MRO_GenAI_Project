# FAA SDR Intelligence Assistant — GenAI Layer

A free, fully local multi-agent GenAI system built on top of the FAA
Service Difficulty Report (SDR) capstone (Tasks 1–5: data prep, EDA,
NLP/root-cause, critical-issue classification, early warning system).

**Stack — 100% free, no API keys, nothing leaves your machine:**
- **Ollama** — local LLM + local embeddings
- **LangChain** — tools, prompts
- **LangGraph** — multi-agent orchestration (supervisor pattern)
- **Chroma** — local vector store for retrieval-augmented root-cause analysis
- **Streamlit** — chat UI

---

## Architecture — 5 agents

```
                 ┌───────────────┐
   question ───► │  Supervisor   │  (routes to 1-3 specialists)
                 └───────┬───────┘
           ┌─────────────┼─────────────┐
           ▼             ▼             ▼
   📊 Data Analytics  🔎 Root-Cause  ⚠️ Risk / Early-Warning
      Agent              (RAG) Agent    Agent
           │             │             │
           └─────────────┴──────┬──────┘
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
| **Report Composer** | Weaves specialist outputs into one coherent answer | — (LLM synthesis only) |

Why 5 and not more: each agent owns one clearly separable job (routing,
quantitative lookup, semantic retrieval, risk scoring, synthesis). More
agents would just fragment these responsibilities without adding
capability; fewer would force one agent/prompt to juggle unrelated tool
sets, which local models handle worse than cloud-scale ones.

Why tool-scoped agents instead of one open "run any pandas code" agent:
safer (no arbitrary code execution), and small local Ollama models are
much more reliable calling 2-3 well-named tools than writing correct
pandas/SQL from scratch every time.

---

## Folder structure

```
genai_project/
├── data/
│   ├── raw/              # put the original FAA SDR CSVs here
│   ├── processed/        # cleaned + labeled dataset, Task 4/5 outputs
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
│       ├── tools.py                  # LangChain tools used by the agents
│       ├── state.py                  # shared LangGraph state
│       ├── graph.py                  # wires the 5 agents into a LangGraph graph
│       └── agents/
│           ├── supervisor.py
│           ├── data_agent.py
│           ├── rag_agent.py
│           ├── risk_agent.py
│           ├── report_agent.py
│           └── _specialist_base.py   # shared tool-agent runner
├── app/
│   └── streamlit_app.py              # chat UI + alerts/control-chart quick views
├── scripts/
│   ├── run_pipeline.py               # runs Task 4 + Task 5 end-to-end
│   └── build_vectorstore.py          # (re)builds the Chroma store
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
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # defaults already work, edit if you used different model names
```

### 3. Add your data
Copy your two source CSVs into `data/raw/`:
- `Concated_Final_Cleaned_Dataset.csv`
- `Cleaned_Output_FAA-SDR-2026.csv`

### 4. Run the pipeline (Task 4 + Task 5 → populates `data/processed/` and `models/`)
```bash
python scripts/run_pipeline.py
```

### 5. Build the vector store (Task 3-style retrieval, powers the RAG agent)
```bash
python scripts/build_vectorstore.py
```
This embeds a capped, deduplicated sample of discrepancy narratives
(15,000 by default — see `VECTORSTORE_MAX_DOCS` in `.env`) using the
local `nomic-embed-text` model. On CPU-only machines this can take a
while; lower the cap if you want a faster first run.

### 6. Launch the app
```bash
streamlit run app/streamlit_app.py
```

---

## Example questions to try
- "How many reports were filed for landing gear in 2024?" → Data Analytics agent
- "Why do battery packs keep failing on the 737?" → Root-Cause (RAG) agent
- "Is anything trending worse right now that I should worry about?" → Risk agent
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
