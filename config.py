"""
Streamlit UI for the FAA SDR multi-agent GenAI assistant.

Run with:
    streamlit run app/streamlit_app.py

Requires:
    - Ollama running locally with the chat + embedding models pulled
      (see README.md)
    - The vector store already built: python scripts/build_vectorstore.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from src.genai.graph import build_graph

st.set_page_config(page_title="FAA SDR Intelligence Assistant", page_icon="✈️", layout="wide")


@st.cache_resource
def get_graph_app():
    return build_graph()


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.title("✈️ FAA SDR Assistant")
    st.caption("Multi-agent GenAI layer on top of the Task 1-5 pipeline")

    st.markdown("### Stack")
    st.markdown(
        f"- **LLM:** Ollama · `{config.OLLAMA_CHAT_MODEL}`\n"
        f"- **Embeddings:** Ollama · `{config.OLLAMA_EMBED_MODEL}`\n"
        "- **Orchestration:** LangGraph (5-agent supervisor graph)\n"
        "- **Vector store:** Chroma (local, on disk)"
    )

    st.markdown("### Agents")
    st.markdown(
        "1. 🧭 Supervisor / Router\n"
        "2. 📊 Data Analytics\n"
        "3. 🔎 Root-Cause (RAG)\n"
        "4. ⚠️ Risk / Early-Warning\n"
        "5. 📝 Report Composer"
    )

    st.markdown("### Quick links")
    tab_choice = st.radio("View", ["Chat", "Current Alerts", "Fleet Control Chart"], label_visibility="collapsed")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ------------------------------------------------------------------
# Alerts / control-chart quick views (read straight from Task 5 CSVs)
# ------------------------------------------------------------------
if tab_choice == "Current Alerts":
    st.header("⚠️ Current Early-Warning Alerts")
    import pandas as pd
    try:
        alerts = pd.read_csv(config.ALERTS_PATH)
        st.dataframe(alerts, use_container_width=True)
    except FileNotFoundError:
        st.warning("Alerts file not found. Run the Task 5 pipeline first.")
    st.stop()

if tab_choice == "Fleet Control Chart":
    st.header("📈 Fleet-wide Monthly Critical Issue Rate")
    import pandas as pd
    try:
        monthly = pd.read_csv(config.MONTHLY_TREND_PATH)
        st.line_chart(monthly.set_index("YearMonth_str")[["critical_rate", "roll_mean"]])
        st.dataframe(monthly, use_container_width=True)
    except FileNotFoundError:
        st.warning("Monthly trend file not found. Run the Task 5 pipeline first.")
    st.stop()

# ------------------------------------------------------------------
# Chat
# ------------------------------------------------------------------
st.header("✈️ FAA SDR Intelligence Assistant")
st.caption("Ask about trends, root causes, or current risk — grounded in your real SDR data. Free & local (Ollama).")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("trace"):
            with st.expander("Agent trace"):
                st.text(msg["trace"])

user_input = st.chat_input("e.g. Why do battery packs keep failing on the 737, and how risky is it right now?")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Routing to the right specialist agent(s)..."):
            try:
                graph_app = get_graph_app()
                result = graph_app.invoke({
                    "question": user_input,
                    "chat_history": [(m["role"], m["content"]) for m in st.session_state.messages[:-1]],
                })
                answer = result.get("final_answer", "I couldn't produce an answer.")
                trace_lines = [result.get("routing_reason", "")]
                for key, label in [
                    ("data_agent_output", "Data Analytics agent"),
                    ("rag_agent_output", "Root-Cause (RAG) agent"),
                    ("risk_agent_output", "Risk / Early-Warning agent"),
                ]:
                    if result.get(key):
                        trace_lines.append(f"\n--- {label} ---\n{result[key]}")
                trace = "\n".join(trace_lines)
            except Exception as e:
                answer = (
                    f"⚠️ Something went wrong talking to Ollama or the agent graph: `{e}`\n\n"
                    "Make sure Ollama is running (`ollama serve`), the models are pulled "
                    f"(`ollama pull {config.OLLAMA_CHAT_MODEL}`, `ollama pull {config.OLLAMA_EMBED_MODEL}`), "
                    "and the vector store has been built (`python scripts/build_vectorstore.py`)."
                )
                trace = ""

        st.markdown(answer)
        if trace:
            with st.expander("Agent trace"):
                st.text(trace)

    st.session_state.messages.append({"role": "assistant", "content": answer, "trace": trace})
