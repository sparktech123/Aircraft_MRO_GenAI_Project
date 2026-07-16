"""Root-Cause RAG Agent — retrieves semantically similar historical
discrepancy narratives from the vector store and synthesizes a grounded
root-cause explanation. Every claim should trace back to retrieved
reports, not to the model's un-grounded prior knowledge."""
from src.genai.agents._specialist_base import run_specialist
from src.genai.state import AgentState
from src.genai.tools import RAG_AGENT_TOOLS

SYSTEM_PROMPT = """You are the Root-Cause Analysis agent for an FAA \
Service Difficulty Report assistant. Use the search_similar_reports tool \
to retrieve real historical discrepancy narratives relevant to the \
question, then synthesize a root-cause explanation GROUNDED in what you \
retrieved. Mention the recurring failure patterns you see across the \
retrieved reports (e.g. corrosion, fatigue cracking, wear). Do not \
invent causes that aren't supported by the retrieved text. If nothing \
relevant is retrieved, say so plainly."""


def rag_agent_node(state: AgentState) -> AgentState:
    return run_specialist(state, RAG_AGENT_TOOLS, SYSTEM_PROMPT, "rag_agent_output")
