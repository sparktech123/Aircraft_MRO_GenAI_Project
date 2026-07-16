"""
Supervisor / Router agent.

Reads the user's question and decides which of the three specialist
agents (data_agent, rag_agent, risk_agent) are relevant. A question can
route to more than one — e.g. "why is the 737 battery pack trending
worse and how risky is it" needs both the RAG agent (root cause) and the
risk agent (current alert status).

Uses a small structured-output call so a local Ollama model doesn't need
to produce perfectly-formed free text — just a short list of labels.
"""
import json
import re

from src.genai.state import AgentState
from src.genai.llm import get_llm

VALID_ROUTES = ["data_agent", "rag_agent", "risk_agent"]

SYSTEM_PROMPT = """You are a routing controller for an FAA Service Difficulty \
Report (SDR) analysis assistant. Given a user question, decide which \
specialist agent(s) are needed to answer it. Respond with ONLY a JSON \
array of one to three of these labels, nothing else:

- "data_agent": for questions about counts, trends over time, top \
  parts/aircraft by volume, general dataset statistics.
- "rag_agent": for "why" / root-cause questions, or requests to find \
  similar historical reports / explain what typically causes an issue.
- "risk_agent": for questions about current early-warning alerts, \
  whether something is trending worse right now, or scoring how risky a \
  new/hypothetical discrepancy sounds.

Examples:
Q: "How many reports were filed for landing gear in 2024?"
A: ["data_agent"]

Q: "Why do battery packs keep failing on the 737?"
A: ["rag_agent"]

Q: "Is anything trending worse right now that I should worry about?"
A: ["risk_agent"]

Q: "Why is the skin corrosion issue on the 717 getting worse, and how risky is it?"
A: ["rag_agent", "risk_agent"]

Only output the JSON array."""


def _parse_routes(text: str) -> list[str]:
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if not match:
        return ["data_agent"]  # safe default
    try:
        routes = json.loads(match.group(0))
    except json.JSONDecodeError:
        return ["data_agent"]
    routes = [r for r in routes if r in VALID_ROUTES]
    return routes or ["data_agent"]


def supervisor_node(state: AgentState) -> AgentState:
    llm = get_llm(temperature=0)
    response = llm.invoke([
        ("system", SYSTEM_PROMPT),
        ("human", state["question"]),
    ])
    routes = _parse_routes(response.content)
    return {
        **state,
        "remaining_routes": routes,
        "routing_reason": f"Routed to: {', '.join(routes)}",
    }


def route_next(state: AgentState) -> str:
    """Conditional-edge function: peek at the next agent to run, or go to
    the report agent once the queue is empty."""
    remaining = state.get("remaining_routes") or []
    if remaining:
        return remaining[0]
    return "report_agent"
