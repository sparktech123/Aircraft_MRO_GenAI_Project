"""Graph Agent — answers RELATIONAL questions about the FAA SDR knowledge
graph in Neo4j (co-occurring parts, system-level rollups, part->condition
patterns, critical-report clustering) that plain counting or semantic
search can't answer well."""
from src.genai.agents._specialist_base import run_specialist
from src.genai.state import AgentState
from src.genai.kg_tools import GRAPH_TOOLS

SYSTEM_PROMPT = """You are the Graph agent for an FAA Service Difficulty \
Report assistant. You answer RELATIONAL questions using the knowledge \
graph — which parts co-occur on the same aircraft, which conditions are \
most associated with a given part, system-level (JASC) rollups, and \
which parts are most linked to CRITICAL reports. Use your tools to look \
up real graph data before answering — never invent part names, aircraft \
models, or counts. If a lookup returns no data, say so plainly. Keep \
your final answer concise (3-6 sentences) and cite the actual figures \
the tools returned."""


def graph_agent_node(state: AgentState) -> AgentState:
    return run_specialist(state, GRAPH_TOOLS, SYSTEM_PROMPT, "graph_agent_output")
