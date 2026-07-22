"""
graph_agent.py
===============
6th agent: relational/graph queries over the SDR knowledge graph
(co-occurring parts, system-level rollups, part->condition patterns).

This follows the same shape as rag_agent.py / risk_agent.py — swap in
whatever your _specialist_base.py's actual runner signature is if it
differs (e.g. make_specialist_agent(name, tools, system_prompt)).
"""
from src.genai.llm import get_chat_llm
from src.genai.agents._specialist_base import make_specialist_agent
from src.genai.kg.kg_tools import GRAPH_TOOLS

SYSTEM_PROMPT = """You are the Graph specialist for an FAA Service
Difficulty Report analytics assistant. You answer RELATIONAL questions
that a plain keyword or semantic search can't: which parts co-occur on
the same aircraft, which conditions are most associated with a given
part, and system-level (JASC) rollups.

Use your tools to look up real graph data before answering. Do not
invent part names, aircraft models, or counts — only report what the
tools return. If a lookup returns no data, say so plainly.
"""


def build_graph_agent():
    llm = get_chat_llm()
    return make_specialist_agent(
        name="graph_agent",
        llm=llm,
        tools=GRAPH_TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )
