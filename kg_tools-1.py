"""
kg_tools.py
============
LangChain tools that query the Neo4j knowledge graph. Used by the new
Graph agent, the same way tools.py's functions are used by the Data /
RAG / Risk agents.
"""
from langchain.tools import tool

from src.genai.graph_db import run_query


@tool
def find_parts_for_aircraft(aircraft_make: str, aircraft_model: str) -> str:
    """Given an aircraft make and model, list the parts most frequently
    involved in discrepancies for that aircraft, ranked by report count."""
    rows = run_query(
        """
        MATCH (a:Aircraft {make: $make, model: $model})-[:HAS_DISCREPANCY]->(d)-[:INVOLVES_PART]->(p:Part)
        RETURN p.name AS part, count(d) AS reports
        ORDER BY reports DESC
        LIMIT 10
        """,
        {"make": aircraft_make, "model": aircraft_model},
    )
    if not rows:
        return f"No parts found for {aircraft_make} {aircraft_model}. Check the make/model spelling."

    lines = [f"Top parts reported for {aircraft_make} {aircraft_model}:"]
    lines += [f"  {i+1}. {r['part']} — {r['reports']} reports" for i, r in enumerate(rows)]
    return "\n".join(lines)


@tool
def get_cooccurring_parts(part_name: str) -> str:
    """Given a part name, find other parts that frequently appear on the
    same aircraft type, indicating potential related-failure clusters."""
    rows = run_query(
        """
        MATCH (p1:Part {name: $part})-[:INSTALLED_ON]->(a:Aircraft)<-[:INSTALLED_ON]-(p2:Part)
        WHERE p2.name <> $part
        RETURN p2.name AS other_part, count(DISTINCT a) AS shared_aircraft
        ORDER BY shared_aircraft DESC
        LIMIT 8
        """,
        {"part": part_name},
    )
    if not rows:
        return f"No co-occurring parts found for '{part_name}'."

    lines = [f"Parts that co-occur with '{part_name}' on the same aircraft:"]
    lines += [f"  {i+1}. {r['other_part']} — {r['shared_aircraft']} shared aircraft" for i, r in enumerate(rows)]
    return "\n".join(lines)


@tool
def get_common_conditions_for_part(part_name: str) -> str:
    """Given a part name, list the most common reported conditions/root
    causes (nature of condition) associated with that part."""
    rows = run_query(
        """
        MATCH (p:Part {name: $part})<-[:INVOLVES_PART]-(d)-[:HAS_CONDITION]->(c:Condition)
        RETURN c.name AS condition, count(d) AS reports
        ORDER BY reports DESC
        LIMIT 5
        """,
        {"part": part_name},
    )
    if not rows:
        return f"No condition data found for '{part_name}'."

    lines = [f"Most common conditions reported for '{part_name}':"]
    lines += [f"  {i+1}. {r['condition']} — {r['reports']} reports" for i, r in enumerate(rows)]
    return "\n".join(lines)


@tool
def get_system_summary(jasc_code: str) -> str:
    """Given a JASC system code, summarize which parts are linked to that
    system and how many discrepancy reports involve it."""
    rows = run_query(
        """
        MATCH (s:System {jasc_code: $jasc})<-[:PART_OF_SYSTEM]-(p:Part)
        OPTIONAL MATCH (p)<-[:INVOLVES_PART]-(d)
        RETURN p.name AS part, count(DISTINCT d) AS reports
        ORDER BY reports DESC
        """,
        {"jasc": jasc_code},
    )
    if not rows:
        return f"No system found for JASC code '{jasc_code}'."

    total_reports = sum(r["reports"] for r in rows)
    lines = [
        f"System JASC {jasc_code}: {len(rows)} distinct parts, {total_reports} total discrepancy reports.",
        "Parts: " + ", ".join(r["part"] for r in rows[:15]),
    ]
    return "\n".join(lines)


@tool
def get_critical_parts_for_aircraft(aircraft_make: str, aircraft_model: str) -> str:
    """Given an aircraft make and model, list the parts most often linked
    to CRITICAL discrepancy reports (using the existing Critical label
    from the ML pipeline), ranked by critical-report count."""
    rows = run_query(
        """
        MATCH (a:Aircraft {make: $make, model: $model})-[:HAS_DISCREPANCY]->(d)-[:INVOLVES_PART]->(p:Part)
        WHERE d.critical = true
        RETURN p.name AS part, count(d) AS critical_reports
        ORDER BY critical_reports DESC
        LIMIT 10
        """,
        {"make": aircraft_make, "model": aircraft_model},
    )
    if not rows:
        return f"No critical reports found for {aircraft_make} {aircraft_model}."

    lines = [f"Parts most linked to CRITICAL reports for {aircraft_make} {aircraft_model}:"]
    lines += [f"  {i+1}. {r['part']} — {r['critical_reports']} critical reports" for i, r in enumerate(rows)]
    return "\n".join(lines)


GRAPH_TOOLS = [
    find_parts_for_aircraft,
    get_cooccurring_parts,
    get_common_conditions_for_part,
    get_system_summary,
    get_critical_parts_for_aircraft,
]
