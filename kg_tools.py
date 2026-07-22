"""
kg_tools.py
============
LangChain tools that query the knowledge graph. Used by the new
Graph agent, the same way src/genai/tools.py's functions are used by
the Data Analytics / Root-Cause / Risk agents.
"""
from collections import Counter
from langchain.tools import tool

from .graph_store import load_graph


@tool
def find_parts_for_aircraft(aircraft_make: str, aircraft_model: str) -> str:
    """Given an aircraft make and model, list the parts most frequently
    involved in discrepancies for that aircraft, ranked by report count."""
    G = load_graph()
    aircraft_id = f"Aircraft::{aircraft_make}::{aircraft_model}"
    if aircraft_id not in G:
        return f"No aircraft found matching make='{aircraft_make}', model='{aircraft_model}'."

    part_counts = Counter()
    for _, discrepancy_id in G.out_edges(aircraft_id):
        for _, part_id, data in G.out_edges(discrepancy_id, data=True):
            if data.get("relation") == "INVOLVES_PART":
                part_counts[part_id.replace("Part::", "")] += 1

    if not part_counts:
        return f"No parts found linked to {aircraft_make} {aircraft_model}."

    top = part_counts.most_common(10)
    lines = [f"Top parts reported for {aircraft_make} {aircraft_model}:"]
    lines += [f"  {i+1}. {name} — {count} reports" for i, (name, count) in enumerate(top)]
    return "\n".join(lines)


@tool
def get_cooccurring_parts(part_name: str) -> str:
    """Given a part name, find other parts that frequently appear on the
    same aircraft type, indicating potential related-failure clusters."""
    G = load_graph()
    part_id = f"Part::{part_name}"
    if part_id not in G:
        return f"No part found matching '{part_name}'."

    aircraft_ids = [
        v for _, v, d in G.out_edges(part_id, data=True)
        if d.get("relation") == "INSTALLED_ON"
    ]
    if not aircraft_ids:
        return f"No aircraft linkage found for part '{part_name}'."

    co_parts = Counter()

    # Find discrepancies that involve this part, then other parts on
    # the same discrepancy report's aircraft.
    discrepancy_ids = [
        u for u, v, d in G.in_edges(part_id, data=True)
        if d.get("relation") == "INVOLVES_PART"
    ]
    for disc_id in discrepancy_ids:
        for u, v, d in G.in_edges(disc_id, data=True):
            if d.get("relation") == "HAS_DISCREPANCY":
                aircraft_id = u
                for _, other_disc, dd in G.out_edges(aircraft_id, data=True):
                    if dd.get("relation") != "HAS_DISCREPANCY" or other_disc == disc_id:
                        continue
                    for _, other_part, ddd in G.out_edges(other_disc, data=True):
                        if ddd.get("relation") == "INVOLVES_PART":
                            name = other_part.replace("Part::", "")
                            if name != part_name:
                                co_parts[name] += 1

    if not co_parts:
        return f"No co-occurring parts found for '{part_name}'."

    top = co_parts.most_common(8)
    lines = [f"Parts that co-occur with '{part_name}' on the same aircraft:"]
    lines += [f"  {i+1}. {name} — {count} shared aircraft reports" for i, (name, count) in enumerate(top)]
    return "\n".join(lines)


@tool
def get_common_conditions_for_part(part_name: str) -> str:
    """Given a part name, list the most common reported conditions/root
    causes (nature of condition) associated with that part."""
    G = load_graph()
    part_id = f"Part::{part_name}"
    if part_id not in G:
        return f"No part found matching '{part_name}'."

    discrepancy_ids = [
        u for u, v, d in G.in_edges(part_id, data=True)
        if d.get("relation") == "INVOLVES_PART"
    ]
    condition_counts = Counter()
    for disc_id in discrepancy_ids:
        for _, cond_id, d in G.out_edges(disc_id, data=True):
            if d.get("relation") == "HAS_CONDITION":
                condition_counts[cond_id.replace("Condition::", "")] += 1

    if not condition_counts:
        return f"No condition data found for '{part_name}'."

    top = condition_counts.most_common(5)
    lines = [f"Most common conditions reported for '{part_name}':"]
    lines += [f"  {i+1}. {name} — {count} reports" for i, (name, count) in enumerate(top)]
    return "\n".join(lines)


@tool
def get_system_summary(jasc_code: str) -> str:
    """Given a JASC system code, summarize which parts and aircraft are
    linked to that system and how many discrepancy reports involve it."""
    G = load_graph()
    system_id = f"System::{jasc_code}"
    if system_id not in G:
        return f"No system found for JASC code '{jasc_code}'."

    parts = [u for u, v, d in G.in_edges(system_id, data=True) if d.get("relation") == "PART_OF_SYSTEM"]
    total_reports = 0
    for part_id in parts:
        total_reports += sum(
            1 for _, _, d in G.in_edges(part_id, data=True) if d.get("relation") == "INVOLVES_PART"
        )

    part_names = [p.replace("Part::", "") for p in parts]
    return (
        f"System JASC {jasc_code}: {len(part_names)} distinct parts, "
        f"{total_reports} total discrepancy reports.\n"
        f"Parts: {', '.join(part_names[:15])}"
    )


GRAPH_TOOLS = [
    find_parts_for_aircraft,
    get_cooccurring_parts,
    get_common_conditions_for_part,
    get_system_summary,
]
