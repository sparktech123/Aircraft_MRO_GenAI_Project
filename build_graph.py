"""
build_graph.py
===============
Builds a NetworkX knowledge graph from the labeled FAA SDR dataset and
persists it to disk. Mirrors scripts/build_vectorstore.py in role: this
is the "build once, load many times" step that powers the new Graph
agent, the same way build_vectorstore.py powers the RAG agent.

Run:
    python scripts/build_graph.py

Output:
    data/processed/kg_graph.gpickle
"""
import pickle
import pandas as pd
import networkx as nx

from config import LABELED_DATASET_PATH, DATA_PROCESSED_DIR

KG_GRAPH_PATH = DATA_PROCESSED_DIR / "kg_graph.gpickle"

# ---------------------------------------------------------------------
# COLUMN_MAP — adjust the values (right side) to match your CSV's exact
# headers. Left side is the internal name used everywhere else in this
# file, so you only ever need to edit this one dict.
# ---------------------------------------------------------------------
COLUMN_MAP = {
    "control_number": "Operator Control Number",
    "difficulty_date": "Difficulty Date",
    "operator": "Operator Designator",
    "aircraft_make": "Aircraft Make",
    "aircraft_model": "Aircraft Model",
    "aircraft_serial": "Aircraft Serial Number",
    "jasc_code": "JASC Code",
    "nature_of_condition": "Nature Of Condition",
    "precautionary_procedure": "Precautionary Procedure",
    "stage_of_operation": "Stage Of Operation Code",
    "how_discovered": "How Discovered Code",
    "part_number": "Part Number",
    "part_name": "Part Name",
    "part_condition": "Part Condition",
    "discrepancy": "Discrepancy",
}


def _get(row, key):
    """Safe lookup that returns None for missing/NaN values."""
    col = COLUMN_MAP[key]
    if col not in row or pd.isna(row[col]):
        return None
    val = str(row[col]).strip()
    return val if val else None


def build_graph(df: pd.DataFrame) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()

    for _, row in df.iterrows():
        control_no = _get(row, "control_number")
        if control_no is None:
            continue  # can't anchor a report without an ID

        discrepancy_id = f"Discrepancy::{control_no}"
        G.add_node(
            discrepancy_id,
            type="Discrepancy",
            date=_get(row, "difficulty_date"),
            text=_get(row, "discrepancy"),
            precautionary_procedure=_get(row, "precautionary_procedure"),
            stage_of_operation=_get(row, "stage_of_operation"),
        )

        # --- Aircraft node (make + model as identity) ---
        make = _get(row, "aircraft_make")
        model = _get(row, "aircraft_model")
        if make or model:
            aircraft_id = f"Aircraft::{make}::{model}"
            G.add_node(aircraft_id, type="Aircraft", make=make, model=model)
            G.add_edge(aircraft_id, discrepancy_id, relation="HAS_DISCREPANCY")

        # --- Part node ---
        part_name = _get(row, "part_name")
        if part_name:
            part_id = f"Part::{part_name}"
            G.add_node(
                part_id,
                type="Part",
                part_number=_get(row, "part_number"),
                condition=_get(row, "part_condition"),
            )
            G.add_edge(discrepancy_id, part_id, relation="INVOLVES_PART")
            if make or model:
                G.add_edge(part_id, aircraft_id, relation="INSTALLED_ON")

        # --- System node (via JASC code) ---
        jasc = _get(row, "jasc_code")
        if jasc:
            system_id = f"System::{jasc}"
            G.add_node(system_id, type="System", jasc_code=jasc)
            if part_name:
                G.add_edge(part_id, system_id, relation="PART_OF_SYSTEM")

        # --- Root cause / condition node ---
        condition = _get(row, "nature_of_condition")
        if condition:
            cause_id = f"Condition::{condition}"
            G.add_node(cause_id, type="Condition")
            G.add_edge(discrepancy_id, cause_id, relation="HAS_CONDITION")

        # --- How discovered ---
        how = _get(row, "how_discovered")
        if how:
            how_id = f"Discovery::{how}"
            G.add_node(how_id, type="DiscoveryMethod")
            G.add_edge(discrepancy_id, how_id, relation="DISCOVERED_VIA")

        # --- Operator ---
        operator = _get(row, "operator")
        if operator:
            operator_id = f"Operator::{operator}"
            G.add_node(operator_id, type="Operator")
            G.add_edge(operator_id, discrepancy_id, relation="FILED_BY")

    return G


def main():
    print(f"Loading labeled dataset from {LABELED_DATASET_PATH} ...")
    df = pd.read_csv(LABELED_DATASET_PATH, low_memory=False)
    print(f"  {len(df):,} rows")

    missing = [c for c in COLUMN_MAP.values() if c not in df.columns]
    if missing:
        print("WARNING: these expected columns were not found in the CSV:")
        for m in missing:
            print(f"   - {m}")
        print("Update COLUMN_MAP in build_graph.py to match your actual headers.\n")

    G = build_graph(df)
    print(f"Graph built: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    with open(KG_GRAPH_PATH, "wb") as f:
        pickle.dump(G, f)
    print(f"Saved to {KG_GRAPH_PATH}")


if __name__ == "__main__":
    main()
