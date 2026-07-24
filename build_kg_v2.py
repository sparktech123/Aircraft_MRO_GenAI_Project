"""
build_kg_v2.py
================
Builds the FAA SDR knowledge graph in Neo4j using the causal-chain
schema (AircraftModel -> Part -> Condition -> RootCause -> Issue ->
Severity/Operator/Region, RootCause -> Recommendation, Report -> Issue).

This REPLACES build_kg.py's schema — it is a different design (relational
part/system graph vs. this causal-chain graph), not an addition to it.
If you already ran build_kg.py, run this against a fresh/cleared database
or the two schemas will coexist confusingly in the same graph.

To clear the database first (CAUTION — deletes everything):
    MATCH (n) DETACH DELETE n

Run from the project root:
    python scripts/build_kg_v2.py
"""
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config
from src.genai.graph_db import get_driver, close_driver

BATCH_SIZE = 500

CONSTRAINTS = [
    "CREATE CONSTRAINT report_id IF NOT EXISTS FOR (r:Report) REQUIRE r.control_number IS UNIQUE",
    "CREATE CONSTRAINT aircraft_id IF NOT EXISTS FOR (a:AircraftModel) REQUIRE (a.make, a.model) IS NODE KEY",
    "CREATE CONSTRAINT part_id IF NOT EXISTS FOR (p:Part) REQUIRE p.name IS UNIQUE",
    "CREATE CONSTRAINT condition_id IF NOT EXISTS FOR (c:Condition) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT rootcause_id IF NOT EXISTS FOR (rc:RootCause) REQUIRE rc.code IS UNIQUE",
    "CREATE CONSTRAINT severity_id IF NOT EXISTS FOR (s:Severity) REQUIRE s.tier IS UNIQUE",
    "CREATE CONSTRAINT operator_id IF NOT EXISTS FOR (o:Operator) REQUIRE o.name IS UNIQUE",
    "CREATE CONSTRAINT region_id IF NOT EXISTS FOR (rg:Region) REQUIRE rg.code IS UNIQUE",
    "CREATE CONSTRAINT recommendation_id IF NOT EXISTS FOR (rm:Recommendation) REQUIRE rm.text IS UNIQUE",
]

MERGE_QUERY = """
UNWIND $rows AS row

MERGE (rep:Report {control_number: row.control_number})
SET rep.date = row.date

WITH row, rep
WHERE row.issue_text IS NOT NULL
MERGE (iss:Issue {control_number: row.control_number})
SET iss.text = row.issue_text
MERGE (rep)-[:CONTAINS]->(iss)

WITH row, iss
WHERE row.make IS NOT NULL OR row.model IS NOT NULL
MERGE (ac:AircraftModel {make: row.make, model: row.model})
WITH row, iss, ac
WHERE row.part_name IS NOT NULL
MERGE (p:Part {name: row.part_name})
MERGE (ac)-[:HAS_PART]->(p)

WITH row, iss, p
WHERE row.condition IS NOT NULL
MERGE (c:Condition {name: row.condition})
MERGE (p)-[:HAS_CONDITION]->(c)

WITH row, iss, c
WHERE row.root_cause_code IS NOT NULL
MERGE (rc:RootCause {code: row.root_cause_code})
MERGE (c)-[:CAUSED_BY]->(rc)
MERGE (rc)-[:LEADS_TO]->(iss)

WITH row, iss, rc
WHERE row.recommendation IS NOT NULL
MERGE (rm:Recommendation {text: row.recommendation})
MERGE (rc)-[:RECOMMENDED_ACTION]->(rm)

WITH row, iss
WHERE row.severity_tier IS NOT NULL
MERGE (sv:Severity {tier: row.severity_tier})
MERGE (iss)-[:HAS_SEVERITY]->(sv)

WITH row, iss
WHERE row.operator IS NOT NULL
MERGE (op:Operator {name: row.operator})
MERGE (iss)-[:REPORTED_BY]->(op)

WITH row, iss
WHERE row.region IS NOT NULL
MERGE (rg:Region {code: row.region})
MERGE (iss)-[:OCCURRED_IN]->(rg)
"""


def _clean(val):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    if isinstance(val, str) and not val.strip():
        return None
    return val


def _row_to_dict(row) -> dict:
    return {
        "control_number": _clean(row.get("OperatorControlNumber")),
        "date": _clean(row.get("DifficultyDate")),
        "make": _clean(row.get("AircraftMake")),
        "model": _clean(row.get("AircraftModel")),
        "part_name": _clean(row.get("PartName")),
        "condition": _clean(row.get("PartCondition")),
        "root_cause_code": _clean(row.get("NatureOfConditionA")),
        "issue_text": _clean(row.get("Discrepancy")),
        "severity_tier": _clean(row.get("SeverityTier")),
        "operator": _clean(row.get("OperatorDesignator")),
        "region": _clean(row.get("ReceivingRegionCode")),
        "recommendation": _clean(row.get("PrecautionaryProcedureA")),
    }


def run_stmt(cypher: str):
    driver = get_driver()
    with driver.session(database=config.NEO4J_DATABASE) as session:
        session.run(cypher)


def main():
    print(f"Loading labeled dataset from {config.LABELED_DATASET_PATH} ...")
    df = pd.read_csv(config.LABELED_DATASET_PATH, low_memory=False)
    df = df[df["OperatorControlNumber"].notna()]
    print(f"  {len(df):,} rows have a control number and will be loaded")

    print("Ensuring constraints exist ...")
    for stmt in CONSTRAINTS:
        run_stmt(stmt)

    total = len(df)
    n_batches = math.ceil(total / BATCH_SIZE)
    driver = get_driver()

    for i in range(n_batches):
        chunk = df.iloc[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        rows = [_row_to_dict(r) for _, r in chunk.iterrows()]
        with driver.session(database=config.NEO4J_DATABASE) as session:
            session.run(MERGE_QUERY, rows=rows)
        print(f"  batch {i + 1}/{n_batches} loaded ({len(rows)} rows)")

    print("Done. Causal-chain knowledge graph built in Neo4j.")
    close_driver()


if __name__ == "__main__":
    main()
