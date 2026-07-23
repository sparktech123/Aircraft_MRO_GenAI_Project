"""
build_kg.py
============
Builds the FAA SDR knowledge graph directly inside your Neo4j Aura
instance, from the same labeled dataset your ML pipeline already uses.
Mirrors scripts/build_vectorstore.py's role (one-time / re-run-when-data-
changes build step), but populates Neo4j instead of Chroma.

Run from the project root (genai_project/):
    python scripts/build_kg.py

Safe to re-run: every write uses MERGE, so re-running just refreshes
data rather than duplicating nodes.
"""
import math
import pandas as pd

import config
from src.genai.graph_db import get_driver, close_driver

BATCH_SIZE = 500

# Constraints make MERGE fast and prevent duplicate nodes.
CONSTRAINTS = [
    "CREATE CONSTRAINT discrepancy_id IF NOT EXISTS FOR (d:Discrepancy) REQUIRE d.control_number IS UNIQUE",
    "CREATE CONSTRAINT aircraft_id IF NOT EXISTS FOR (a:Aircraft) REQUIRE (a.make, a.model) IS NODE KEY",
    "CREATE CONSTRAINT part_id IF NOT EXISTS FOR (p:Part) REQUIRE p.name IS UNIQUE",
    "CREATE CONSTRAINT system_id IF NOT EXISTS FOR (s:System) REQUIRE s.jasc_code IS UNIQUE",
    "CREATE CONSTRAINT condition_id IF NOT EXISTS FOR (c:Condition) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT operator_id IF NOT EXISTS FOR (o:Operator) REQUIRE o.name IS UNIQUE",
]

# One UNWIND-based MERGE query per batch handles every node/edge type for
# that batch of rows in a single round trip — much faster than one query
# per row against a remote Aura instance.
MERGE_QUERY = """
UNWIND $rows AS row

MERGE (d:Discrepancy {control_number: row.control_number})
SET d.date = row.date,
    d.text = row.text,
    d.precautionary_procedure = row.precautionary_procedure,
    d.stage_of_operation = row.stage_of_operation,
    d.severity_tier = row.severity_tier,
    d.critical = row.critical,
    d.risk_score = row.risk_score

WITH row, d
WHERE row.aircraft_make IS NOT NULL OR row.aircraft_model IS NOT NULL
MERGE (a:Aircraft {make: row.aircraft_make, model: row.aircraft_model})
MERGE (a)-[:HAS_DISCREPANCY]->(d)

WITH row, d
WHERE row.part_name IS NOT NULL
MERGE (p:Part {name: row.part_name})
SET p.condition = row.part_condition
MERGE (d)-[:INVOLVES_PART]->(p)
WITH row, d, p
WHERE row.aircraft_make IS NOT NULL OR row.aircraft_model IS NOT NULL
MERGE (a2:Aircraft {make: row.aircraft_make, model: row.aircraft_model})
MERGE (p)-[:INSTALLED_ON]->(a2)

WITH row, d
WHERE row.part_name IS NOT NULL AND row.jasc_code IS NOT NULL
MERGE (p3:Part {name: row.part_name})
MERGE (s:System {jasc_code: row.jasc_code})
MERGE (p3)-[:PART_OF_SYSTEM]->(s)

WITH row, d
WHERE row.condition IS NOT NULL
MERGE (c:Condition {name: row.condition})
MERGE (d)-[:HAS_CONDITION]->(c)

WITH row, d
WHERE row.operator IS NOT NULL
MERGE (o:Operator {name: row.operator})
MERGE (o)-[:FILED]->(d)
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
        "text": _clean(row.get("Discrepancy")),
        "precautionary_procedure": _clean(row.get("PrecautionaryProcedureA")),
        "stage_of_operation": _clean(row.get("StageOfOperationCode")),
        "severity_tier": _clean(row.get("SeverityTier")),
        "critical": bool(row["Critical"]) if _clean(row.get("Critical")) is not None else None,
        "risk_score": float(row["RiskScore"]) if _clean(row.get("RiskScore")) is not None else None,
        "aircraft_make": _clean(row.get("AircraftMake")),
        "aircraft_model": _clean(row.get("AircraftModel")),
        "part_name": _clean(row.get("PartName")),
        "part_condition": _clean(row.get("PartCondition")),
        "jasc_code": _clean(row.get("JASCCode")),
        "condition": _clean(row.get("NatureOfConditionA")),
        "operator": _clean(row.get("OperatorDesignator")),
    }


def main():
    print(f"Loading labeled dataset from {config.LABELED_DATASET_PATH} ...")
    df = pd.read_csv(config.LABELED_DATASET_PATH, low_memory=False)
    print(f"  {len(df):,} rows")

    # Drop rows with no usable ID — can't MERGE a Discrepancy without one.
    df = df[df["OperatorControlNumber"].notna()]
    print(f"  {len(df):,} rows have a control number and will be loaded")

    print("Ensuring constraints exist ...")
    for stmt in CONSTRAINTS:
        run_query_stmt(stmt)

    total = len(df)
    n_batches = math.ceil(total / BATCH_SIZE)
    driver = get_driver()

    for i in range(n_batches):
        chunk = df.iloc[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        rows = [_row_to_dict(r) for _, r in chunk.iterrows()]
        with driver.session(database=config.NEO4J_DATABASE) as session:
            session.run(MERGE_QUERY, rows=rows)
        print(f"  batch {i + 1}/{n_batches} loaded ({len(rows)} rows)")

    print("Done. Knowledge graph built in Neo4j.")
    close_driver()


def run_query_stmt(cypher: str):
    driver = get_driver()
    with driver.session(database=config.NEO4J_DATABASE) as session:
        session.run(cypher)


if __name__ == "__main__":
    main()
