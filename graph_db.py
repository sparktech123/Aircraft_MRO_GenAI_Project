"""
graph_db.py
============
Neo4j Aura driver singleton + a small run_query helper. Mirrors the role
of vectorstore.py (which loads the Chroma DB for the RAG agent) but for
the Neo4j-backed knowledge graph used by the new Graph agent.

Requires these in your .env (see NEO4J_* config in config.py):
    NEO4J_URI=neo4j+s://<your-instance>.databases.neo4j.io
    NEO4J_USERNAME=<your-instance-id>
    NEO4J_PASSWORD=<your-password>
    NEO4J_DATABASE=<your-instance-id>
"""
from neo4j import GraphDatabase

import config

_driver = None


def get_driver():
    """Returns a cached Neo4j driver, creating it on first use."""
    global _driver
    if _driver is None:
        if not config.NEO4J_URI:
            raise RuntimeError(
                "NEO4J_URI is not set. Add NEO4J_URI / NEO4J_USERNAME / "
                "NEO4J_PASSWORD / NEO4J_DATABASE to your .env file."
            )
        _driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD),
        )
    return _driver


def run_query(cypher: str, params: dict | None = None) -> list[dict]:
    """Runs a Cypher query against the configured database and returns a
    list of plain dicts (one per result row)."""
    driver = get_driver()
    with driver.session(database=config.NEO4J_DATABASE) as session:
        result = session.run(cypher, params or {})
        return [record.data() for record in result]


def close_driver():
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None
