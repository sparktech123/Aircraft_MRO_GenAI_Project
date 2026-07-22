"""
graph_store.py
===============
Load/access helper for the persisted knowledge graph. Mirrors the role
of src/genai/vectorstore.py (which loads the Chroma DB for the RAG
agent) but for the graph used by the new Graph agent.
"""
import pickle
import networkx as nx

from config import DATA_PROCESSED_DIR

KG_GRAPH_PATH = DATA_PROCESSED_DIR / "kg_graph.gpickle"

_graph_cache = None


def load_graph() -> nx.MultiDiGraph:
    """Loads the graph once per process and caches it in memory."""
    global _graph_cache
    if _graph_cache is None:
        if not KG_GRAPH_PATH.exists():
            raise FileNotFoundError(
                f"{KG_GRAPH_PATH} not found. Run `python scripts/build_graph.py` first."
            )
        with open(KG_GRAPH_PATH, "rb") as f:
            _graph_cache = pickle.load(f)
    return _graph_cache
