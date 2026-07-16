"""CLI: build (or rebuild) the local Chroma vector store used by the
Root-Cause RAG agent.

Usage:
    python scripts/build_vectorstore.py            # build if missing
    python scripts/build_vectorstore.py --rebuild   # force rebuild
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.genai.vectorstore import build_vectorstore

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild even if a store already exists")
    args = parser.parse_args()
    build_vectorstore(force_rebuild=args.rebuild)
