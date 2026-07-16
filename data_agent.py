# Copy this file to .env and adjust if needed. All defaults already
# point at a local, free Ollama install — no API keys required anywhere
# in this project.

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=llama3.1:8b
OLLAMA_EMBED_MODEL=nomic-embed-text
LLM_TEMPERATURE=0.1

# Cap on how many unique discrepancy narratives get embedded into the
# vector store (local CPU embedding is slow — raise this once you've
# confirmed throughput, or set to -1 for no cap).
VECTORSTORE_MAX_DOCS=15000
