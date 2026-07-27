// src/config.js
// Mirrors the values in your Python .env — since React can't read a
// .env from a different (Python) app, these are just for display in
// the sidebar. Update if you change your Ollama model names.
export default {
  OLLAMA_CHAT_MODEL: "llama3.1:8b",
  OLLAMA_EMBED_MODEL: "nomic-embed-text",
};
