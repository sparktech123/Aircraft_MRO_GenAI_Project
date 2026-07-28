"""
main.py — FastAPI backend for the FAA SDR Assistant.

Wraps your EXISTING pipeline (src/genai/graph.py) and Task 4/5 artifacts
behind a REST API for the React frontend.

Run from the project root:
    uvicorn backend.main:app --reload --port 8000
"""
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from src.genai.graph import build_graph
from backend.dashboard_routes import router as dashboard_router

app = FastAPI(title="FAA SDR Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dashboard_router)

# Serves the PNGs saved by 01_train_model.py / 02_predict_2026.py so
# React can just <img src="http://localhost:8000/plots/xyz.png" />
PLOTS_DIR = os.environ.get("PLOTS_DIR", os.path.join("outputs", "plots"))
if os.path.isdir(PLOTS_DIR):
    app.mount("/plots", StaticFiles(directory=PLOTS_DIR), name="plots")

_graph_app = None


def get_graph_app():
    global _graph_app
    if _graph_app is None:
        _graph_app = build_graph()
    return _graph_app


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    chat_history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    final_answer: str
    routing_reason: str | None = None
    trace: dict[str, str] = {}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    graph_app = get_graph_app()
    history_tuples = [(m.role, m.content) for m in req.chat_history]

    result = graph_app.invoke({
        "question": req.question,
        "chat_history": history_tuples,
    })

    trace = {}
    for key, label in [
        ("data_agent_output", "Data Analytics agent"),
        ("rag_agent_output", "Root-Cause (RAG) agent"),
        ("risk_agent_output", "Risk / Early-Warning agent"),
        ("graph_agent_output", "Knowledge Graph agent"),
    ]:
        if result.get(key):
            trace[label] = result[key]

    return ChatResponse(
        final_answer=result.get("final_answer", "I couldn't produce an answer."),
        routing_reason=result.get("routing_reason"),
        trace=trace,
    )


@app.get("/api/alerts")
def get_alerts():
    if not config.ALERTS_PATH.exists():
        raise HTTPException(status_code=404, detail="Alerts file not found. Run the Task 5 pipeline first.")
    df = pd.read_csv(config.ALERTS_PATH)
    return df.to_dict(orient="records")


@app.get("/api/monthly-trend")
def get_monthly_trend():
    if not config.MONTHLY_TREND_PATH.exists():
        raise HTTPException(status_code=404, detail="Monthly trend file not found. Run the Task 5 pipeline first.")
    df = pd.read_csv(config.MONTHLY_TREND_PATH)
    return df.to_dict(orient="records")


@app.get("/api/health")
def health():
    return {"status": "ok"}
