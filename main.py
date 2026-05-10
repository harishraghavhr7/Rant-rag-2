from datetime import date
from typing import Optional
import numpy as np
import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from db import (
    init_db,
    insert_chat,
    get_user_sessions,
    get_session_messages,
    get_or_create_session,
    store_day_embedding,
    get_day_embeddings,
)

class ChatRequest(BaseModel):
    user_id: str
    session_id: Optional[int] = None
    message: str


class QueryRequest(BaseModel):
    user_id: str
    query: str


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


def get_embedding(text: str) -> list:
    """Get embedding from Ollama (or swap with another embedding model)"""
    try:
        res = requests.post(
            "http://localhost:11434/api/embeddings",
            json={
                "model": "nomic-embed-text",
                "prompt": text
            },
            timeout=30
        )
        return res.json().get("embedding", [])
    except:
        return []


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors"""
    if not a or not b:
        return 0.0
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def summarize_text(text):
    prompt = f"""
    Summarize this conversation in 1-2 lines:

    {text}
    """
    return generate_response(prompt)


def generate_response(prompt):
    res = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "gemma3:1b",
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    print("STATUS:", res.status_code)
    print("RAW RESPONSE:", res.text)

    return res.json().get("response", "")


@app.on_event("startup")
def startup():
    init_db()


@app.get("/")
async def home():
    return FileResponse("static/index.html")


@app.post("/chat")
async def chat(req: ChatRequest):
    session_day = date.today().isoformat()

    session_id = req.session_id
    if session_id is None:
        session_id = get_or_create_session(req.user_id, session_day)

    response = generate_response(req.message)
    conversation = f"User: {req.message}\nBot: {response}"
    summary = summarize_text(conversation)

    insert_chat(req.user_id, session_id, req.message, response, summary)

    embedding = get_embedding(summary)
    if embedding:
        store_day_embedding(req.user_id, session_day, summary, embedding)

    return {
        "session_id": session_id,
        "session_day": session_day,
        "response": response,
        "summary": summary
    }


@app.get("/sessions/{user_id}")
async def list_sessions(user_id: str):
    return {"sessions": get_user_sessions(user_id)}


@app.get("/sessions/messages/{session_id}")
async def session_messages(session_id: int):
    return {"messages": get_session_messages(session_id)}


@app.post("/query")
async def query_days(req: QueryRequest):
    """
    RAG: Query past days by semantic similarity.
    Returns relevant day summaries + LLM answer.
    """
    day_embeddings = get_day_embeddings(req.user_id)

    if not day_embeddings:
        return {
            "answer": "No conversation history found.",
            "context": []
        }

    query_embedding = get_embedding(req.query)
    if not query_embedding:
        return {
            "answer": "Could not process query.",
            "context": []
        }

    similarities = []
    for day_data in day_embeddings:
        sim = cosine_similarity(query_embedding, day_data["embedding"])
        similarities.append({
            "day": day_data["session_day"],
            "summary": day_data["summary"],
            "similarity": sim
        })

    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    top_results = similarities[:3]

    context = "\n".join([
        f"On {r['day']}: {r['summary']}"
        for r in top_results
    ])

    rag_prompt = f"""
    Based on this conversation history:

    {context}

    Answer this question: {req.query}

    Keep the answer concise.
    """

    answer = generate_response(rag_prompt)

    return {
        "answer": answer,
        "context": [
            {
                "day": r["day"],
                "summary": r["summary"],
                "similarity": round(r["similarity"], 3)
            }
            for r in top_results
        ]
    }