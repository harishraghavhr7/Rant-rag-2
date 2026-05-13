from datetime import date, datetime, timedelta
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
    create_or_get_daily_memory,
    store_memory_embedding,
    add_foods,
    add_tasks,
    get_foods_by_date,
    get_foods_in_date_range,
    get_tasks_by_status,
    get_tasks_in_date_range,
    get_all_daily_memories,
)
from extraction import extract_facts
from faiss_index import get_faiss_manager

class ChatRequest(BaseModel):
    user_id: str
    session_id: Optional[int] = None
    message: str


class QueryRequest(BaseModel):
    user_id: str
    query: str


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


# ========== EMBEDDING & SIMILARITY ==========
def get_embedding(text: str) -> list:
    """Get embedding from Ollama."""
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
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return []


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    if not a or not b:
        return 0.0
    a = np.array(a)
    b = np.array(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


# ========== LLM RESPONSES ==========
def generate_response(prompt: str) -> str:
    """Generate response from LLM."""
    try:
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
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Sorry, I couldn't generate a response."


def summarize_text(text: str) -> str:
    """Summarize a conversation."""
    prompt = f"""
    Summarize this conversation in 1-2 lines:

    {text}
    """
    return generate_response(prompt)


# ========== QUERY CLASSIFICATION ==========
def classify_query(query: str) -> str:
    """
    Classify query into one of:
    - FACTUAL_FOOD
    - FACTUAL_TASKS_COMPLETED
    - FACTUAL_TASKS_PENDING
    - SEMANTIC
    """
    query_lower = query.lower()
    
    # Food keywords
    food_keywords = ["food", "eat", "breakfast", "lunch", "dinner", "meal", "drink", "coffee", "tea", "ate"]
    if any(kw in query_lower for kw in food_keywords):
        return "FACTUAL_FOOD"
    
    # Completed tasks keywords
    completed_keywords = ["completed", "finished", "done", "accomplished", "implemented"]
    task_keywords = ["task", "work", "project", "feature"]
    if any(kw in query_lower for kw in completed_keywords) and any(kw in query_lower for kw in task_keywords):
        return "FACTUAL_TASKS_COMPLETED"
    
    # Pending tasks keywords
    pending_keywords = ["pending", "todo", "uncompleted", "incomplete", "remaining", "still need", "need to"]
    if any(kw in query_lower for kw in pending_keywords) and any(kw in query_lower for kw in task_keywords):
        return "FACTUAL_TASKS_PENDING"
    
    # If query asks for "this week" or date range with tasks
    if ("this week" in query_lower or "past" in query_lower or "week" in query_lower) and any(kw in query_lower for kw in task_keywords):
        if "completed" in query_lower or "finished" in query_lower:
            return "FACTUAL_TASKS_COMPLETED"
        else:
            return "FACTUAL_TASKS_PENDING"
    
    # Default to semantic
    return "SEMANTIC"


# ========== QUERY HANDLERS ==========
def handle_food_query(user_id: str, query: str) -> dict:
    """Handle food/diet queries."""
    # Extract date from query (simplified - looks for "May 5" or "2026-05-05" patterns)
    import re
    
    # Try to find date in query
    date_patterns = [
        r'(\d{4}-\d{2}-\d{2})',  # 2026-05-05
        r'([A-Za-z]+\s+\d{1,2})',  # May 5
        r'(\d{1,2}\s+[A-Za-z]+)',  # 5 May
    ]
    
    target_date = None
    for pattern in date_patterns:
        match = re.search(pattern, query)
        if match:
            target_date = match.group(1)
            break
    
    if target_date:
        # Parse date (simplified)
        try:
            if "-" in target_date:
                target_date = target_date  # Already ISO format
            else:
                # Try to parse "May 5" format
                parsed = datetime.strptime(target_date + " 2026", "%b %d %Y")
                target_date = parsed.date().isoformat()
        except:
            pass
        
        foods = get_foods_by_date(user_id, target_date)
        if foods:
            context = f"On {target_date}, you ate: {', '.join(foods)}"
            return {
                "type": "FACTUAL",
                "context": context,
                "foods": foods
            }
    
    # If no specific date, get recent foods
    end_date = date.today().isoformat()
    start_date = (date.today() - timedelta(days=30)).isoformat()
    
    foods_by_date = get_foods_in_date_range(user_id, start_date, end_date)
    if foods_by_date:
        context_lines = []
        for d, foods in sorted(foods_by_date.items(), reverse=True)[:7]:
            context_lines.append(f"On {d}: {', '.join(foods)}")
        context = "\n".join(context_lines) if context_lines else "No food records found."
        
        return {
            "type": "FACTUAL",
            "context": context,
            "foods_by_date": foods_by_date
        }
    
    return {
        "type": "FACTUAL",
        "context": "No food records found.",
        "foods": []
    }


def handle_tasks_query(user_id: str, query: str, status: str) -> dict:
    """Handle task queries (completed/pending)."""
    import re
    
    # Check for date range (week, month, etc)
    if "week" in query.lower():
        start_date = (date.today() - timedelta(days=7)).isoformat()
        end_date = date.today().isoformat()
    elif "month" in query.lower():
        start_date = (date.today() - timedelta(days=30)).isoformat()
        end_date = date.today().isoformat()
    else:
        # Default: get all
        start_date = "2000-01-01"
        end_date = date.today().isoformat()
    
    tasks = get_tasks_in_date_range(user_id, start_date, end_date, status=status)
    
    if tasks:
        context_lines = []
        for task in tasks:
            context_lines.append(f"[{task['memory_date']}] {task['task_text']}")
        context = "\n".join(context_lines)
        
        return {
            "type": "FACTUAL",
            "context": context,
            "tasks": tasks
        }
    
    status_text = "completed" if status == "completed" else "pending"
    return {
        "type": "FACTUAL",
        "context": f"No {status_text} tasks found.",
        "tasks": []
    }


def handle_semantic_query(user_id: str, query: str) -> dict:
    """Handle semantic/open-ended queries using embeddings."""
    # Get all memories with embeddings
    all_memories = get_all_daily_memories(user_id)
    
    if not all_memories:
        return {
            "type": "SEMANTIC",
            "context": "No memories found.",
            "results": []
        }
    
    # Get query embedding
    query_embedding = get_embedding(query)
    if not query_embedding:
        return {
            "type": "SEMANTIC",
            "context": "Could not process query.",
            "results": []
        }
    
    # Use FAISS for fast search
    faiss_manager = get_faiss_manager()
    faiss_results = faiss_manager.search(query_embedding, k=5)
    
    if faiss_results:
        # Convert FAISS results (which use L2 distance) to similarity scores
        results = []
        for memory_id, distance in faiss_results:
            # Find memory details
            memory = next((m for m in all_memories if m["id"] == memory_id), None)
            if memory:
                # Convert L2 distance to similarity (inverse)
                similarity = 1 / (1 + distance)
                results.append({
                    "memory_date": memory["memory_date"],
                    "summary": memory["summary"],
                    "similarity": round(similarity, 3)
                })
        
        context_lines = []
        for r in results[:3]:
            context_lines.append(f"On {r['memory_date']}: {r['summary']}")
        context = "\n".join(context_lines)
        
        return {
            "type": "SEMANTIC",
            "context": context,
            "results": results
        }
    
    return {
        "type": "SEMANTIC",
        "context": "No similar memories found.",
        "results": []
    }


# ========== STARTUP & ENDPOINTS ==========
@app.on_event("startup")
def startup():
    init_db()
    
    # Rebuild FAISS index on startup
    try:
        from db import get_all_daily_memories as db_get_all
        from db import get_conn
        
        def get_all_memories_with_embeddings():
            conn = get_conn()
            rows = conn.execute("""
                SELECT id, memory_date, summary, embedding
                FROM daily_memories
                ORDER BY memory_date DESC
                LIMIT 1000
            """).fetchall()
            conn.close()
            
            result = []
            for row in rows:
                import json
                embedding = None
                if row["embedding"]:
                    try:
                        embedding = json.loads(row["embedding"])
                    except:
                        pass
                
                result.append({
                    "id": row["id"],
                    "memory_date": row["memory_date"],
                    "summary": row["summary"],
                    "embedding": embedding
                })
            
            return result
        
        memories = get_all_memories_with_embeddings()
        if memories:
            faiss_manager = get_faiss_manager()
            faiss_manager.rebuild_from_db(memories)
    except Exception as e:
        print(f"Warning: Could not rebuild FAISS index on startup: {e}")


@app.get("/")
async def home():
    return FileResponse("static/index.html")


@app.post("/chat")
async def chat(req: ChatRequest):
    """Process user message and store structured facts."""
    session_day = date.today().isoformat()
    
    # Get or create session
    session_id = req.session_id
    if session_id is None:
        session_id = get_or_create_session(req.user_id, session_day)
    
    # Generate response
    response = generate_response(req.message)
    conversation = f"User: {req.message}\nBot: {response}"
    summary = summarize_text(conversation)
    
    # Store in legacy tables
    insert_chat(req.user_id, session_id, req.message, response, summary)
    
    # Create/get daily memory
    memory_id = create_or_get_daily_memory(req.user_id, session_day, summary)
    
    # Extract and store facts
    facts = extract_facts(conversation)
    add_foods(memory_id, facts.get("foods", []))
    add_tasks(memory_id, facts.get("tasks", []))
    
    # Generate and store embedding
    embedding = get_embedding(summary)
    if embedding:
        store_memory_embedding(memory_id, embedding)
        
        # Add to FAISS index
        faiss_manager = get_faiss_manager()
        faiss_manager.add_embedding(memory_id, embedding)
        
        # Also store in legacy table for backward compatibility
        store_day_embedding(req.user_id, session_day, summary, embedding)
    
    return {
        "session_id": session_id,
        "session_day": session_day,
        "response": response,
        "summary": summary,
        "extracted_foods": facts.get("foods", []),
        "extracted_tasks": facts.get("tasks", [])
    }


@app.get("/sessions/{user_id}")
async def list_sessions(user_id: str):
    return {"sessions": get_user_sessions(user_id)}


@app.get("/sessions/messages/{session_id}")
async def session_messages(session_id: int):
    return {"messages": get_session_messages(session_id)}


@app.post("/query")
async def query_memories(req: QueryRequest):
    """
    Hybrid query: classifies query and uses appropriate retrieval strategy.
    """
    query_type = classify_query(req.query)
    
    print(f"Query type detected: {query_type}")
    
    # Route based on query type
    if query_type == "FACTUAL_FOOD":
        retrieval = handle_food_query(req.user_id, req.query)
    elif query_type == "FACTUAL_TASKS_COMPLETED":
        retrieval = handle_tasks_query(req.user_id, req.query, "completed")
    elif query_type == "FACTUAL_TASKS_PENDING":
        retrieval = handle_tasks_query(req.user_id, req.query, "pending")
    else:  # SEMANTIC
        retrieval = handle_semantic_query(req.user_id, req.query)
    
    # Generate final answer using LLM
    rag_prompt = f"""
    Based on the following information:

    {retrieval['context']}

    Answer this question: {req.query}

    Keep the answer concise and natural.
    """
    
    answer = generate_response(rag_prompt)
    
    return {
        "query_type": query_type,
        "answer": answer,
        "context": retrieval.get("context", ""),
        "raw_results": retrieval
    }


@app.get("/memory/{user_id}")
async def get_user_memory(user_id: str, limit: int = 30):
    """Get user's memory timeline."""
    memories = get_all_daily_memories(user_id, limit=limit)
    return {
        "memories": memories,
        "count": len(memories)
    }


@app.get("/memory/foods/{user_id}")
async def get_user_foods(user_id: str, days: int = 30):
    """Get foods from past N days."""
    end_date = date.today().isoformat()
    start_date = (date.today() - timedelta(days=days)).isoformat()
    
    foods = get_foods_in_date_range(user_id, start_date, end_date)
    return {
        "foods_by_date": foods,
        "days_with_food": len(foods)
    }


@app.get("/memory/tasks/{user_id}")
async def get_user_tasks(user_id: str):
    """Get pending and completed tasks."""
    completed = get_tasks_by_status(user_id, "completed")
    pending = get_tasks_by_status(user_id, "pending")
    
    return {
        "completed": completed,
        "completed_count": len(completed),
        "pending": pending,
        "pending_count": len(pending)
    }