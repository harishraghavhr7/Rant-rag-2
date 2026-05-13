import sqlite3
from datetime import datetime
from pathlib import Path
import json

DB_PATH = Path("data") / "chats.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = get_conn()

    # Keep legacy tables for backward compatibility
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            session_day TEXT NOT NULL,
            title TEXT,
            summary TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(user_id, session_day)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            user_message TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES chat_sessions (id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS day_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            session_day TEXT NOT NULL,
            summary TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(user_id, session_day)
        )
    """)

    # NEW: Core memory table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            memory_date TEXT NOT NULL,
            summary TEXT NOT NULL,
            embedding BLOB,
            created_at TEXT NOT NULL,
            UNIQUE(user_id, memory_date)
        )
    """)

    # NEW: Foods table (structured facts)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS foods (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            daily_memory_id INTEGER NOT NULL,
            food_name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (daily_memory_id) REFERENCES daily_memories (id)
        )
    """)

    # NEW: Tasks table (structured facts)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            daily_memory_id INTEGER NOT NULL,
            task_text TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            FOREIGN KEY (daily_memory_id) REFERENCES daily_memories (id)
        )
    """)

    conn.commit()
    conn.close()


# ========== Legacy Functions (backward compatible) ==========
def create_session(user_id: str, session_day: str, title: str = "New day chat", summary: str = ""):
    conn = get_conn()
    now = datetime.now().isoformat()

    cur = conn.execute("""
        INSERT INTO chat_sessions (user_id, session_day, title, summary, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (user_id, session_day, title, summary, now, now))

    conn.commit()
    session_id = cur.lastrowid
    conn.close()
    return session_id


def get_or_create_session(user_id: str, session_day: str):
    conn = get_conn()
    row = conn.execute("""
        SELECT id
        FROM chat_sessions
        WHERE user_id = ? AND session_day = ?
    """, (user_id, session_day)).fetchone()
    conn.close()

    if row:
        return row["id"]

    return create_session(user_id, session_day)


def insert_chat(user_id: str, session_id: int, user_message: str, bot_response: str, summary: str = None):
    conn = get_conn()
    now = datetime.now().isoformat()

    conn.execute("""
        INSERT INTO chat_messages (session_id, user_message, bot_response, created_at)
        VALUES (?, ?, ?, ?)
    """, (session_id, user_message, bot_response, now))

    title = user_message[:40] if user_message else "New day chat"

    if summary is not None:
        conn.execute("""
            UPDATE chat_sessions
            SET summary = ?, updated_at = ?
            WHERE id = ?
        """, (summary, now, session_id))

    row = conn.execute("""
        SELECT title
        FROM chat_sessions
        WHERE id = ?
    """, (session_id,)).fetchone()

    if row and (row["title"] is None or row["title"] == "New day chat"):
        conn.execute("""
            UPDATE chat_sessions
            SET title = ?, updated_at = ?
            WHERE id = ?
        """, (title, now, session_id))

    conn.execute("""
        UPDATE chat_sessions
        SET updated_at = ?
        WHERE id = ?
    """, (now, session_id))

    conn.commit()
    conn.close()


def store_day_embedding(user_id: str, session_day: str, summary: str, embedding: list):
    conn = get_conn()
    now = datetime.now().isoformat()

    embedding_blob = json.dumps(embedding)

    conn.execute("""
        INSERT OR REPLACE INTO day_embeddings (user_id, session_day, summary, embedding, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, session_day, summary, embedding_blob, now))

    conn.commit()
    conn.close()


def get_day_embeddings(user_id: str):
    conn = get_conn()
    rows = conn.execute("""
        SELECT session_day, summary, embedding
        FROM day_embeddings
        WHERE user_id = ?
        ORDER BY session_day DESC
    """, (user_id,)).fetchall()
    conn.close()

    result = []
    for row in rows:
        result.append({
            "session_day": row["session_day"],
            "summary": row["summary"],
            "embedding": json.loads(row["embedding"])
        })
    return result


def get_user_sessions(user_id: str):
    conn = get_conn()
    rows = conn.execute("""
        SELECT *
        FROM chat_sessions
        WHERE user_id = ?
        ORDER BY session_day DESC, updated_at DESC
    """, (user_id,)).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_session_messages(session_id: int):
    conn = get_conn()
    rows = conn.execute("""
        SELECT *
        FROM chat_messages
        WHERE session_id = ?
        ORDER BY created_at ASC
    """, (session_id,)).fetchall()
    conn.close()
    return [dict(row) for row in rows]


# ========== NEW: Structured Memory Functions ==========
def create_or_get_daily_memory(user_id: str, memory_date: str, summary: str):
    """Create or retrieve daily memory record."""
    conn = get_conn()
    now = datetime.now().isoformat()
    
    row = conn.execute("""
        SELECT id FROM daily_memories
        WHERE user_id = ? AND memory_date = ?
    """, (user_id, memory_date)).fetchone()
    
    if row:
        memory_id = row["id"]
        conn.close()
        return memory_id
    
    cur = conn.execute("""
        INSERT INTO daily_memories (user_id, memory_date, summary, created_at)
        VALUES (?, ?, ?, ?)
    """, (user_id, memory_date, summary, now))
    
    conn.commit()
    memory_id = cur.lastrowid
    conn.close()
    return memory_id


def store_memory_embedding(daily_memory_id: int, embedding: list):
    """Store embedding for a daily memory."""
    conn = get_conn()
    embedding_blob = json.dumps(embedding)
    
    conn.execute("""
        UPDATE daily_memories
        SET embedding = ?
        WHERE id = ?
    """, (embedding_blob, daily_memory_id))
    
    conn.commit()
    conn.close()


def add_foods(daily_memory_id: int, foods: list):
    """Add foods extracted for a daily memory."""
    conn = get_conn()
    now = datetime.now().isoformat()
    
    # Clear existing foods for this memory
    conn.execute("DELETE FROM foods WHERE daily_memory_id = ?", (daily_memory_id,))
    
    # Insert new foods
    for food in foods:
        if food.strip():  # Skip empty strings
            conn.execute("""
                INSERT INTO foods (daily_memory_id, food_name, created_at)
                VALUES (?, ?, ?)
            """, (daily_memory_id, food.strip(), now))
    
    conn.commit()
    conn.close()


def add_tasks(daily_memory_id: int, tasks: list):
    """Add tasks extracted for a daily memory.
    
    tasks: list of dicts with 'text' and 'status' keys
    """
    conn = get_conn()
    now = datetime.now().isoformat()
    
    # Clear existing tasks for this memory
    conn.execute("DELETE FROM tasks WHERE daily_memory_id = ?", (daily_memory_id,))
    
    # Insert new tasks
    for task in tasks:
        if task.get("text", "").strip():
            status = task.get("status", "pending").lower()
            if status not in ("pending", "completed"):
                status = "pending"
            
            conn.execute("""
                INSERT INTO tasks (daily_memory_id, task_text, status, created_at)
                VALUES (?, ?, ?, ?)
            """, (daily_memory_id, task["text"].strip(), status, now))
    
    conn.commit()
    conn.close()


def get_foods_by_date(user_id: str, memory_date: str):
    """Retrieve foods for a specific date."""
    conn = get_conn()
    rows = conn.execute("""
        SELECT f.food_name
        FROM foods f
        JOIN daily_memories d ON f.daily_memory_id = d.id
        WHERE d.user_id = ? AND d.memory_date = ?
        ORDER BY f.created_at ASC
    """, (user_id, memory_date)).fetchall()
    conn.close()
    
    return [row["food_name"] for row in rows]


def get_foods_in_date_range(user_id: str, start_date: str, end_date: str):
    """Retrieve foods within a date range."""
    conn = get_conn()
    rows = conn.execute("""
        SELECT d.memory_date, f.food_name
        FROM foods f
        JOIN daily_memories d ON f.daily_memory_id = d.id
        WHERE d.user_id = ? AND d.memory_date >= ? AND d.memory_date <= ?
        ORDER BY d.memory_date DESC, f.created_at ASC
    """, (user_id, start_date, end_date)).fetchall()
    conn.close()
    
    result = {}
    for row in rows:
        if row["memory_date"] not in result:
            result[row["memory_date"]] = []
        result[row["memory_date"]].append(row["food_name"])
    
    return result


def get_tasks_by_status(user_id: str, status: str):
    """Retrieve tasks with specific status (pending/completed)."""
    conn = get_conn()
    rows = conn.execute("""
        SELECT d.memory_date, t.task_text
        FROM tasks t
        JOIN daily_memories d ON t.daily_memory_id = d.id
        WHERE d.user_id = ? AND t.status = ?
        ORDER BY d.memory_date DESC, t.created_at ASC
    """, (user_id, status.lower())).fetchall()
    conn.close()
    
    result = []
    for row in rows:
        result.append({
            "task_text": row["task_text"],
            "memory_date": row["memory_date"]
        })
    
    return result


def get_tasks_in_date_range(user_id: str, start_date: str, end_date: str, status: str = None):
    """Retrieve tasks within date range, optionally filtered by status."""
    conn = get_conn()
    
    if status:
        rows = conn.execute("""
            SELECT d.memory_date, t.task_text, t.status
            FROM tasks t
            JOIN daily_memories d ON t.daily_memory_id = d.id
            WHERE d.user_id = ? AND d.memory_date >= ? AND d.memory_date <= ? AND t.status = ?
            ORDER BY d.memory_date DESC, t.created_at ASC
        """, (user_id, start_date, end_date, status.lower())).fetchall()
    else:
        rows = conn.execute("""
            SELECT d.memory_date, t.task_text, t.status
            FROM tasks t
            JOIN daily_memories d ON t.daily_memory_id = d.id
            WHERE d.user_id = ? AND d.memory_date >= ? AND d.memory_date <= ?
            ORDER BY d.memory_date DESC, t.created_at ASC
        """, (user_id, start_date, end_date)).fetchall()
    
    conn.close()
    
    result = []
    for row in rows:
        result.append({
            "task_text": row["task_text"],
            "memory_date": row["memory_date"],
            "status": row["status"]
        })
    
    return result


def get_all_daily_memories(user_id: str, limit: int = 100):
    """Retrieve all daily memories for a user."""
    conn = get_conn()
    rows = conn.execute("""
        SELECT id, memory_date, summary, embedding
        FROM daily_memories
        WHERE user_id = ?
        ORDER BY memory_date DESC
        LIMIT ?
    """, (user_id, limit)).fetchall()
    conn.close()
    
    result = []
    for row in rows:
        embedding = None
        if row["embedding"]:
            embedding = json.loads(row["embedding"])
        
        result.append({
            "id": row["id"],
            "memory_date": row["memory_date"],
            "summary": row["summary"],
            "embedding": embedding
        })
    
    return result