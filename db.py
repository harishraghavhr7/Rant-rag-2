import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path("data") / "chats.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = get_conn()

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

    conn.commit()
    conn.close()


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
    import json
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
    import json
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