import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB_PATH = Path("professors.db")


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS professors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            title TEXT,
            school TEXT NOT NULL,
            faculty TEXT,
            department TEXT,
            email TEXT,
            phone TEXT,
            profile_url TEXT UNIQUE NOT NULL,
            expertise_raw TEXT,
            expertise_keywords TEXT,
            embedding BLOB,
            scraped_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def insert_professor(db_path: Path, prof: dict) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """INSERT OR IGNORE INTO professors
            (name, title, school, faculty, department, email, phone,
             profile_url, expertise_raw, expertise_keywords, scraped_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                prof["name"],
                prof.get("title"),
                prof["school"],
                prof.get("faculty"),
                prof.get("department"),
                prof.get("email"),
                prof.get("phone"),
                prof["profile_url"],
                prof.get("expertise_raw"),
                json.dumps(prof.get("expertise_keywords", [])),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_professor_by_url(db_path: Path, url: str) -> dict | None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute("SELECT * FROM professors WHERE profile_url = ?", (url,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_professors(db_path: Path) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM professors").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_professors_without_embeddings(db_path: Path) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM professors WHERE embedding IS NULL").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_professor_expertise(
    db_path: Path, profile_url: str, expertise_raw: str,
    expertise_keywords: list, rich_profile_url: str | None = None,
) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """UPDATE professors
        SET expertise_raw = ?, expertise_keywords = ?, embedding = NULL
        WHERE profile_url = ?""",
        (expertise_raw, json.dumps(expertise_keywords), profile_url),
    )
    if rich_profile_url:
        # Check if the rich URL already exists (duplicate entry)
        existing = conn.execute(
            "SELECT id FROM professors WHERE profile_url = ?", (rich_profile_url,)
        ).fetchone()
        if existing:
            # Rich profile already in DB — delete the thin duplicate instead
            conn.execute("DELETE FROM professors WHERE profile_url = ?", (profile_url,))
        else:
            conn.execute(
                "UPDATE professors SET profile_url = ? WHERE profile_url = ?",
                (rich_profile_url, profile_url),
            )
    conn.commit()
    conn.close()


def update_embedding(db_path: Path, profile_url: str, embedding) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE professors SET embedding = ? WHERE profile_url = ?",
        (embedding.tobytes(), profile_url),
    )
    conn.commit()
    conn.close()


def clear_embeddings(db_path: Path) -> int:
    """Set all embeddings to NULL. Returns the number of rows cleared."""
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("UPDATE professors SET embedding = NULL")
    count = cursor.rowcount
    conn.commit()
    conn.close()
    return count
