from __future__ import annotations

"""
db.py — SQLite database operations.

Schema
------
news_items
  id              INTEGER  PK autoincrement
  submitter_name  TEXT     Who submitted the item
  url             TEXT     Link to the source content
  reason          TEXT     Private audit note — why they added it (not published)
  agreed_text     TEXT     The agreed newsletter blurb — the publishable copy
  created_at      TEXT     ISO-8601 UTC timestamp (auto-set on insert)
"""

import sqlite3
from config import settings


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't already exist."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS news_items (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                submitter_name  TEXT    NOT NULL,
                url             TEXT    NOT NULL,
                reason          TEXT    NOT NULL,
                agreed_text     TEXT    NOT NULL,
                created_at      TEXT    NOT NULL
                    DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                done            INTEGER NOT NULL DEFAULT 0
            )
        """)
        # Migrate existing databases that predate the done column
        try:
            conn.execute("ALTER TABLE news_items ADD COLUMN done INTEGER NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass  # column already exists
        conn.commit()


def save_news_item(
    submitter_name: str,
    url: str,
    reason: str,
    agreed_text: str,
) -> dict:
    """Insert a new news item and return the full saved record."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO news_items (submitter_name, url, reason, agreed_text)
            VALUES (?, ?, ?, ?)
            """,
            (
                submitter_name.strip(),
                url.strip(),
                reason.strip(),
                agreed_text.strip(),
            ),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM news_items WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
        return dict(row)


def update_news_item(
    item_id: int,
    submitter_name: str,
    url: str,
    reason: str,
    agreed_text: str,
) -> dict:
    """Update an existing news item and return the updated record."""
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE news_items
            SET submitter_name = ?, url = ?, reason = ?, agreed_text = ?
            WHERE id = ?
            """,
            (submitter_name.strip(), url.strip(), reason.strip(), agreed_text.strip(), item_id),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM news_items WHERE id = ?", (item_id,)
        ).fetchone()
        return dict(row)


def get_items_by_name(submitter_name: str, limit: int = 10) -> list[dict]:
    """Return recent items submitted by a given name (case-insensitive)."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM news_items
            WHERE lower(submitter_name) = lower(?)
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (submitter_name.strip(), limit),
        ).fetchall()
        return [dict(row) for row in rows]


def mark_done(item_id: int, done: bool = True) -> None:
    """Set the done flag on a news item."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE news_items SET done = ? WHERE id = ?",
            (int(done), item_id),
        )
        conn.commit()


def get_feed(limit: int = 50, include_done: bool = True) -> list[dict]:
    """Return the most recent news items, newest first."""
    with get_connection() as conn:
        if include_done:
            rows = conn.execute(
                "SELECT * FROM news_items ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM news_items WHERE done = 0 ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]


def delete_news_item(item_id: int) -> None:
    """Permanently delete a news item by ID."""
    with get_connection() as conn:
        conn.execute("DELETE FROM news_items WHERE id = ?", (item_id,))
        conn.commit()


def get_item_count() -> int:
    """Return the total number of saved news items."""
    with get_connection() as conn:
        return conn.execute(
            "SELECT COUNT(*) AS n FROM news_items"
        ).fetchone()["n"]
