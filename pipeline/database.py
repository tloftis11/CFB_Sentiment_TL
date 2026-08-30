import os
import sqlite3
import logging
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

# On Render, set DATA_DIR=/data (persistent disk). Locally defaults to data/ subdir.
_data_dir = Path(os.getenv("DATA_DIR", str(Path(__file__).parent.parent / "data")))
DB_PATH = _data_dir / "cfb_sentiment.db"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _add_column_if_missing(conn: sqlite3.Connection, column: str, coltype: str):
    try:
        conn.execute(f"ALTER TABLE daily_rankings ADD COLUMN {column} {coltype}")
    except Exception:
        pass  # column already exists


def init_db():
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS teams (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                school   TEXT UNIQUE NOT NULL,
                mascot   TEXT,
                conference TEXT,
                division TEXT
            );

            CREATE TABLE IF NOT EXISTS daily_rankings (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                school             TEXT    NOT NULL,
                conference         TEXT,
                run_date           DATE    NOT NULL,

                -- Quality inputs
                sp_rating          REAL,
                win_pct            REAL,
                games_played       INTEGER,

                -- Sentiment inputs
                ap_rank            INTEGER,
                google_trends_score REAL,
                wikipedia_score    REAL,
                reddit_score       REAL,
                recruiting_rank    INTEGER,

                -- Composite scores (0-100)
                quality_score      REAL,
                sentiment_score    REAL,
                divergence_score   REAL,
                divergence_label   TEXT,

                -- Rank positions (lower = better except divergence: higher = more overrated)
                quality_rank       INTEGER,
                sentiment_rank     INTEGER,
                divergence_rank    INTEGER,

                UNIQUE(school, run_date)
            );

            CREATE INDEX IF NOT EXISTS idx_rankings_date   ON daily_rankings(run_date);
            CREATE INDEX IF NOT EXISTS idx_rankings_school ON daily_rankings(school);
            CREATE INDEX IF NOT EXISTS idx_rankings_div    ON daily_rankings(divergence_score);
        """)
        # Migrate existing databases that predate the wikipedia/reddit columns
        _add_column_if_missing(conn, "wikipedia_score", "REAL")
        _add_column_if_missing(conn, "reddit_score",    "REAL")
    logger.info(f"Database initialized at {DB_PATH}")


def upsert_ranking(row: dict):
    sql = """
        INSERT INTO daily_rankings
            (school, conference, run_date, sp_rating, win_pct, games_played,
             ap_rank, google_trends_score, recruiting_rank,
             quality_score, sentiment_score, divergence_score, divergence_label,
             quality_rank, sentiment_rank, divergence_rank)
        VALUES
            (:school, :conference, :run_date, :sp_rating, :win_pct, :games_played,
             :ap_rank, :google_trends_score, :recruiting_rank,
             :quality_score, :sentiment_score, :divergence_score, :divergence_label,
             :quality_rank, :sentiment_rank, :divergence_rank)
        ON CONFLICT(school, run_date) DO UPDATE SET
            conference          = excluded.conference,
            sp_rating           = excluded.sp_rating,
            win_pct             = excluded.win_pct,
            games_played        = excluded.games_played,
            ap_rank             = excluded.ap_rank,
            google_trends_score = excluded.google_trends_score,
            recruiting_rank     = excluded.recruiting_rank,
            quality_score       = excluded.quality_score,
            sentiment_score     = excluded.sentiment_score,
            divergence_score    = excluded.divergence_score,
            divergence_label    = excluded.divergence_label,
            quality_rank        = excluded.quality_rank,
            sentiment_rank      = excluded.sentiment_rank,
            divergence_rank     = excluded.divergence_rank
    """
    with get_connection() as conn:
        conn.execute(sql, row)


def get_latest_rankings(
    divergence_filter: str = None,
    conference_filter: str = None,
    limit: int = None,
) -> list[dict]:
    with get_connection() as conn:
        query = """
            SELECT * FROM daily_rankings
            WHERE run_date = (SELECT MAX(run_date) FROM daily_rankings)
        """
        params = []
        if divergence_filter:
            query += " AND divergence_label = ?"
            params.append(divergence_filter)
        if conference_filter:
            query += " AND conference = ?"
            params.append(conference_filter)
        query += " ORDER BY divergence_score DESC"
        if limit:
            query += f" LIMIT {limit}"

        return [dict(r) for r in conn.execute(query, params).fetchall()]


def get_trend_history(school: str, days: int = 30) -> list[dict]:
    """Return the last N days of rankings for a specific team."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT run_date, quality_score, sentiment_score, divergence_score, divergence_label
            FROM daily_rankings
            WHERE school = ?
            ORDER BY run_date DESC
            LIMIT ?
        """, (school, days)).fetchall()
        return [dict(r) for r in rows]


def get_available_dates() -> list[str]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT run_date FROM daily_rankings ORDER BY run_date DESC"
        ).fetchall()
        return [r["run_date"] for r in rows]
