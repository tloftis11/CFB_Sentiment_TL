"""
FastAPI web server — entry point for Render deployment.

Data flow:
  1. Pipeline runs locally (or via GitHub Actions) → writes data/rankings.json
  2. That JSON is committed to GitHub
  3. Render deploys the new commit → this server seeds its SQLite from the JSON
  4. Routes serve data from SQLite

Local dev:
    uvicorn server:app --reload --port 8000
"""

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

from pipeline.database import init_db, upsert_ranking, get_latest_rankings
from app.routes.rankings import router as rankings_router
from app.routes.chat import router as chat_router

app = FastAPI(
    title="CFB Sentiment Rankings",
    description="Public sentiment vs. actual quality — college football analytics",
    version="1.0.0",
)

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

app.include_router(rankings_router)
app.include_router(chat_router)

JSON_PATH = Path(__file__).parent / "data" / "rankings.json"


def _seed_from_json():
    """Populate SQLite from rankings.json if the DB is empty."""
    if get_latest_rankings():
        logger.info("DB already has data — skipping JSON seed")
        return

    if not JSON_PATH.exists():
        logger.warning("rankings.json not found — DB will start empty")
        return

    with open(JSON_PATH) as f:
        payload = json.load(f)

    teams = payload.get("teams", [])
    for row in teams:
        upsert_ranking(row)

    logger.info(f"Seeded DB from rankings.json ({len(teams)} teams, updated {payload.get('last_updated')})")


@app.on_event("startup")
async def startup():
    init_db()
    _seed_from_json()


@app.get("/health")
async def health():
    return {"status": "ok"}
