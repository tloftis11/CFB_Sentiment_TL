"""
FastAPI web server — entry point for Render deployment.

Local dev:
    uvicorn server:app --reload --port 8000

Render:
    Start command: uvicorn server:app --host 0.0.0.0 --port $PORT
"""

import logging
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(message)s",
)

from pipeline.database import init_db
from app.routes.rankings import router as rankings_router
from app.routes.chat import router as chat_router

app = FastAPI(
    title="CFB Sentiment Rankings",
    description="Public sentiment vs. actual quality — college football analytics",
    version="1.0.0",
)

# Static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Routes
app.include_router(rankings_router)
app.include_router(chat_router)


@app.on_event("startup")
async def startup():
    init_db()


@app.get("/health")
async def health():
    return {"status": "ok"}
