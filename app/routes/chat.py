import json
import os
import logging
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from pipeline.database import get_latest_rankings

logger = logging.getLogger(__name__)
router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent.parent / "templates"))

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-5"


def build_system_prompt() -> str:
    rows = get_latest_rankings()
    last_updated = rows[0]["run_date"] if rows else "unknown"

    # Summarise rankings as compact text for context injection
    lines = []
    for r in rows:
        ap = f"AP#{r['ap_rank']}" if r.get("ap_rank") else "Unranked"
        lines.append(
            f"{r['school']} ({r.get('conference','')}) | "
            f"Quality={r['quality_score']:.1f} Sentiment={r['sentiment_score']:.1f} "
            f"Divergence={r['divergence_score']:+.1f} [{r['divergence_label']}] | "
            f"{ap} | SP+={r.get('sp_rating') or 'N/A'} | "
            f"W%={r['win_pct']:.2f if r.get('win_pct') is not None else 'N/A'} "
            f"({r.get('games_played', 0)} games)"
        )
    rankings_text = "\n".join(lines)

    return f"""You are a college football analytics assistant specializing in public sentiment analysis and sports betting market inefficiencies.

You have access to the CFB Public Sentiment Rankings, last updated {last_updated}. These rankings measure the gap between how good a team actually is (Quality Score) vs. how the public perceives them (Sentiment Score).

SCORING METHODOLOGY:
- Quality Score (0-100): Weighted composite of SP+ rating (65%) and win percentage (35%). Higher = better actual team.
- Sentiment Score (0-100): Weighted composite of AP Poll rank (40%), Google Trends search interest (35%), and recruiting ranking (25%). Higher = more public attention/hype.
- Divergence Score: Sentiment minus Quality. Positive = overrated by public (fade candidates). Negative = underrated (value plays).
- Labels: Strongly Overrated (>20), Overrated (>10), Fairly Rated (-10 to 10), Underrated (<-10), Strongly Underrated (<-20)

CURRENT RANKINGS ({len(rows)} FBS teams):
{rankings_text}

When answering questions:
- Be direct and specific. Reference actual scores and ranks from the data above.
- Explain betting implications clearly: overrated teams get too much public money → lines shade against them → value is on their opponents or the under.
- Underrated teams attract too little public money → lines give them too many points → they can be good ATS plays.
- Acknowledge limitations: this is one model, not financial advice. Sharp bettors use many signals.
- You can compare teams, identify interesting matchups, highlight trends, and explain why a team might be in their divergence category.
- If asked about a team not in the data, say so clearly.
"""


@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    rows = get_latest_rankings()
    overrated = [r for r in rows if (r["divergence_score"] or 0) >= 10][:5]
    underrated = sorted(rows, key=lambda x: x["divergence_score"] or 0)[:5]
    has_api_key = bool(ANTHROPIC_API_KEY)

    return templates.TemplateResponse("chat.html", {
        "request": request,
        "overrated": overrated,
        "underrated": underrated,
        "has_api_key": has_api_key,
        "total_teams": len(rows),
    })


@router.post("/api/chat")
async def chat_stream(request: Request):
    if not ANTHROPIC_API_KEY:
        async def no_key():
            yield "data: " + json.dumps({"text": "⚠️ No Anthropic API key configured. Add ANTHROPIC_API_KEY to your .env file."}) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(no_key(), media_type="text/event-stream")

    body = await request.json()
    messages = body.get("messages", [])

    if not messages:
        async def empty():
            yield "data: [DONE]\n\n"
        return StreamingResponse(empty(), media_type="text/event-stream")

    async def generate():
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            system = build_system_prompt()

            with client.messages.stream(
                model=MODEL,
                max_tokens=1024,
                system=system,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    yield "data: " + json.dumps({"text": text}) + "\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Chat stream error: {e}")
            yield "data: " + json.dumps({"text": f"\n\n⚠️ Error: {str(e)}"}) + "\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
