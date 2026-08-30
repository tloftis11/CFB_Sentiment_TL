from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from pipeline.database import get_latest_rankings, get_trend_history, get_available_dates

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent.parent / "templates"))


@router.get("/", response_class=HTMLResponse)
async def rankings_page(
    request: Request,
    conf: str = Query(None),
    label: str = Query(None),
):
    rows = get_latest_rankings(divergence_filter=label, conference_filter=conf)
    dates = get_available_dates()
    last_updated = dates[0] if dates else "No data yet"

    # Unique conferences for filter dropdown
    all_rows = get_latest_rankings()
    conferences = sorted({r["conference"] for r in all_rows if r.get("conference")})

    return templates.TemplateResponse("rankings.html", {
        "request": request,
        "rows": rows,
        "conferences": conferences,
        "selected_conf": conf or "",
        "selected_label": label or "",
        "last_updated": last_updated,
        "total_teams": len(all_rows),
    })


@router.get("/api/rankings")
async def api_rankings(
    conf: str = Query(None),
    label: str = Query(None),
):
    return get_latest_rankings(divergence_filter=label, conference_filter=conf)


@router.get("/api/team/{school}")
async def api_team_history(school: str, days: int = 30):
    return get_trend_history(school, days=days)


@router.get("/api/conferences")
async def api_conferences():
    rows = get_latest_rankings()
    return sorted({r["conference"] for r in rows if r.get("conference")})
