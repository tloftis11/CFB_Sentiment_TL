"""
CFB Sentiment Rankings — main entry point.

Usage:
    python main.py               # Run full pipeline and display results
    python main.py --display     # Display latest stored rankings only
    python main.py --team "Ohio State"  # Show trend history for one team
    python main.py --overrated   # Show only overrated teams
    python main.py --underrated  # Show only underrated teams
    python main.py --conf SEC    # Filter by conference
"""

import argparse
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from tabulate import tabulate

load_dotenv()

from pipeline.cfbd_client import CFBDClient, POWER_CONFERENCES
from pipeline.google_trends import GoogleTrendsClient
from pipeline.scoring import compute_rankings
from pipeline.database import (
    init_db,
    upsert_ranking,
    get_latest_rankings,
    get_trend_history,
    get_available_dates,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

CFBD_API_KEY   = os.getenv("CFBD_API_KEY", "")
SEASON_YEAR    = int(os.getenv("CFB_SEASON_YEAR", date.today().year))
POWER_ONLY     = os.getenv("TRENDS_POWER_ONLY", "true").lower() == "true"
SKIP_TRENDS    = os.getenv("SKIP_TRENDS", "false").lower() == "true"

JSON_PATH = Path(__file__).parent / "data" / "rankings.json"


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

DIVERG_COLORS = {
    "Strongly Overrated":   "\033[91m",   # red
    "Overrated":            "\033[93m",   # yellow
    "Fairly Rated":         "\033[0m",    # default
    "Underrated":           "\033[96m",   # cyan
    "Strongly Underrated":  "\033[92m",   # green
}
RESET = "\033[0m"


def _color(text: str, label: str) -> str:
    return f"{DIVERG_COLORS.get(label, '')}{text}{RESET}"


def display_rankings(rows: list[dict], title: str = "CFB Public Sentiment Rankings"):
    if not rows:
        print("No rankings available. Run the pipeline first.")
        return

    run_date = rows[0].get("run_date", "unknown")
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"  As of: {run_date}  |  {len(rows)} teams")
    print(f"{'='*80}\n")

    table = []
    for i, r in enumerate(rows, 1):
        div = r["divergence_score"] or 0
        label = r["divergence_label"] or ""
        table.append([
            i,
            r["school"],
            r.get("conference", "")[:10],
            f"{r['quality_score']:.1f}"  if r.get("quality_score")   else "—",
            f"#{r['quality_rank']}"       if r.get("quality_rank")    else "—",
            f"{r['sentiment_score']:.1f}" if r.get("sentiment_score") else "—",
            f"#{r['sentiment_rank']}"     if r.get("sentiment_rank")  else "—",
            f"{div:+.1f}",
            label,
        ])

    headers = ["#", "School", "Conf", "Quality", "Ql Rk", "Sentmt", "St Rk", "Diverg", "Label"]
    print(tabulate(table, headers=headers, tablefmt="simple"))

    # Summary callouts
    overrated   = [r for r in rows if (r["divergence_score"] or 0) >= 10]
    underrated  = sorted(rows, key=lambda x: x["divergence_score"] or 0)
    underrated  = [r for r in underrated if (r["divergence_score"] or 0) <= -10]

    if overrated:
        print(f"\n🔴  OVERRATED — Fade Candidates (sentiment >> quality):")
        for r in overrated[:8]:
            print(f"   {r['school']:<25} {r['divergence_label']}  ({r['divergence_score']:+.1f})")

    if underrated:
        print(f"\n🟢  UNDERRATED — Value Plays (quality >> sentiment):")
        for r in underrated[:8]:
            print(f"   {r['school']:<25} {r['divergence_label']}  ({r['divergence_score']:+.1f})")

    print()


def display_team_history(school: str):
    rows = get_trend_history(school, days=30)
    if not rows:
        print(f"No history found for '{school}'.")
        return
    print(f"\nTrend History — {school}")
    table = [[r["run_date"], f"{r['quality_score']:.1f}", f"{r['sentiment_score']:.1f}",
              f"{r['divergence_score']:+.1f}", r["divergence_label"]] for r in rows]
    print(tabulate(table, headers=["Date", "Quality", "Sentiment", "Divergence", "Label"],
                   tablefmt="simple"))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline():
    logger.info(f"=== CFB Sentiment Pipeline  |  {date.today()}  |  Season {SEASON_YEAR} ===")

    if not CFBD_API_KEY:
        raise RuntimeError(
            "CFBD_API_KEY not set. "
            "Register free at https://collegefootballdata.com/key and add it to .env"
        )

    init_db()

    cfbd   = CFBDClient(CFBD_API_KEY)
    trends = GoogleTrendsClient()

    # -- Teams ----------------------------------------------------------------
    logger.info("Fetching FBS teams...")
    raw_teams = cfbd.get_fbs_teams(SEASON_YEAR)
    conf_map  = {t["school"]: t.get("conference", "Independent") for t in raw_teams}
    all_schools = [t["school"] for t in raw_teams]
    logger.info(f"  {len(all_schools)} FBS teams found")

    # -- SP+ Ratings ----------------------------------------------------------
    logger.info("Fetching SP+ ratings...")
    sp_map = cfbd.get_sp_ratings(SEASON_YEAR)
    logger.info(f"  {len(sp_map)} teams with SP+ ratings")

    # -- AP Poll --------------------------------------------------------------
    logger.info("Fetching AP poll rankings...")
    ap_ranks = cfbd.get_ap_rankings(SEASON_YEAR)
    logger.info(f"  {len(ap_ranks)} teams ranked in AP poll")

    # -- Recruiting -----------------------------------------------------------
    logger.info("Fetching recruiting rankings...")
    recruiting_map = cfbd.get_recruiting_rankings(SEASON_YEAR)
    logger.info(f"  {len(recruiting_map)} teams with recruiting data")

    # -- Win Records ----------------------------------------------------------
    logger.info("Fetching game results...")
    win_pcts = cfbd.get_win_pcts(SEASON_YEAR)
    games_played = sum(1 for v in win_pcts.values() if v["games_played"] > 0)
    logger.info(f"  {games_played} teams with game results")

    # -- Google Trends --------------------------------------------------------
    if SKIP_TRENDS:
        logger.info("Skipping Google Trends (SKIP_TRENDS=true — CI mode)")
        gt_scores = {}
    elif POWER_ONLY:
        trend_schools = [s for s in all_schools if conf_map.get(s) in POWER_CONFERENCES]
        logger.info(f"Fetching Google Trends for {len(trend_schools)} Power/top-G5 teams...")
        gt_scores = trends.get_scores(trend_schools)
        logger.info(f"  Trends fetched for {len(gt_scores)} teams")
    else:
        trend_schools = all_schools
        logger.info(f"Fetching Google Trends for all {len(trend_schools)} FBS teams (slow)...")
        gt_scores = trends.get_scores(trend_schools)
        logger.info(f"  Trends fetched for {len(gt_scores)} teams")

    # -- Assemble & Score -----------------------------------------------------
    logger.info("Assembling and scoring...")
    teams_data = []
    for school in all_schools:
        record = win_pcts.get(school, {})
        teams_data.append({
            "school":               school,
            "conference":           conf_map.get(school, "Independent"),
            "sp_rating":            sp_map.get(school),
            "win_pct":              record.get("win_pct"),
            "games_played":         record.get("games_played", 0),
            "ap_rank":              ap_ranks.get(school),
            "google_trends_score":  gt_scores.get(school, 0.0),
            "recruiting_rank":      recruiting_map.get(school),
        })

    df = compute_rankings(teams_data, run_date=date.today())
    logger.info(f"  Scored {len(df)} teams")

    # -- Store ----------------------------------------------------------------
    logger.info("Storing rankings in database...")
    for _, row in df.iterrows():
        upsert_ranking({
            "school":               row["school"],
            "conference":           row["conference"],
            "run_date":             row["run_date"],
            "sp_rating":            row.get("sp_rating"),
            "win_pct":              row.get("win_pct"),
            "games_played":         int(row.get("games_played", 0)),
            "ap_rank":              int(row["ap_rank"]) if row.get("ap_rank") and str(row["ap_rank"]) != "nan" else None,
            "google_trends_score":  round(float(row.get("google_trends_score", 0)), 3),
            "recruiting_rank":      int(row["recruiting_rank"]) if row.get("recruiting_rank") and str(row["recruiting_rank"]) != "nan" else None,
            "quality_score":        round(float(row["quality_score"]), 2),
            "sentiment_score":      round(float(row["sentiment_score"]), 2),
            "divergence_score":     round(float(row["divergence_score"]), 2),
            "divergence_label":     row["divergence_label"],
            "quality_rank":         int(row["quality_rank"]),
            "sentiment_rank":       int(row["sentiment_rank"]),
            "divergence_rank":      int(row["divergence_rank"]),
        })

    logger.info("Pipeline complete.")

    # -- Export JSON (committed to repo so Render can read it) ----------------
    export_rankings_json()

    # -- Display --------------------------------------------------------------
    rows = get_latest_rankings()
    display_rankings(rows)


def export_rankings_json():
    """Write latest rankings to data/rankings.json for Render to consume."""
    rows = get_latest_rankings()
    out = {
        "last_updated": rows[0]["run_date"] if rows else str(date.today()),
        "has_trends": any(r.get("google_trends_score", 0) for r in rows),
        "teams": rows,
    }
    JSON_PATH.parent.mkdir(exist_ok=True)
    with open(JSON_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info(f"Exported {len(rows)} teams → {JSON_PATH}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CFB Public Sentiment Rankings")
    parser.add_argument("--display",    action="store_true", help="Show latest rankings (no pipeline run)")
    parser.add_argument("--overrated",  action="store_true", help="Show only overrated teams")
    parser.add_argument("--underrated", action="store_true", help="Show only underrated teams")
    parser.add_argument("--conf",       type=str,            help="Filter by conference name")
    parser.add_argument("--team",       type=str,            help="Show trend history for a team")
    parser.add_argument("--dates",      action="store_true", help="List available dates in DB")
    args = parser.parse_args()

    init_db()

    if args.dates:
        dates = get_available_dates()
        print("Available dates:", dates if dates else "None (run pipeline first)")
        return

    if args.team:
        display_team_history(args.team)
        return

    if args.display or args.overrated or args.underrated or args.conf:
        div_filter = None
        if args.overrated:
            rows = get_latest_rankings()
            rows = [r for r in rows if (r["divergence_score"] or 0) >= 10]
            title = "Overrated Teams — Fade Candidates"
        elif args.underrated:
            rows = get_latest_rankings()
            rows = sorted([r for r in rows if (r["divergence_score"] or 0) <= -10],
                          key=lambda x: x["divergence_score"])
            title = "Underrated Teams — Value Plays"
        else:
            rows = get_latest_rankings(conference_filter=args.conf)
            title = f"CFB Sentiment Rankings{' — ' + args.conf if args.conf else ''}"
        display_rankings(rows, title)
        return

    run_pipeline()


if __name__ == "__main__":
    main()
