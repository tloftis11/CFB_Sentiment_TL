import logging
from datetime import datetime, date
from typing import Optional

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.collegefootballdata.com"

# Conferences considered "Power" for filtering purposes
POWER_CONFERENCES = {
    "SEC", "Big Ten", "Big 12", "ACC", "Pac-12",
    # Top G5
    "American Athletic", "Mountain West", "Sun Belt",
}


class CFBDClient:
    def __init__(self, api_key: str):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        })

    def _get(self, endpoint: str, params: dict = None) -> list:
        url = f"{BASE_URL}{endpoint}"
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    # -------------------------------------------------------------------------
    # Teams
    # -------------------------------------------------------------------------

    def get_fbs_teams(self, year: int) -> list[dict]:
        """All FBS teams with conference info."""
        return self._get("/teams/fbs", {"year": year})

    # -------------------------------------------------------------------------
    # Ratings
    # -------------------------------------------------------------------------

    def get_sp_ratings(self, year: int) -> dict[str, float]:
        """Returns {school: sp_rating}."""
        data = self._get("/ratings/sp", {"year": year})
        return {r["team"]: r["rating"] for r in data if r.get("rating") is not None}

    # -------------------------------------------------------------------------
    # Poll Rankings
    # -------------------------------------------------------------------------

    def get_current_week(self, year: int) -> tuple[int, str]:
        """
        Returns (week_number, season_type) for today.
        season_type is 'regular', 'postseason', or 'preseason'.
        """
        try:
            weeks = self._get("/calendar", {"year": year})
        except Exception as e:
            logger.warning(f"Calendar fetch failed: {e}. Defaulting to week 1 preseason.")
            return 1, "preseason"

        today = date.today()

        # Check preseason (before week 1 kickoff)
        if weeks:
            first_game = datetime.strptime(weeks[0]["firstGameStart"][:10], "%Y-%m-%d").date()
            if today < first_game:
                return 1, "preseason"

        for week_info in weeks:
            try:
                start = datetime.strptime(week_info["firstGameStart"][:10], "%Y-%m-%d").date()
                end = datetime.strptime(week_info["lastGameStart"][:10], "%Y-%m-%d").date()
                if start <= today <= end:
                    return week_info["week"], "regular"
            except (KeyError, ValueError):
                continue

        # After season – find most recent completed week
        completed = []
        for w in weeks:
            try:
                end = datetime.strptime(w["lastGameStart"][:10], "%Y-%m-%d").date()
                if end < today:
                    completed.append(w)
            except (KeyError, ValueError):
                continue

        if completed:
            last = completed[-1]
            return last["week"], "regular"

        return 1, "preseason"

    def get_ap_rankings(self, year: int) -> dict[str, int]:
        """Returns {school: ap_rank} for the most current AP poll available."""
        week, season_type = self.get_current_week(year)
        logger.info(f"Fetching AP poll: year={year} week={week} type={season_type}")

        try:
            data = self._get("/rankings", {
                "year": year,
                "week": week,
                "seasonType": season_type,
            })
        except requests.HTTPError as e:
            logger.warning(f"Rankings fetch failed (week {week}): {e}")
            return {}

        ap_ranks: dict[str, int] = {}
        for entry in data:
            for poll in entry.get("polls", []):
                if poll["poll"] == "AP Top 25":
                    for rank_entry in poll.get("ranks", []):
                        ap_ranks[rank_entry["school"]] = rank_entry["rank"]
        return ap_ranks

    # -------------------------------------------------------------------------
    # Recruiting
    # -------------------------------------------------------------------------

    def get_recruiting_rankings(self, year: int) -> dict[str, int]:
        """Returns {school: recruiting_rank}."""
        try:
            data = self._get("/recruiting/teams", {"year": year})
            return {r["team"]: r["rank"] for r in data if r.get("rank") is not None}
        except Exception as e:
            logger.warning(f"Recruiting rankings unavailable: {e}")
            return {}

    # -------------------------------------------------------------------------
    # Game Results → Win %
    # -------------------------------------------------------------------------

    def get_win_pcts(self, year: int) -> dict[str, dict]:
        """
        Returns {school: {wins, losses, games_played, win_pct}}.
        Only counts games that have been played (home_points is not None).
        """
        try:
            games = self._get("/games", {"year": year, "seasonType": "regular"})
        except Exception as e:
            logger.warning(f"Games fetch failed: {e}")
            return {}

        records: dict[str, dict] = {}

        for game in games:
            if game.get("home_points") is None:
                continue  # Not yet played

            for side, pts, opp in [
                (game["home_team"], game["home_points"], game["away_points"]),
                (game["away_team"], game["away_points"], game["home_points"]),
            ]:
                if side not in records:
                    records[side] = {"wins": 0, "losses": 0}
                if pts > opp:
                    records[side]["wins"] += 1
                else:
                    records[side]["losses"] += 1

        result = {}
        for school, rec in records.items():
            total = rec["wins"] + rec["losses"]
            result[school] = {
                "wins": rec["wins"],
                "losses": rec["losses"],
                "games_played": total,
                "win_pct": rec["wins"] / total if total > 0 else None,
            }
        return result
