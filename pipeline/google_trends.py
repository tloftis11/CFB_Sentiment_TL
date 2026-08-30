"""
Google Trends client for CFB team interest scores.

Strategy: compare each team against the anchor term "college football"
using pytrends' interest_over_time. We chain batches of 4 teams + anchor
so all scores are on the same relative scale.

Rate-limit note: pytrends is unofficial and Google will throttle aggressive
requests. We sleep between batches and retry on 429s.
"""

import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

ANCHOR = "college football"
BATCH_SIZE = 4          # teams per batch (+ 1 anchor = 5 total, pytrends max)
SLEEP_BETWEEN = 12.0    # seconds between batches (avoids 429s; 29 batches ≈ 6 min total)
RETRY_SLEEP = 60.0      # seconds to wait after a 429
MAX_RETRIES = 2
TIMEFRAME = "today 1-m"  # past 30 days (valid pytrends preset)

# Override search terms for schools whose name alone is ambiguous
SEARCH_TERM_OVERRIDES: dict[str, str] = {
    "Alabama":        "Alabama Crimson Tide football",
    "Georgia":        "Georgia Bulldogs football",
    "Michigan":       "Michigan Wolverines football",
    "LSU":            "LSU Tigers football",
    "Ohio State":     "Ohio State Buckeyes football",
    "Notre Dame":     "Notre Dame Fighting Irish",
    "Texas":          "Texas Longhorns football",
    "Oklahoma":       "Oklahoma Sooners football",
    "USC":            "USC Trojans football",
    "Penn State":     "Penn State Nittany Lions football",
    "Oregon":         "Oregon Ducks football",
    "Florida":        "Florida Gators football",
    "Tennessee":      "Tennessee Volunteers football",
    "Miami":          "Miami Hurricanes football",
    "Washington":     "Washington Huskies football",
    "Missouri":       "Missouri Tigers football",
    "Iowa":           "Iowa Hawkeyes football",
    "Kansas":         "Kansas Jayhawks football",
    "Kansas State":   "Kansas State Wildcats football",
    "Mississippi":    "Ole Miss Rebels football",
    "Ole Miss":       "Ole Miss Rebels football",
    "Utah":           "Utah Utes football",
    "Colorado":       "Colorado Buffaloes football",
    "TCU":            "TCU Horned Frogs football",
    "Arkansas":       "Arkansas Razorbacks football",
    "Auburn":         "Auburn Tigers football",
    "Florida State":  "Florida State Seminoles football",
    "Clemson":        "Clemson Tigers football",
    "North Carolina": "UNC Tar Heels football",
    "Virginia Tech":  "Virginia Tech Hokies football",
    "Pittsburgh":     "Pitt Panthers football",
    "Louisville":     "Louisville Cardinals football",
    "Houston":        "Houston Cougars football",
    "Cincinnati":     "Cincinnati Bearcats football",
    "BYU":            "BYU Cougars football",
    "Boise State":    "Boise State Broncos football",
    "Air Force":      "Air Force Falcons football",
    "Navy":           "Navy Midshipmen football",
    "Army":           "Army Black Knights football",
}


def _search_term(school: str) -> str:
    return SEARCH_TERM_OVERRIDES.get(school, f"{school} football")


class GoogleTrendsClient:
    def __init__(self, timeframe: str = TIMEFRAME):
        self.timeframe = timeframe

    def _build_pytrends(self):
        from pytrends.request import TrendReq
        return TrendReq(
            hl="en-US",
            tz=360,
            timeout=(10, 30),
            requests_args={"headers": {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            }},
        )

    def _fetch_batch(self, schools: list[str]) -> dict[str, float]:
        """
        Fetch interest for up to BATCH_SIZE schools.
        Returns {school: raw_avg_score} where score is the team's average
        interest over time (0–100 scale within each batch).
        Cross-batch comparability is handled by the scoring engine's min-max
        normalization across all teams at the end.
        """
        terms = [_search_term(s) for s in schools]
        pt = self._build_pytrends()

        for attempt in range(MAX_RETRIES + 1):
            try:
                pt.build_payload(terms, timeframe=self.timeframe, geo="US")
                df = pt.interest_over_time()
                break
            except Exception as e:
                msg = str(e)
                if "429" in msg or "Too Many" in msg:
                    if attempt < MAX_RETRIES:
                        logger.warning(f"Rate limited. Sleeping {RETRY_SLEEP}s before retry {attempt+1}...")
                        time.sleep(RETRY_SLEEP)
                        continue
                logger.warning(f"Google Trends batch failed (attempt {attempt+1}): {e}")
                return {s: 0.0 for s in schools}

        if df.empty:
            logger.warning("Empty response from Google Trends.")
            return {s: 0.0 for s in schools}

        if "isPartial" in df.columns:
            df = df.drop(columns=["isPartial"])

        results: dict[str, float] = {}
        for school in schools:
            term = _search_term(school)
            if term in df.columns:
                results[school] = float(df[term].mean())
            else:
                results[school] = 0.0

        return results

    def get_scores(self, schools: list[str]) -> dict[str, float]:
        """
        Fetch Google Trends scores for all schools.
        Batches BATCH_SIZE teams at a time with sleep between batches.
        """
        all_results: dict[str, float] = {}
        total_batches = (len(schools) + BATCH_SIZE - 1) // BATCH_SIZE

        for i in range(0, len(schools), BATCH_SIZE):
            batch = schools[i: i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            logger.info(f"  Google Trends batch {batch_num}/{total_batches}: {', '.join(batch)}")

            batch_results = self._fetch_batch(batch)
            all_results.update(batch_results)

            if i + BATCH_SIZE < len(schools):
                time.sleep(SLEEP_BETWEEN)

        return all_results
