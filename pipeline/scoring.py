"""
Scoring engine: converts raw inputs into Quality, Sentiment, and Divergence scores.

Quality Score  (0–100) = how good the team actually is
Sentiment Score (0–100) = how the public perceives the team
Divergence Score        = Sentiment – Quality
  Positive → Overrated by public (fade candidates)
  Negative → Underrated by public (value plays)
"""

import numpy as np
import pandas as pd
from datetime import date

# --- Weights ----------------------------------------------------------------

QUALITY_WEIGHTS = {
    "sp_normalized":       0.65,   # SP+ is the best available objective metric
    "win_pct_normalized":  0.35,   # Actual W/L record (schedule-adjusted via SP+)
}

SENTIMENT_WEIGHTS = {
    "ap_rank_normalized":      0.40,  # Poll perception = strongest public signal
    "google_trends_normalized": 0.35, # National interest / search attention
    "recruiting_normalized":   0.25,  # Blue-chip hype; public buys into recruiting
}

# --- Divergence labels -------------------------------------------------------

# Thresholds are on the raw divergence scale (Sentiment – Quality, both 0–100)
DIVERGENCE_LABELS = [
    (20,  "Strongly Overrated"),
    (10,  "Overrated"),
    (-10, "Fairly Rated"),
    (-20, "Underrated"),
]
# Anything below -20 falls through to "Strongly Underrated"


# --- Helper functions --------------------------------------------------------

def _minmax(s: pd.Series, invert: bool = False) -> pd.Series:
    """Min-max normalise to 0–100. Ties stay tied; constant series → 50."""
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series([50.0] * len(s), index=s.index)
    out = (s - mn) / (mx - mn) * 100.0
    return 100.0 - out if invert else out


def _ap_to_score(rank) -> float:
    """AP rank 1 → 100, rank 25 → 4, unranked → 0."""
    if rank is None or (isinstance(rank, float) and np.isnan(rank)):
        return 0.0
    rank = int(rank)
    return max(0.0, (26 - rank) / 25 * 100)


def _recruiting_to_score(rank) -> float:
    """
    Recruiting rank 1 → 100, rank 50 → 2, beyond 50 → 0.
    Captures that public overweights blue-chip recruiting.
    """
    if rank is None or (isinstance(rank, float) and np.isnan(rank)):
        return 0.0
    rank = int(rank)
    return max(0.0, (51 - min(rank, 51)) / 50 * 100)


def _label(score: float) -> str:
    for threshold, label in DIVERGENCE_LABELS:
        if score >= threshold:
            return label
    return "Strongly Underrated"


# --- Main scoring function ---------------------------------------------------

def compute_rankings(teams_data: list[dict], run_date: date = None) -> pd.DataFrame:
    """
    Parameters
    ----------
    teams_data : list of dicts, each with:
        school, conference, sp_rating, win_pct, games_played,
        ap_rank, google_trends_score, recruiting_rank

    Returns
    -------
    pd.DataFrame sorted by divergence_score descending (most overrated first).
    """
    if run_date is None:
        run_date = date.today()

    df = pd.DataFrame(teams_data)

    # Need SP+ at minimum; drop teams without it
    df = df[df["sp_rating"].notna()].copy()
    if df.empty:
        return df

    # ---- Quality Score -------------------------------------------------------

    df["sp_normalized"]      = _minmax(df["sp_rating"])
    # Win pct: teams with no games played get neutral 50
    df["win_pct_filled"]     = df["win_pct"].fillna(0.5)
    df["win_pct_normalized"] = _minmax(df["win_pct_filled"])

    df["quality_score"] = (
        df["sp_normalized"]      * QUALITY_WEIGHTS["sp_normalized"] +
        df["win_pct_normalized"] * QUALITY_WEIGHTS["win_pct_normalized"]
    )

    # ---- Sentiment Score -----------------------------------------------------

    df["ap_raw_score"]         = df["ap_rank"].apply(_ap_to_score)
    df["ap_rank_normalized"]   = _minmax(df["ap_raw_score"])

    df["gt_filled"]                  = df["google_trends_score"].fillna(0.0)
    df["google_trends_normalized"]   = _minmax(df["gt_filled"])

    df["rec_raw_score"]        = df["recruiting_rank"].apply(_recruiting_to_score)
    df["recruiting_normalized"] = _minmax(df["rec_raw_score"])

    df["sentiment_score"] = (
        df["ap_rank_normalized"]      * SENTIMENT_WEIGHTS["ap_rank_normalized"] +
        df["google_trends_normalized"] * SENTIMENT_WEIGHTS["google_trends_normalized"] +
        df["recruiting_normalized"]    * SENTIMENT_WEIGHTS["recruiting_normalized"]
    )

    # ---- Divergence ----------------------------------------------------------

    df["divergence_score"] = df["sentiment_score"] - df["quality_score"]
    df["divergence_label"] = df["divergence_score"].apply(_label)

    # ---- Rank positions ------------------------------------------------------

    df["quality_rank"]    = df["quality_score"].rank(ascending=False,    method="min").astype(int)
    df["sentiment_rank"]  = df["sentiment_score"].rank(ascending=False,   method="min").astype(int)
    # Divergence rank: most overrated = rank 1
    df["divergence_rank"] = df["divergence_score"].rank(ascending=False,  method="min").astype(int)

    df["run_date"] = run_date.isoformat()

    return df.sort_values("divergence_score", ascending=False).reset_index(drop=True)
