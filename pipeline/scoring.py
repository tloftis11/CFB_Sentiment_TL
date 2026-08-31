"""
Scoring engine: converts raw inputs into Quality, Sentiment, and Divergence scores.

Quality Score  (0–100) = how good the team actually is
Sentiment Score (0–100) = how the public perceives the team
Divergence Score        = Sentiment – Quality
  Positive → Overrated by public (fade candidates)
  Negative → Underrated by public (value plays)

When Google Trends data is absent (SKIP_TRENDS=true on CI), the Trends weight
falls entirely to AP Poll + Recruiting via the normalised zero series, which
naturally redistributes by min-max into equal values and drops out — so the
remaining two components still drive the divergence ranking.
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

# --- Divergence label percentile cutoffs -------------------------------------

# Labels are assigned by where a team's divergence falls within the current
# field's distribution — not by fixed absolute thresholds.  This ensures the
# output is always balanced regardless of season, missing data, or score skew.
#
# Approximate team counts for a 138-team FBS field:
#   Strongly Overrated  : top 10%  → ~14 teams
#   Overrated           : 75–90th  → ~21 teams
#   Fairly Rated        : 25–75th  → ~69 teams
#   Underrated          : 10–25th  → ~21 teams
#   Strongly Underrated : bot 10%  → ~14 teams

_LABEL_PCTS = (0.10, 0.25, 0.75, 0.90)   # (p10, p25, p75, p90)


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


def _assign_labels(series: pd.Series) -> pd.Series:
    """Assign divergence labels using quantile cutoffs of the distribution."""
    p10, p25, p75, p90 = [series.quantile(q) for q in _LABEL_PCTS]

    def _label(s):
        if s >= p90: return "Strongly Overrated"
        if s >= p75: return "Overrated"
        if s >  p25: return "Fairly Rated"
        if s >  p10: return "Underrated"
        return "Strongly Underrated"

    return series.apply(_label)


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
    # Convert both scores to percentile ranks (0–100) before subtracting.
    # This eliminates the AP Poll sparsity bias: only 25 teams are ranked in
    # the poll, so raw sentiment scores cluster near zero for the other ~113
    # teams, making almost everyone appear "underrated." By comparing WHERE
    # each team ranks in sentiment vs WHERE it ranks in quality, the divergence
    # distribution is naturally centered at zero.

    # method='min': tied teams get the lowest rank in their group.
    # This prevents zero-sentiment teams (tied at minimum) from receiving an
    # inflated middle rank and falsely appearing overrated.
    df["quality_pct"]   = df["quality_score"].rank(pct=True, ascending=True, method="min") * 100
    df["sentiment_pct"] = df["sentiment_score"].rank(pct=True, ascending=True, method="min") * 100
    df["divergence_score"] = df["sentiment_pct"] - df["quality_pct"]

    df["divergence_label"] = _assign_labels(df["divergence_score"])

    # ---- Rank positions ------------------------------------------------------

    df["quality_rank"]    = df["quality_score"].rank(ascending=False,    method="min").astype(int)
    df["sentiment_rank"]  = df["sentiment_score"].rank(ascending=False,   method="min").astype(int)
    # Divergence rank: most overrated = rank 1
    df["divergence_rank"] = df["divergence_score"].rank(ascending=False,  method="min").astype(int)

    df["run_date"] = run_date.isoformat()

    return df.sort_values("divergence_score", ascending=False).reset_index(drop=True)
