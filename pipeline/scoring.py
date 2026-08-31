"""
Scoring engine: converts raw inputs into Quality, Sentiment, and Divergence scores.

Quality Score  (0–100) = SP+ rating, min-max normalised across all FBS teams
Sentiment Score (0–100) = public perception, computed in two tiers:
  • AP top-25 teams : AP (40%) + Trends (35%) + Recruiting (25%)
  • Unranked teams  : Trends (58.3%) + Recruiting (41.7%)
  Both groups share the same 0–100 scale.  AP is normalised within the ranked
  pool only (rank 1 = 100, rank 25 = 0) so being #25 is meaningfully different
  from being unranked — not nearly the same.

Divergence Score = sentiment percentile − quality percentile (both 0–100).
  Positive → Overrated by public (more sentiment attention than quality warrants)
  Negative → Underrated by public (higher quality than public attention)
  Labels are assigned via quantile cutoffs so the distribution stays balanced.
"""

import numpy as np
import pandas as pd
from datetime import date

# --- Weights ----------------------------------------------------------------

QUALITY_WEIGHTS = {
    "sp_normalized": 1.00,  # SP+ is the only quality input; win% is noise preseason
}

# Sentiment weights — AP-ranked teams use all three; unranked use Trends + Recruiting
# reweighted to 100% so both groups stay on the same 0-100 scale.
_W_AP, _W_GT, _W_REC = 0.40, 0.35, 0.25
_W_GT_ONLY  = _W_GT  / (_W_GT + _W_REC)   # ≈ 0.583
_W_REC_ONLY = _W_REC / (_W_GT + _W_REC)   # ≈ 0.417

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

    df["sp_normalized"]  = _minmax(df["sp_rating"])
    df["quality_score"]  = df["sp_normalized"] * QUALITY_WEIGHTS["sp_normalized"]

    # ---- Sentiment Score -----------------------------------------------------
    # AP: normalised within the ranked pool only (rank 1=100, rank 25=0).
    # Unranked teams receive NaN here and skip the AP component entirely.
    ap_mask = df["ap_rank"].notna()
    df["ap_rank_normalized"] = np.nan
    if ap_mask.sum() > 0:
        ranked_ap = df.loc[ap_mask, "ap_rank"].astype(float)
        # Invert: lower rank number = better; _minmax on negative values maps
        # rank 1 → 100 and rank 25 → 0 within the ranked pool.
        df.loc[ap_mask, "ap_rank_normalized"] = _minmax(-ranked_ap).values

    df["gt_filled"]                = df["google_trends_score"].fillna(0.0)
    df["google_trends_normalized"] = _minmax(df["gt_filled"])

    df["rec_raw_score"]            = df["recruiting_rank"].apply(_recruiting_to_score)
    df["recruiting_normalized"]    = _minmax(df["rec_raw_score"])

    # Ranked teams: AP (40%) + Trends (35%) + Recruiting (25%)
    # Unranked teams: Trends (58.3%) + Recruiting (41.7%) — same ratio, rescaled to 100%
    df["sentiment_score"] = df.apply(
        lambda r: (
            r["ap_rank_normalized"] * _W_AP
            + r["google_trends_normalized"] * _W_GT
            + r["recruiting_normalized"] * _W_REC
        ) if pd.notna(r["ap_rank_normalized"]) else (
            r["google_trends_normalized"] * _W_GT_ONLY
            + r["recruiting_normalized"] * _W_REC_ONLY
        ),
        axis=1,
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
