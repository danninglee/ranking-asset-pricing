"""
portfolio.py — Portfolio construction and annualised financial statistics.

Label scheme : bottom-33 / middle-33 / top-33  (balanced terciles)
Portfolio    : within each predicted tercile, select the most-confident
               ``long_pct`` / ``short_pct`` fraction of the universe to form
               the Long / Short legs, recovering the traditional top-10 /
               bottom-10 extreme-decile portfolios without any resampling.

               Long  : top    ``long_pct``  × N stocks by P(class=2)
                        restricted to stocks predicted as class 2.
               Short : top    ``short_pct`` × N stocks by P(class=0)
                        restricted to stocks predicted as class 0.

Aligns with GKX (2020) Table 4 methodology for annualised statistics:
  Annual Excess Return : mean(r − rf) × 12           [arithmetic]
  Sharpe Ratio         : mean(r − rf) / std(r − rf) × √12
  t-statistic          : mean(r − rf) / (std(r − rf) / √T)
  Max Drawdown         : max peak-to-trough decline in gross wealth
  Hit Rate             : fraction of months with r > rf
"""

import numpy as np


# ── Portfolio selection ───────────────────────────────────────────────────

def _select_portfolio_stocks(yp: np.ndarray,
                              y_prob: np.ndarray | None,
                              target_class: int,
                              n_total: int,
                              pct: float) -> np.ndarray:
    """
    Return a boolean mask of ``ceil(n_total × pct)`` stocks for one leg.

    Strategy
    --------
    1. Restrict candidates to stocks whose predicted class == ``target_class``
       (i.e. the predicted tercile).
    2. Among candidates, rank by P(target_class) descending and keep the top
       ``ceil(n_total × pct)`` — these are the most-confident predictions
       within the tercile, corresponding to the extreme decile of the universe.
    3. If ``y_prob`` is None (no probability available), fall back to using
       the full predicted tercile as the portfolio leg (original behaviour).

    Parameters
    ----------
    yp           : predicted class labels for the current month, shape (n,)
    y_prob       : predicted probabilities, shape (n, 3), or None
    target_class : 2 for Long leg, 0 for Short leg
    n_total      : total number of stocks this month
    pct          : desired fraction of universe (e.g. 0.10 for decile)
    """
    tercile_mask = (yp == target_class)

    if y_prob is None or tercile_mask.sum() == 0:
        # Fallback: use the whole predicted tercile
        return tercile_mask

    k = max(1, int(np.ceil(n_total * pct)))

    # Score only the tercile members; set outsiders to -inf so they are never
    # picked by argsort.
    scores = np.where(tercile_mask, y_prob[:, target_class], -np.inf)
    # argsort ascending → last k indices are the k highest scores
    top_k_idx = np.argpartition(scores, -k)[-k:]
    sel = np.zeros(n_total, dtype=bool)
    sel[top_k_idx] = True
    return sel


# ── Monthly return construction ───────────────────────────────────────────

def _build_monthly_returns(y_pred: np.ndarray,
                            fwd_ret: np.ndarray,
                            mktcap: np.ndarray,
                            eom: np.ndarray,
                            y_prob: np.ndarray | None = None,
                            weighting: str = "ew",
                            long_pct: float = 0.10,
                            short_pct: float = 0.10) -> dict:
    """
    Construct monthly Long / Short / Long-Short return series.

    Parameters
    ----------
    y_pred    : predicted class labels  (0=bottom-33, 1=mid-33, 2=top-33)
    fwd_ret   : forward return array  (test split)
    mktcap    : market cap array      (test split)
    eom       : end-of-month date array (test split)
    y_prob    : predicted probabilities, shape (n, 3); used to select the
                extreme ``long_pct`` / ``short_pct`` decile from each tercile.
                Pass None to use the full predicted tercile (original mode).
    weighting : "ew" | "vw"
    long_pct  : fraction of universe to hold long  (default 0.10 = top decile)
    short_pct : fraction of universe to sell short (default 0.10 = bot decile)
    """
    months = np.unique(eom)
    long_r, short_r, ls_r = [], [], []

    for m in months:
        mask = (eom == m)
        n    = int(mask.sum())
        yp   = y_pred[mask]
        r    = fwd_ret[mask]
        mc   = mktcap[mask]
        prob = y_prob[mask] if y_prob is not None else None

        long_sel  = _select_portfolio_stocks(yp, prob, 2, n, long_pct)
        short_sel = _select_portfolio_stocks(yp, prob, 0, n, short_pct)

        def _port(sel: np.ndarray) -> float:
            if sel.sum() == 0:
                return np.nan
            r_s = r[sel]
            if weighting == "vw":
                mc_s  = mc[sel]
                valid = ~np.isnan(mc_s) & (mc_s > 0)
                if valid.sum() == 0:
                    return float(np.nanmean(r_s))
                w = mc_s[valid] / mc_s[valid].sum()
                return float(np.dot(r_s[valid], w))
            return float(np.nanmean(r_s))

        lr = _port(long_sel)
        sr = _port(short_sel)
        long_r.append(lr)
        short_r.append(sr)
        ls_r.append(lr - sr if not (np.isnan(lr) or np.isnan(sr)) else np.nan)

    return {
        "long":       np.array(long_r),
        "short":      np.array(short_r),
        "long_short": np.array(ls_r),
    }


# ── Portfolio statistics ──────────────────────────────────────────────────

def _portfolio_stats(monthly_ret: np.ndarray,
                     rf: float | np.ndarray = 0.0) -> dict:
    """Annualised financial statistics from a monthly return series."""
    r = monthly_ret[~np.isnan(monthly_ret)]
    if len(r) == 0:
        return {"annual_ret": np.nan, "annual_excess_ret": np.nan,
                "sharpe": np.nan, "t_stat": np.nan,
                "max_drawdown": np.nan, "hit_rate": np.nan, "n_months": 0}

    rf_arr        = (np.full(len(r), rf) if np.isscalar(rf)
                     else np.asarray(rf)[~np.isnan(monthly_ret)])
    excess        = r - rf_arr
    annual_ret    = float(np.mean(r) * 12)
    annual_excess = float(np.mean(excess) * 12)
    std_excess    = float(np.std(excess, ddof=1))

    sharpe   = float(np.mean(excess) / std_excess * np.sqrt(12)) if std_excess > 0 else np.nan
    t_stat   = float(np.mean(excess) / (std_excess / np.sqrt(len(r)))) if std_excess > 0 else np.nan
    wealth   = np.cumprod(1.0 + r)
    peak     = np.maximum.accumulate(wealth)
    max_dd   = float(np.max((peak - wealth) / peak))
    hit_rate = float(np.mean(r > rf_arr))

    return {
        "annual_ret":        annual_ret,
        "annual_excess_ret": annual_excess,
        "sharpe":            sharpe,
        "t_stat":            t_stat,
        "max_drawdown":      max_dd,
        "hit_rate":          hit_rate,
        "n_months":          len(r),
    }


# ── Public API ────────────────────────────────────────────────────────────

def compute_portfolio_metrics(y_pred, fwd_ret, mktcap, eom,
                               rf: float | np.ndarray = 0.0,
                               y_prob: np.ndarray | None = None,
                               long_pct: float = 0.10,
                               short_pct: float = 0.10) -> dict:
    """
    Compute EW and VW portfolio metrics for Long, Short, Long-Short.

    Parameters
    ----------
    y_pred    : predicted class labels (0 / 1 / 2 for bottom/mid/top tercile)
    fwd_ret   : forward returns (test split)
    mktcap    : market cap     (test split)
    eom       : end-of-month dates (test split)
    rf        : monthly risk-free rate — scalar or per-month array aligned
                with the unique months in ``eom``.  Default = 0.
    y_prob    : predicted probabilities (n, 3).  When provided, only the top
                ``long_pct`` / ``short_pct`` fraction of the universe (by
                confidence) within each predicted tercile is traded.
                Pass None to trade the full predicted tercile.
    long_pct  : universe fraction for the Long  leg (default 0.10)
    short_pct : universe fraction for the Short leg (default 0.10)
    """
    return {
        w: {
            leg: _portfolio_stats(
                _build_monthly_returns(
                    y_pred, fwd_ret, mktcap, eom,
                    y_prob=y_prob,
                    weighting=w,
                    long_pct=long_pct,
                    short_pct=short_pct,
                )[leg],
                rf=rf)
            for leg in ["long", "short", "long_short"]
        }
        for w in ["ew", "vw"]
    }