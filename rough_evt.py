"""
Core Rough Path EVT engine using esig + iisignature (no PyTorch dependency).
"""

import numpy as np
import pandas as pd
from scipy.stats import genpareto
from typing import Dict, Tuple
import warnings
warnings.filterwarnings("ignore")

# Try esig first, fall back to iisignature
try:
    import esig
    HAS_ESIG = True
except ImportError:
    HAS_ESIG = False
    try:
        import iisignature
        HAS_IISIG = True
    except ImportError:
        HAS_IISIG = False
        raise ImportError("Install esig or iisignature for path signature computation")


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns from raw prices."""
    return np.log(prices / prices.shift(1)).dropna()


def path_signature_norm(
    returns: np.ndarray,
    depth: int = 3,
    lookahead: int = 5
) -> float:
    """
    Compute the L∞-norm of the truncated signature using esig or iisignature.
    """
    if len(returns) < 2:
        return np.nan
    
    # Construct path: time + returns (2D path)
    n = len(returns)
    t = np.linspace(0, 1, n)
    path = np.column_stack([t, returns])
    
    try:
        if HAS_ESIG:
            # esig: stream is (n, d) array
            sig = esig.stream2sig(path, depth)
            # L∞ norm
            sig_norm = np.max(np.abs(sig))
            return float(sig_norm)
        elif HAS_IISIG:
            # iisignature: computes signature
            sig = iisignature.prepare(path, depth, include_time=False)
            sig_norm = np.max(np.abs(sig))
            return float(sig_norm)
        else:
            return np.nan
    except Exception:
        return np.nan


def rolling_signature_norms(
    returns: pd.Series,
    window_days: int,
    depth: int = 3,
    lookahead: int = 5
) -> pd.Series:
    """Compute rolling L∞-norm of path signature."""
    norms = []
    dates = []
    
    for i in range(window_days + lookahead, len(returns)):
        window_returns = returns.iloc[i - window_days:i].values
        lookahead_returns = returns.iloc[i:i + lookahead].values
        full_path = np.concatenate([window_returns, lookahead_returns])
        
        norm = path_signature_norm(full_path, depth, lookahead)
        if not np.isnan(norm):
            norms.append(norm)
            dates.append(returns.index[i])
    
    return pd.Series(norms, index=dates)


def fit_gpd(
    data: np.ndarray,
    threshold_quantile: float = 0.95
) -> Tuple[float, float, float, int, float]:
    """Fit Generalized Pareto Distribution to exceedances."""
    if len(data) < 20:
        return np.nan, np.nan, np.nan, 0, np.nan
    
    threshold = np.quantile(data, threshold_quantile)
    exceedances = data[data > threshold] - threshold
    
    if len(exceedances) < 10:
        return threshold, np.nan, np.nan, len(exceedances), np.nan
    
    try:
        xi, loc, sigma = genpareto.fit(exceedances, floc=0)
        n_exceed = len(exceedances)
        exceed_prob = n_exceed / len(data)
        return threshold, xi, sigma, n_exceed, exceed_prob
    except Exception:
        return threshold, np.nan, np.nan, len(exceedances), np.nan


def return_level(
    xi: float,
    sigma: float,
    exceed_prob: float,
    n_years: int = 100,
    data_points_per_year: int = 252
) -> float:
    """Compute the 1-in-N-year return level."""
    if any(np.isnan([xi, sigma, exceed_prob])) or exceed_prob <= 0:
        return np.nan
    
    N = n_years * data_points_per_year
    
    if abs(xi) < 1e-10:
        return sigma * np.log(N * exceed_prob)
    else:
        return (sigma / xi) * ((N * exceed_prob) ** xi - 1)


def compute_rough_evt(
    prices: pd.Series,
    window_days: int = 252,
    depth: int = 3,
    lookahead: int = 5,
    threshold_quantile: float = 0.95,
    return_period_years: int = 100
) -> Dict:
    """Compute full Rough-EVT analysis for a single ticker."""
    returns = compute_log_returns(prices)
    if len(returns) < window_days + lookahead + 20:
        return {
            "return_level_100yr": np.nan,
            "tail_index": np.nan,
            "threshold": np.nan,
            "exceedances": 0,
            "best_window": window_days,
            "lookahead": lookahead,
            "error": "Insufficient data"
        }
    
    norms = rolling_signature_norms(
        returns,
        window_days=window_days,
        depth=depth,
        lookahead=lookahead
    )
    
    if len(norms) < 20:
        return {
            "return_level_100yr": np.nan,
            "tail_index": np.nan,
            "threshold": np.nan,
            "exceedances": len(norms),
            "best_window": window_days,
            "lookahead": lookahead,
            "error": f"Insufficient norms: {len(norms)}"
        }
    
    threshold, xi, sigma, n_exceed, exceed_prob = fit_gpd(
        norms.values,
        threshold_quantile=threshold_quantile
    )
    
    if np.isnan(xi) or n_exceed < 10:
        return {
            "return_level_100yr": np.nan,
            "tail_index": xi,
            "threshold": threshold,
            "exceedances": n_exceed,
            "best_window": window_days,
            "lookahead": lookahead,
            "error": "GPD fit failed or insufficient exceedances"
        }
    
    rl = return_level(xi, sigma, exceed_prob, n_years=return_period_years)
    
    return {
        "return_level_100yr": rl,
        "tail_index": xi,
        "threshold": threshold,
        "exceedances": n_exceed,
        "best_window": window_days,
        "lookahead": lookahead,
        "exceed_prob": exceed_prob,
        "error": None
    }


def compute_cross_sectional_zscore(scores: Dict[str, float]) -> Dict[str, float]:
    """Compute cross-sectional z-scores within a universe."""
    values = np.array([v for v in scores.values() if not np.isnan(v)])
    if len(values) < 2:
        return {t: np.nan for t in scores.keys()}
    
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return {t: 0.0 for t in scores.keys()}
    
    return {t: (scores[t] - mean) / std if not np.isnan(scores[t]) else np.nan 
            for t in scores.keys()}
