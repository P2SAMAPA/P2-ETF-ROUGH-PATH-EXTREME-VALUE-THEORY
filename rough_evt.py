"""
Core Rough Path EVT engine - Pure Python implementation.
No external signature libraries required.
"""

import numpy as np
import pandas as pd
from scipy.stats import genpareto
from typing import Dict, Tuple
import warnings
warnings.filterwarnings("ignore")


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns from raw prices."""
    return np.log(prices / prices.shift(1)).dropna()


def compute_signature_norm_pure_python(
    returns: np.ndarray,
    depth: int = 3
) -> float:
    """
    Compute the L∞-norm of the truncated signature using pure Python.
    
    This implements the signature computation via iterated integrals
    using the Chen-Strichartz expansion.
    
    For a 1D path (returns), the signature terms are:
        S^0 = 1
        S^1 = ∫ dX = final_value - initial_value
        S^2 = ∫∫ dX⊗dX = (final_value - initial_value)^2 / 2
        S^3 = (final_value - initial_value)^3 / 6
        ...
    
    For a 2D path (time + returns), we compute the full signature.
    """
    if len(returns) < 2:
        return np.nan
    
    # Construct path: time + returns (2D path)
    n = len(returns)
    t = np.linspace(0, 1, n)
    path = np.column_stack([t, returns])
    
    # Compute increments
    increments = np.diff(path, axis=0)
    
    # Compute iterated integrals up to depth
    sig_terms = []
    
    # Level 1: ∫ dX = X_T - X_0
    level1 = path[-1] - path[0]
    sig_terms.extend(level1.tolist())
    
    # Level 2: ∫∫ dX⊗dX
    if depth >= 2:
        # For 2D path, there are 4 terms
        level2 = np.zeros((2, 2))
        for i in range(len(increments)):
            # Trapezoidal approximation of iterated integrals
            inc_i = increments[i]
            for j in range(len(increments)):
                if i <= j:
                    inc_j = increments[j]
                    level2 += np.outer(inc_i, inc_j)
        # Normalize by number of increments
        level2 = level2 / len(increments)
        sig_terms.extend(level2.flatten().tolist())
    
    # Level 3: ∫∫∫ dX⊗dX⊗dX
    if depth >= 3:
        level3 = np.zeros((2, 2, 2))
        for i in range(len(increments)):
            inc_i = increments[i]
            for j in range(len(increments)):
                if i <= j:
                    inc_j = increments[j]
                    for k in range(len(increments)):
                        if j <= k:
                            inc_k = increments[k]
                            # Outer product of three vectors
                            level3 += np.einsum('i,j,k->ijk', inc_i, inc_j, inc_k)
        # Normalize by number of increments
        level3 = level3 / (len(increments) ** 2)
        sig_terms.extend(level3.flatten().tolist())
    
    # L∞ norm of the signature vector
    if len(sig_terms) == 0:
        return np.nan
    
    return float(np.max(np.abs(sig_terms)))


def compute_signature_norm_leadlag(
    returns: np.ndarray,
    depth: int = 3
) -> float:
    """
    Lead-lag transformation signature norm.
    This is more robust for financial time series.
    """
    if len(returns) < 2:
        return np.nan
    
    # Lead-lag transformation: (x_t, x_{t+1})
    lead_lag_path = []
    for i in range(len(returns) - 1):
        lead_lag_path.append([returns[i], returns[i+1]])
    lead_lag_path = np.array(lead_lag_path)
    
    # Normalize
    mean = np.mean(lead_lag_path, axis=0)
    std = np.std(lead_lag_path, axis=0) + 1e-10
    path_normalized = (lead_lag_path - mean) / std
    
    # Compute simplified signature
    sig_terms = []
    
    # Level 1: increments
    increments = np.diff(path_normalized, axis=0)
    if len(increments) > 0:
        level1 = np.mean(increments, axis=0)
        sig_terms.extend(level1.tolist())
    
    # Level 2: covariance-like terms
    if depth >= 2 and len(increments) > 1:
        cov = np.cov(increments.T)
        sig_terms.extend(cov.flatten().tolist())
    
    # L∞ norm
    if len(sig_terms) == 0:
        return np.nan
    
    return float(np.max(np.abs(sig_terms)))


def rolling_signature_norms(
    returns: pd.Series,
    window_days: int,
    depth: int = 3,
    lookahead: int = 5
) -> pd.Series:
    """
    Compute rolling L∞-norm of path signature using pure Python.
    """
    norms = []
    dates = []
    
    for i in range(window_days + lookahead, len(returns)):
        window_returns = returns.iloc[i - window_days:i].values
        lookahead_returns = returns.iloc[i:i + lookahead].values
        full_path = np.concatenate([window_returns, lookahead_returns])
        
        # Use lead-lag signature for better stability
        norm = compute_signature_norm_leadlag(full_path, depth)
        if not np.isnan(norm) and norm > 0:
            norms.append(norm)
            dates.append(returns.index[i])
    
    return pd.Series(norms, index=dates)


def fit_gpd(
    data: np.ndarray,
    threshold_quantile: float = 0.95
) -> Tuple[float, float, float, int, float]:
    """
    Fit Generalized Pareto Distribution to exceedances over threshold.
    """
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
    """
    Compute the 1-in-N-year return level.
    """
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
    """
    Compute full Rough-EVT analysis for a single ticker.
    """
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
    """
    Compute cross-sectional z-scores within a universe.
    """
    values = np.array([v for v in scores.values() if not np.isnan(v)])
    if len(values) < 2:
        return {t: np.nan for t in scores.keys()}
    
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return {t: 0.0 for t in scores.keys()}
    
    return {t: (scores[t] - mean) / std if not np.isnan(scores[t]) else np.nan 
            for t in scores.keys()}
