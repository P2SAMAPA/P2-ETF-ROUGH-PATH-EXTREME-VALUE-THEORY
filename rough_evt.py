"""
Core Rough Path EVT engine - Pure Python implementation.
No external signature libraries required - all computations are built-in.
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


def compute_path_signature_terms(
    path: np.ndarray,
    depth: int = 3
) -> np.ndarray:
    """
    Compute signature terms of a path using the Chen-Strichartz expansion.
    
    For a path X, the signature is the collection of all iterated integrals:
        S^0 = 1
        S^1_i = ∫ dX_i
        S^2_{ij} = ∫∫ dX_i ⊗ dX_j
        S^3_{ijk} = ∫∫∫ dX_i ⊗ dX_j ⊗ dX_k
    
    This uses the lead-lag transformation for better numerical stability.
    """
    if len(path) < 2:
        return np.array([np.nan])
    
    # Normalize path for numerical stability
    mean = np.mean(path, axis=0)
    std = np.std(path, axis=0) + 1e-10
    path_norm = (path - mean) / std
    
    # Compute increments
    increments = np.diff(path_norm, axis=0)
    n = len(increments)
    
    sig_terms = []
    
    # Level 0: S^0 = 1 (always)
    sig_terms.append(1.0)
    
    # Level 1: ∫ dX = sum of increments
    level1 = np.sum(increments, axis=0)
    sig_terms.extend(level1.tolist())
    
    if depth >= 2 and n > 1:
        # Level 2: ∫∫ dX⊗dX (approximated via Riemann sums)
        level2 = np.zeros((2, 2))
        cumsum = np.zeros_like(increments[0])
        for i in range(n):
            cumsum += increments[i]
            # Outer product with current increment
            level2 += np.outer(cumsum, increments[i])
        # Normalize by number of steps
        level2 = level2 / n
        sig_terms.extend(level2.flatten().tolist())
    
    if depth >= 3 and n > 2:
        # Level 3: third-order iterated integrals
        level3 = np.zeros((2, 2, 2))
        cumsum2 = np.zeros((2, 2))
        cumsum1 = np.zeros(2)
        for i in range(n):
            # Update cumulative sums
            cumsum1 += increments[i]
            # Update second-order cumsum
            cumsum2 += np.outer(cumsum1, increments[i])
            # Third-order term
            level3 += np.einsum('ij,k->ijk', cumsum2, increments[i])
        # Normalize
        level3 = level3 / (n ** 1.5)
        sig_terms.extend(level3.flatten().tolist())
    
    return np.array(sig_terms)


def signature_l_infinity_norm(
    returns: np.ndarray,
    depth: int = 3,
    lookahead: int = 5
) -> float:
    """
    Compute the L∞-norm of the truncated signature.
    """
    if len(returns) < 2:
        return np.nan
    
    # Construct path: time + returns (2D path)
    n = len(returns)
    t = np.linspace(0, 1, n)
    path = np.column_stack([t, returns])
    
    # Apply lead-lag transformation for better financial path representation
    # Lead-lag: (X_t, X_{t+1}) for each consecutive pair
    lead_lag = np.column_stack([path[:-1], path[1:]])
    # Reshape to 2D: each row is (time_t, return_t, time_{t+1}, return_{t+1})
    lead_lag_flat = lead_lag.reshape(-1, 4)
    
    # Compute signature terms on lead-lag path
    sig_terms = compute_path_signature_terms(lead_lag_flat, depth)
    
    # L∞ norm
    if len(sig_terms) == 0 or np.all(np.isnan(sig_terms)):
        return np.nan
    
    return float(np.max(np.abs(sig_terms)))


def rolling_signature_norms(
    returns: pd.Series,
    window_days: int,
    depth: int = 3,
    lookahead: int = 5
) -> pd.Series:
    """
    Compute rolling L∞-norm of path signature.
    """
    norms = []
    dates = []
    
    total_required = window_days + lookahead
    for i in range(total_required, len(returns)):
        # Window of returns for signature computation
        window_returns = returns.iloc[i - window_days:i].values
        lookahead_returns = returns.iloc[i:i + lookahead].values
        full_path = np.concatenate([window_returns, lookahead_returns])
        
        norm = signature_l_infinity_norm(full_path, depth, lookahead)
        if not np.isnan(norm) and norm > 1e-10:
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
    
    Formula: Return Level = u + (σ/ξ) * ((N*ζ_u)^ξ - 1)
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
