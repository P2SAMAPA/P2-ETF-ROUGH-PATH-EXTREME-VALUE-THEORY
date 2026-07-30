"""
Core Rough Path EVT engine:
- Path signature computation via signatory
- L∞-norm of signature over sliding windows
- GPD fitting to extreme values
- Return level calculation
"""

import numpy as np
import pandas as pd
import torch
import signatory
from scipy.stats import genpareto
from scipy.optimize import minimize
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns from raw prices."""
    return np.log(prices / prices.shift(1)).dropna()


def path_signature_norm(
    returns: np.ndarray,
    depth: int = 3,
    lookahead: int = 5
) -> float:
    """
    Compute the L∞-norm of the truncated signature for a path segment.
    
    Args:
        returns: Array of log returns over the window
        depth: Truncation depth of the signature
        lookahead: Forward horizon to include in path shape
        
    Returns:
        L∞-norm of the truncated signature
    """
    if len(returns) < 2:
        return np.nan
    
    # Construct path: time + returns (2D path)
    n = len(returns)
    # Normalize time to [0,1] for numerical stability
    t = np.linspace(0, 1, n)
    path = np.column_stack([t, returns])
    
    # Convert to torch tensor
    path_tensor = torch.tensor(path, dtype=torch.float32).unsqueeze(0)  # (1, T, 2)
    
    try:
        # Compute signature up to depth
        sig = signatory.signature(
            path_tensor,
            depth=depth,
            basepoint=True
        )
        # L∞ norm of the signature vector
        sig_norm = torch.max(torch.abs(sig)).item()
        return sig_norm
    except Exception:
        return np.nan


def rolling_signature_norms(
    returns: pd.Series,
    window_days: int,
    depth: int = 3,
    lookahead: int = 5
) -> pd.Series:
    """
    Compute rolling L∞-norm of path signature.
    
    Args:
        returns: Log returns series
        window_days: Rolling window size
        depth: Signature truncation depth
        lookahead: Forward horizon
        
    Returns:
        Series of signature norms (aligned with returns index)
    """
    norms = []
    dates = []
    
    for i in range(window_days + lookahead, len(returns)):
        # Window of returns for signature computation
        window_returns = returns.iloc[i - window_days:i].values
        
        # Include lookahead returns to capture forward path shape
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
    """
    Fit Generalized Pareto Distribution to exceedances over threshold.
    
    Returns:
        threshold: The threshold value
        xi: Shape parameter (tail index)
        sigma: Scale parameter
        n_exceedances: Number of exceedances
        exceedance_prob: Probability of exceeding threshold (ζ_u)
    """
    if len(data) < 20:
        return np.nan, np.nan, np.nan, 0, np.nan
    
    threshold = np.quantile(data, threshold_quantile)
    exceedances = data[data > threshold] - threshold
    
    if len(exceedances) < 10:
        return threshold, np.nan, np.nan, len(exceedances), np.nan
    
    try:
        # Fit GPD using MLE
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
    
    Args:
        xi: Shape parameter (tail index)
        sigma: Scale parameter
        exceed_prob: Probability of exceeding threshold (ζ_u)
        n_years: Return period in years
        data_points_per_year: Trading days per year
        
    Returns:
        Return level (in log-return units)
    """
    if any(np.isnan([xi, sigma, exceed_prob])) or exceed_prob <= 0:
        return np.nan
    
    # N-year return period in terms of data points
    N = n_years * data_points_per_year
    
    # Return level formula: u + (σ/ξ) * ((N*ζ_u)^ξ - 1)
    # Note: ζ_u = P(X > u)
    if abs(xi) < 1e-10:
        # Gumbel limit (ξ → 0)
        return_level_val = sigma * np.log(N * exceed_prob)
    else:
        return_level_val = (sigma / xi) * ((N * exceed_prob) ** xi - 1)
    
    return max(0, return_level_val)  # Return level should be non-negative


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
    
    Returns:
        Dictionary with:
        - return_level_100yr: 1-in-100-year signature norm
        - tail_index: GPD shape parameter (ξ)
        - threshold: EVT threshold value
        - exceedances: Number of exceedances
        - best_window: The window used
        - lookahead: The lookahead used
    """
    # Compute log returns
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
    
    # Compute rolling signature norms
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
    
    # Fit GPD
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
    
    # Compute return level
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


def compute_cross_sectional_zscore(
    scores: Dict[str, float]
) -> Dict[str, float]:
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
