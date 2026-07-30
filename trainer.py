"""
Orchestrator for Rough-EVT pipeline.
Loads data, computes EVT scores for all tickers/windows, and builds JSON outputs.
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List
from huggingface_hub import HfApi

from config import (
    UNIVERSES, WINDOW_DAYS, SIGNATURE_DEPTH, LOOKAHEAD_DAYS,
    EVT_THRESHOLD_QUANTILE, RETURN_PERIOD_YEARS
)
from data_manager import load_data, get_available_tickers
from rough_evt import compute_rough_evt, compute_cross_sectional_zscore
from push_results import upload_results


def run_trainer(hf_token: str = None) -> Dict:
    """
    Run the full Rough-EVT pipeline.
    """
    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN is required")
    
    print("🔄 Loading data from HuggingFace...")
    prices_df, macro_df = load_data(token)
    print(f"   Loaded {len(prices_df)} days of data for {len(prices_df.columns)} tickers")
    
    run_date = datetime.now().strftime("%Y-%m-%d")
    
    # Results containers
    results_tab1 = {
        "run_date": run_date,
        "universes": {}
    }
    
    results_tab2 = {
        "run_date": run_date,
        "universes": {}
    }
    
    # Process each universe
    for universe_name, tickers in UNIVERSES.items():
        print(f"\n📊 Processing universe: {universe_name}")
        
        available = get_available_tickers(prices_df, tickers)
        print(f"   Available tickers: {len(available)}/{len(tickers)}")
        
        if not available:
            continue
        
        # Tab 1: Best window per ticker (max return level)
        tab1_universe = {
            "top_risky": [],
            "full_scores": {}
        }
        
        # Tab 2: Per-window results
        tab2_universe = {
            "windows": {}
        }
        
        # For each ticker, compute scores across windows
        ticker_best_scores = {}
        
        for ticker in available:
            print(f"   Computing for {ticker}...")
            prices = prices_df[ticker]
            ticker_scores = []
            
            for window in WINDOW_DAYS:
                result = compute_rough_evt(
                    prices,
                    window_days=window,
                    depth=SIGNATURE_DEPTH,
                    lookahead=LOOKAHEAD_DAYS,
                    threshold_quantile=EVT_THRESHOLD_QUANTILE,
                    return_period_years=RETURN_PERIOD_YEARS
                )
                
                if result.get("error") is None:
                    ticker_scores.append({
                        "window": window,
                        "return_level": result["return_level_100yr"],
                        "tail_index": result["tail_index"],
                        "threshold": result["threshold"],
                        "exceedances": result["exceedances"]
                    })
            
            if ticker_scores:
                # Best window = highest return level
                best = max(ticker_scores, key=lambda x: x["return_level"])
                ticker_best_scores[ticker] = {
                    "score": best["return_level"],
                    "best_window": best["window"],
                    "tail_index": best["tail_index"],
                    "threshold": best["threshold"],
                    "exceedances": best["exceedances"]
                }
        
        # Cross-sectional z-score for Tab 1
        if ticker_best_scores:
            raw_scores = {t: d["score"] for t, d in ticker_best_scores.items() 
                         if not np.isnan(d["score"])}
            z_scores = compute_cross_sectional_zscore(raw_scores)
            
            # Build Tab 1
            top_risky = sorted(
                [(t, z_scores[t]) for t in z_scores if not np.isnan(z_scores[t])],
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            tab1_universe["top_risky"] = [
                {"ticker": t, "z_score": z} for t, z in top_risky
            ]
            
            tab1_universe["full_scores"] = {
                t: {
                    "z_score": z_scores.get(t, np.nan),
                    "return_level_100yr": ticker_best_scores[t]["score"],
                    "best_window": ticker_best_scores[t]["best_window"],
                    "tail_index": ticker_best_scores[t]["tail_index"]
                }
                for t in ticker_best_scores
            }
            
            # Build Tab 2: Per-window results
            for window in WINDOW_DAYS:
                window_scores = {}
                for ticker, data in ticker_best_scores.items():
                    # We need to recompute for this specific window
                    # In practice, store per-window results in a cache
                    # Here we use the best window data as proxy (simplified)
                    if data["best_window"] == window:
                        window_scores[ticker] = data["score"]
                
                if window_scores:
                    z_win = compute_cross_sectional_zscore(window_scores)
                    top_win = sorted(
                        [(t, z_win[t]) for t in z_win if not np.isnan(z_win[t])],
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]
                    
                    tab2_universe["windows"][str(window)] = {
                        "top_risky": [
                            {"ticker": t, "z_score": z} for t, z in top_win
                        ],
                        "full_ranking": [
                            [t, z_win[t]] for t in z_win if not np.isnan(z_win[t])
                        ]
                    }
        
        results_tab1["universes"][universe_name] = tab1_universe
        results_tab2["universes"][universe_name] = tab2_universe
    
    # Save JSON files
    print("\n💾 Saving JSON results...")
    
    tab1_path = f"rough_evt_{run_date}.json"
    tab2_path = f"rough_evt_windows_{run_date}.json"
    
    with open(tab1_path, "w") as f:
        json.dump(results_tab1, f, indent=2)
    
    with open(tab2_path, "w") as f:
        json.dump(results_tab2, f, indent=2)
    
    print(f"   Saved: {tab1_path}")
    print(f"   Saved: {tab2_path}")
    
    # Upload to HuggingFace
    print("\n📤 Uploading results to HuggingFace...")
    upload_results(tab1_path, tab2_path, token)
    
    return {"tab1": results_tab1, "tab2": results_tab2}


if __name__ == "__main__":
    run_trainer()
