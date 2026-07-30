"""
trainer.py  —  Orchestrator for Rough-EVT pipeline
===================================================

Loads data → computes EVT scores for all tickers/windows →
builds JSON outputs → uploads to HuggingFace.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from huggingface_hub import HfApi

import config
from data_manager import load_master_data, validate_data
from rough_evt import compute_rough_evt, compute_cross_sectional_zscore
from push_results import upload_results

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Runner ──────────────────────────────────────────────────────────────────

def run_trainer(hf_token: Optional[str] = None) -> Dict:
    """
    Run the full Rough-EVT pipeline.
    """
    token = hf_token or config.HF_TOKEN or os.environ.get("HF_TOKEN")
    if not token:
        logger.warning("HF_TOKEN not set — will skip HuggingFace upload.")

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("🔄 Loading master data from HuggingFace...")
    try:
        prices_df, macro_df = load_master_data(token)
        validate_data(prices_df, macro_df)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

    logger.info(
        f"✅ Loaded {len(prices_df)} days, "
        f"{len(prices_df.columns)} ETFs, "
        f"{len(macro_df.columns)} macro cols"
    )

    run_date = datetime.now().strftime("%Y-%m-%d")

    # ── Results containers ────────────────────────────────────────────────────
    results_tab1 = {
        "run_date": run_date,
        "universes": {}
    }

    results_tab2 = {
        "run_date": run_date,
        "universes": {}
    }

    # ── Process each universe ─────────────────────────────────────────────────
    for universe_name, tickers in config.UNIVERSES.items():
        logger.info(f"\n📊 Processing universe: {universe_name}")

        # Filter available tickers
        available = [t for t in tickers if t in prices_df.columns]
        logger.info(f"   Available: {len(available)}/{len(tickers)}")

        if not available:
            continue

        # Tab 1: Best window per ticker
        tab1_universe = {
            "top_risky": [],
            "full_scores": {}
        }

        # Tab 2: Per-window results
        tab2_universe = {
            "windows": {}
        }

        # Store per-ticker per-window results
        all_window_scores = {str(w): {} for w in config.WINDOW_DAYS}
        ticker_best_scores = {}

        # ── Compute for each ticker ────────────────────────────────────────────
        for ticker in available:
            logger.info(f"   Computing {ticker}...")
            prices = prices_df[ticker]
            ticker_scores = []

            for window in config.WINDOW_DAYS:
                result = compute_rough_evt(
                    prices,
                    window_days=window,
                    depth=config.SIGNATURE_DEPTH,
                    lookahead=config.LOOKAHEAD_DAYS,
                    threshold_quantile=config.EVT_THRESHOLD_QUANTILE,
                    return_period_years=config.RETURN_PERIOD_YEARS
                )

                if result.get("error") is None:
                    ticker_scores.append({
                        "window": window,
                        "return_level": result["return_level_100yr"],
                        "tail_index": result["tail_index"],
                        "threshold": result["threshold"],
                        "exceedances": result["exceedances"]
                    })
                    all_window_scores[str(window)][ticker] = result["return_level_100yr"]

            # Keep best window (max return level)
            if ticker_scores:
                best = max(ticker_scores, key=lambda x: x["return_level"])
                ticker_best_scores[ticker] = {
                    "score": best["return_level"],
                    "best_window": best["window"],
                    "tail_index": best["tail_index"],
                    "threshold": best["threshold"],
                    "exceedances": best["exceedances"]
                }

        # ── Cross-sectional z-scores ──────────────────────────────────────────
        if ticker_best_scores:
            raw_scores = {t: d["score"] for t, d in ticker_best_scores.items()
                         if not np.isnan(d["score"])}
            z_scores = compute_cross_sectional_zscore(raw_scores)

            # Top 5 risky
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

            # Tab 2: Per-window rankings
            for window_str, window_scores in all_window_scores.items():
                if window_scores:
                    z_win = compute_cross_sectional_zscore(window_scores)
                    top_win = sorted(
                        [(t, z_win[t]) for t in z_win if not np.isnan(z_win[t])],
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]

                    tab2_universe["windows"][window_str] = {
                        "top_risky": [
                            {"ticker": t, "z_score": z} for t, z in top_win
                        ],
                        "full_ranking": [
                            [t, z_win[t]] for t in z_win if not np.isnan(z_win[t])
                        ]
                    }

        results_tab1["universes"][universe_name] = tab1_universe
        results_tab2["universes"][universe_name] = tab2_universe

    # ── Save JSON files ──────────────────────────────────────────────────────
    logger.info("\n💾 Saving JSON results...")

    tab1_path = f"rough_evt_{run_date}.json"
    tab2_path = f"rough_evt_windows_{run_date}.json"

    with open(tab1_path, "w") as f:
        json.dump(results_tab1, f, indent=2)

    with open(tab2_path, "w") as f:
        json.dump(results_tab2, f, indent=2)

    logger.info(f"   Saved: {tab1_path}")
    logger.info(f"   Saved: {tab2_path}")

    # ── Upload to HuggingFace ───────────────────────────────────────────────
    if token:
        logger.info("\n📤 Uploading results to HuggingFace...")
        try:
            upload_results(tab1_path, tab2_path, token)
        except Exception as e:
            logger.error(f"   Upload failed: {e}")
    else:
        logger.info("\n📤 Skipping upload (no HF_TOKEN)")

    return {"tab1": results_tab1, "tab2": results_tab2}


if __name__ == "__main__":
    run_trainer()
