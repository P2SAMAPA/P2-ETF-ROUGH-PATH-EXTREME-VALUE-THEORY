"""
Data manager for HuggingFace dataset loading.
Follows the exact same pattern as the Hilbert-PLV engine.
"""

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
import os
from config import DATA_REPO, MACRO_COLS_CORE


def load_data(hf_token: str = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load price and macro data from HuggingFace master dataset.
    
    Returns:
        prices_df: DataFrame with price columns for all tickers
        macro_df: DataFrame with macro indicator columns
    """
    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN is required")

    # Load prices
    prices_path = hf_hub_download(
        repo_id=DATA_REPO,
        filename="prices.parquet",
        token=token
    )
    prices_df = pd.read_parquet(prices_path)
    prices_df.index = pd.to_datetime(prices_df.index)
    
    # Load macro data
    macro_path = hf_hub_download(
        repo_id=DATA_REPO,
        filename="macro.parquet",
        token=token
    )
    macro_df = pd.read_parquet(macro_path)
    macro_df.index = pd.to_datetime(macro_df.index)
    
    # Align dates
    common_dates = prices_df.index.intersection(macro_df.index)
    prices_df = prices_df.loc[common_dates]
    macro_df = macro_df.loc[common_dates]
    
    # Drop rows with missing core macro columns only (preserve history)
    macro_df = macro_df.dropna(subset=MACRO_COLS_CORE)
    prices_df = prices_df.loc[macro_df.index]
    
    return prices_df, macro_df


def get_available_tickers(prices_df: pd.DataFrame, universe: list) -> list:
    """
    Get tickers from a universe that are available in the prices dataframe.
    """
    return [t for t in universe if t in prices_df.columns]
