"""
Streamlit dashboard for Rough-EVT results.
Two tabs:
1. Tail Risk Summary - Best window per ETF with z-scores
2. Signature Distribution Explorer - Per-window analysis
"""

import streamlit as st
import pandas as pd
import json
import requests
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Rough Path EVT Dashboard",
    page_icon="📈",
    layout="wide"
)

# Constants
RESULTS_REPO = "P2SAMAPA/p2-rough-evt-results"
BASE_URL = f"https://huggingface.co/datasets/{RESULTS_REPO}/raw/main/"


@st.cache_data(ttl=3600)
def load_results(run_date: str = None):
    """Load the latest result files from HuggingFace."""
    if run_date is None:
        # Try to get latest
        run_date = datetime.now().strftime("%Y-%m-%d")
    
    tab1_url = f"{BASE_URL}rough_evt_{run_date}.json"
    tab2_url = f"{BASE_URL}rough_evt_windows_{run_date}.json"
    
    try:
        tab1 = requests.get(tab1_url).json()
        tab2 = requests.get(tab2_url).json()
        return tab1, tab2
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None


def render_tab1(data):
    """Render Tab 1: Best Window per ETF."""
    st.header("📊 Tail Risk Summary: Best Window per ETF")
    st.caption("Cross-sectional z-scores show relative tail risk within each universe")
    
    if not data or "universes" not in data:
        st.warning("No data available")
        return
    
    for universe_name, universe_data in data["universes"].items():
        st.subheader(f"🌐 {universe_name}")
        
        if not universe_data.get("full_scores"):
            st.info(f"No scores available for {universe_name}")
            continue
        
        # Create DataFrame
        rows = []
        for ticker, scores in universe_data["full_scores"].items():
            rows.append({
                "Ticker": ticker,
                "Z-Score": scores.get("z_score", 0),
                "Return Level (1-in-100yr)": scores.get("return_level_100yr", 0),
                "Best Window": scores.get("best_window", 0),
                "Tail Index (ξ)": scores.get("tail_index", 0)
            })
        
        df = pd.DataFrame(rows)
        df = df.sort_values("Z-Score", ascending=False)
        
        # Highlight top 5
        st.dataframe(
            df.style.background_gradient(subset=["Z-Score"], cmap="RdYlGn_r"),
            use_container_width=True
        )
        
        # Show top risky
        st.caption("🔥 Highest tail risk (top 5)")
        top = df.head(5)[["Ticker", "Z-Score", "Return Level (1-in-100yr)"]]
        st.dataframe(top, use_container_width=True)
        
        st.divider()


def render_tab2(data):
    """Render Tab 2: Explore by Window."""
    st.header("📐 Signature Distribution Explorer")
    st.caption("Analyze tail risk distribution across different rolling windows")
    
    if not data or "universes" not in data:
        st.warning("No data available")
        return
    
    # Select universe
    universes = list(data["universes"].keys())
    selected_universe = st.selectbox("Select Universe", universes)
    
    if selected_universe not in data["universes"]:
        return
    
    universe_data = data["universes"][selected_universe]
    windows_data = universe_data.get("windows", {})
    
    if not windows_data:
        st.info("No window data available")
        return
    
    # Select window
    window_options = list(windows_data.keys())
    selected_window = st.selectbox("Select Window (days)", window_options)
    
    if selected_window not in windows_data:
        return
    
    window_data = windows_data[selected_window]
    
    # Display ranking
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"🏆 Top 5 Risky (Window: {selected_window}d)")
        top = window_data.get("top_risky", [])
        if top:
            top_df = pd.DataFrame(top)
            st.dataframe(top_df, use_container_width=True)
        else:
            st.info("No top data")
    
    with col2:
        st.subheader("📊 Full Ranking")
        ranking = window_data.get("full_ranking", [])
        if ranking:
            rank_df = pd.DataFrame(ranking, columns=["Ticker", "Z-Score"])
            rank_df = rank_df.sort_values("Z-Score", ascending=False)
            st.dataframe(rank_df, use_container_width=True)
        else:
            st.info("No ranking data")
    
    # Distribution plot (if we had more data)
    st.subheader("📈 Tail Risk Distribution")
    st.caption("Z-scores distribution across all tickers for this window")
    
    if ranking:
        scores = [r[1] for r in ranking if not pd.isna(r[1])]
        if scores:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.hist(scores, bins=20, edgecolor='black', alpha=0.7)
            ax.axvline(0, color='red', linestyle='--', label='Mean')
            ax.set_xlabel("Z-Score")
            ax.set_ylabel("Frequency")
            ax.set_title(f"Distribution of Tail Risk Z-Scores ({selected_window}d)")
            ax.legend()
            st.pyplot(fig)


def main():
    st.title("🔮 Rough Path EVT Dashboard")
    st.caption("Extreme Value Theory applied to path signatures")
    
    # Load data
    run_date = datetime.now().strftime("%Y-%m-%d")
    tab1_data, tab2_data = load_results(run_date)
    
    if tab1_data is None or tab2_data is None:
        st.error("Failed to load data. Please check the results repository.")
        st.info(f"Looking for: rough_evt_{run_date}.json and rough_evt_windows_{run_date}.json")
        return
    
    # Create tabs
    tab1, tab2 = st.tabs(["📊 Tail Risk Summary", "📐 Per-Window Explorer"])
    
    with tab1:
        render_tab1(tab1_data)
    
    with tab2:
        render_tab2(tab2_data)
    
    # Footer
    st.divider()
    st.caption(f"Data from {RESULTS_REPO} | Run date: {run_date}")
    st.caption("Powered by signatory (path signatures) + scipy (GPD fitting)")


if __name__ == "__main__":
    main()
