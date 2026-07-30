import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfApi
from datetime import date, timedelta
import config
import os

st.set_page_config(page_title="Rough Path EVT Engine", layout="wide")

st.markdown("""
<style>
.main-header{font-size:2.3rem;font-weight:700;color:#1a1a2e;margin-bottom:0.2rem}
.sub-header{font-size:1rem;color:#555;margin-bottom:1.5rem}
.uni-title{font-size:1.3rem;font-weight:600;margin-top:1rem;margin-bottom:0.8rem;
           padding-left:0.5rem;border-left:5px solid #27ae60}
.uni-title-risk{font-size:1.3rem;font-weight:600;margin-top:1rem;margin-bottom:0.8rem;
                padding-left:0.5rem;border-left:5px solid #e74c3c}
.hero-card-buy{background:linear-gradient(135deg,#1a472a 0%,#2d6a4f 60%,#40916c 100%);
               color:white;border-radius:16px;padding:1.2rem;margin:0.4rem;text-align:center;
               box-shadow:0 6px 20px rgba(39,174,96,0.3)}
.hero-card-sell{background:linear-gradient(135deg,#4a1a1a 0%,#6a2d2d 60%,#914040 100%);
                color:white;border-radius:16px;padding:1.2rem;margin:0.4rem;text-align:center;
                box-shadow:0 6px 20px rgba(231,76,60,0.3)}
.win-card-buy{background:linear-gradient(135deg,#1a472a 0%,#2d6a4f 100%);
              color:white;border-radius:16px;padding:1.2rem;margin:0.4rem;text-align:center;
              box-shadow:0 4px 12px rgba(39,174,96,0.3)}
.win-card-sell{background:linear-gradient(135deg,#4a1a1a 0%,#6a2d2d 100%);
               color:white;border-radius:16px;padding:1.2rem;margin:0.4rem;text-align:center;
               box-shadow:0 4px 12px rgba(231,76,60,0.3)}
.ticker{font-size:1.6rem;font-weight:800;letter-spacing:1px}
.score{font-size:0.9rem;margin-top:0.3rem;opacity:0.85}
.next-day{font-size:0.8rem;margin-top:0.2rem;opacity:0.7}
.risk-badge-high{background:#e74c3c;border-radius:6px;padding:2px 8px;font-size:0.75rem;
                font-weight:700;color:white}
.risk-badge-med{background:#f39c12;border-radius:6px;padding:2px 8px;font-size:0.75rem;
               font-weight:700;color:white}
.risk-badge-low{background:#27ae60;border-radius:6px;padding:2px 8px;font-size:0.75rem;
               font-weight:700;color:white}
.tail-badge{background:#8e44ad;border-radius:6px;padding:2px 8px;font-size:0.7rem;
            font-weight:700;color:white}
.tail-badge-heavy{background:#c0392b;border-radius:6px;padding:2px 8px;font-size:0.7rem;
                  font-weight:700;color:white}
.tail-badge-moderate{background:#f39c12;border-radius:6px;padding:2px 8px;font-size:0.7rem;
                     font-weight:700;color:white}
.tail-badge-thin{background:#27ae60;border-radius:6px;padding:2px 8px;font-size:0.7rem;
                 font-weight:700;color:white}
.buy-signal{background:#27ae60;border-radius:6px;padding:2px 12px;font-size:0.8rem;
            font-weight:700;color:white;display:inline-block}
.sell-signal{background:#e74c3c;border-radius:6px;padding:2px 12px;font-size:0.8rem;
             font-weight:700;color:white;display:inline-block}
.neutral-signal{background:#f39c12;border-radius:6px;padding:2px 12px;font-size:0.8rem;
                font-weight:700;color:white;display:inline-block}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">〜 Rough Path Extreme Value Theory Engine</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Path-signature tail risk · EVT on iterated integrals · '
    '1-in-100-year path-shape events · Cross-sectional z-scores</div>',
    unsafe_allow_html=True)

HF_TOKEN = config.HF_TOKEN or os.environ.get("HF_TOKEN", "")
RESULTS_REPO = config.RESULTS_REPO

US_HOLIDAYS = {
    date(2025,1,1),date(2025,1,20),date(2025,2,17),date(2025,4,18),
    date(2025,5,26),date(2025,6,19),date(2025,7,4),date(2025,9,1),
    date(2025,11,27),date(2025,12,25),
    date(2026,1,1),date(2026,1,19),date(2026,2,16),date(2026,4,3),
    date(2026,5,25),date(2026,6,19),date(2026,7,3),date(2026,9,7),
    date(2026,11,26),date(2026,12,25),
}

def next_trading_day() -> str:
    d = date.today() + timedelta(days=1)
    while d.weekday() >= 5 or d in US_HOLIDAYS:
        d += timedelta(days=1)
    return d.strftime("%B %d, %Y")

def risk_badge(score: float) -> str:
    if score > 1.5:   return f'<span class="risk-badge-high">EXTREME RISK</span>'
    elif score > 0.5: return f'<span class="risk-badge-med">ELEVATED RISK</span>'
    else:             return f'<span class="risk-badge-low">LOW RISK</span>'

def tail_badge(xi: float) -> str:
    if pd.isna(xi): return f'<span class="tail-badge">ξ = N/A</span>'
    if xi > 0.5:    return f'<span class="tail-badge-heavy">ξ = {xi:.3f} (VERY HEAVY)</span>'
    elif xi > 0.3:  return f'<span class="tail-badge-heavy">ξ = {xi:.3f} (HEAVY)</span>'
    elif xi > 0.0:  return f'<span class="tail-badge-moderate">ξ = {xi:.3f} (MODERATE)</span>'
    else:           return f'<span class="tail-badge-thin">ξ = {xi:.3f} (THIN)</span>'

def action_signal(score: float) -> str:
    if score < -1.0:    return f'<span class="buy-signal">🟢 STRONG BUY</span>'
    elif score < -0.5:  return f'<span class="buy-signal">🟢 BUY</span>'
    elif score < 0.5:   return f'<span class="neutral-signal">🟡 HOLD</span>'
    elif score < 1.0:   return f'<span class="sell-signal">🔴 REDUCE</span>'
    else:               return f'<span class="sell-signal">🔴 STRONG SELL</span>'


@st.cache_data(ttl=3600)
def list_repo_files():
    """List files in the results repo with better error handling."""
    if not HF_TOKEN:
        st.sidebar.warning("⚠️ HF_TOKEN not set")
        return []
    
    try:
        api = HfApi(token=HF_TOKEN)
        files = api.list_repo_files(
            repo_id=RESULTS_REPO,
            repo_type="dataset",
            token=HF_TOKEN
        )
        return files
    except Exception as e:
        st.sidebar.error(f"Error listing files: {str(e)}")
        return []


def find_latest(files, prefix):
    matches = sorted([f for f in files if f.endswith(".json") and prefix in f], reverse=True)
    return matches[0] if matches else None


@st.cache_data(ttl=3600)
def load_json_from_hf(path):
    """Load JSON from HuggingFace with better error handling."""
    if not HF_TOKEN:
        return {"error": "HF_TOKEN not set"}
    
    try:
        api = HfApi(token=HF_TOKEN)
        content = api.hf_hub_download(
            repo_id=RESULTS_REPO,
            filename=path,
            repo_type="dataset",
            token=HF_TOKEN
        )
        with open(content, 'r') as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 〜 Rough EVT")
st.sidebar.markdown(f"**Next Trading Day**")
st.sidebar.markdown(f"`{next_trading_day()}`")
st.sidebar.markdown(f"**Windows:** {config.WINDOW_DAYS}")
st.sidebar.markdown(f"**Signature Depth:** {config.SIGNATURE_DEPTH}")
st.sidebar.markdown(f"**Lookahead:** {config.LOOKAHEAD_DAYS} days")
st.sidebar.markdown(f"**EVT Threshold:** {config.EVT_THRESHOLD_QUANTILE:.0%}")
st.sidebar.markdown(f"**Return Period:** 1-in-{config.RETURN_PERIOD_YEARS} years")
st.sidebar.markdown("---")
st.sidebar.markdown("**Macro signals:**")
for col, desc, w, sign in config.MACRO_SIGNALS:
    arrow = "↑risk-on" if sign > 0 else "↑risk-off"
    st.sidebar.markdown(f"  • {col} ({arrow}, w={w:.0%})")

# ── Load data ─────────────────────────────────────────────────────────────────
files = list_repo_files()

if not files:
    st.error("No files found in the results repository. Please run trainer.py first.")
    st.info(f"Looking in: {RESULTS_REPO}")
    st.stop()

with st.sidebar.expander("📁 Available files", expanded=False):
    for f in files[:10]:
        st.code(f)

tab1_path = find_latest(files, "rough_evt_")
tab2_path = find_latest(files, "rough_evt_windows_")

if not tab1_path:
    st.error("No results found. Run trainer.py first.")
    st.info("Looking for files with prefix: rough_evt_")
    st.stop()

data1 = load_json_from_hf(tab1_path)
if "error" in data1:
    st.error(f"Error loading data: {data1['error']}")
    st.stop()

data2 = load_json_from_hf(tab2_path) if tab2_path else None
universes1 = data1.get("universes", {})
universes2 = data2.get("universes", {}) if data2 and "error" not in data2 else None

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Run date:** `{data1.get('run_date','?')}`")
st.sidebar.success(f"✅ Loaded {len(universes1)} universes")

tab1, tab2 = st.tabs(["🏆 Best Buys - Low Risk", "🔍 Per-Window Buys"])

UNIVERSE_ORDER = ["FI_COMMODITIES", "EQUITY_SECTORS", "COMBINED"]
UNIVERSE_LABELS = {
    "FI_COMMODITIES": "🏦 FI & Commodities",
    "EQUITY_SECTORS": "📈 Equity Sectors",
    "COMBINED": "🌐 Combined",
}

ntd = next_trading_day()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 - BEST BUYS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🏆 Best ETFs to ADD — Lowest Path-Shape Risk")

    with st.expander("📖 How to Read This", expanded=True):
        st.markdown("""
**This view shows the BEST ETFs to ADD to your portfolio** — those with the lowest path-shape risk.

| Metric | What to Look For | Signal |
|--------|------------------|--------|
| **z-score** | **NEGATIVE** z-score = lower risk than peers | 🟢 z < -0.5 = BUY |
| **Tail Index (ξ)** | **LOW** tail index = thinner tails, less extreme risk | 🟢 ξ < 0.2 = SAFER |
| **1-in-100yr** | **LOWER** return level = smaller worst-case path event | 🟢 Lower is better |

**Best case:** z-score < -1.0 + tail index < 0.2 = **STRONG BUY** (low risk, thin tails)

**Worst case:** z-score > 1.0 + tail index > 0.3 = **STRONG SELL** (high risk, heavy tails)
        """)

    for universe_name in UNIVERSE_ORDER:
        uni_data = universes1.get(universe_name, {})
        full_scores = uni_data.get("full_scores", {})
        
        if not full_scores:
            continue

        label = UNIVERSE_LABELS.get(universe_name, universe_name)
        
        # Sort by z-score (lowest = best to buy)
        sorted_etfs = sorted(
            [{"ticker": t, **info} for t, info in full_scores.items()],
            key=lambda x: x.get("z_score", 999)
        )
        
        # Top 3 best buys (lowest z-score)
        best_buys = sorted_etfs[:3]
        
        # Top 3 worst (highest z-score - to avoid)
        worst_risky = sorted_etfs[-3:][::-1]

        # Show Best Buys
        st.markdown(f'<div class="uni-title">🟢 {label} — Best Buys (Lowest Risk)</div>', unsafe_allow_html=True)
        
        cols = st.columns(3)
        for idx, etf in enumerate(best_buys):
            ticker = etf["ticker"]
            z_score = etf.get("z_score", 0)
            return_level = etf.get("return_level_100yr", 0)
            tail_index = etf.get("tail_index", 0)
            best_window = etf.get("best_window", "N/A")

            badge = risk_badge(z_score)
            tail_badge_html = tail_badge(tail_index)
            signal = action_signal(z_score)

            with cols[idx]:
                st.markdown(f"""
<div class="hero-card-buy">
  <div class="ticker">{ticker}</div>
  <div class="score">z-score = {z_score:+.3f}</div>
  <div class="score">{badge}</div>
  <div class="score">{signal}</div>
  <div class="score">{tail_badge_html}</div>
  <div class="score">1-in-100yr = {return_level:.2f}</div>
  <div class="score">best window = {best_window}d</div>
  <div class="next-day">📅 {ntd}</div>
</div>
""", unsafe_allow_html=True)

        # Show Worst (to avoid) in a smaller expander
        with st.expander(f"🔴 {label} — ETFs to AVOID (Highest Risk)"):
            rows = []
            for etf in worst_risky:
                rows.append({
                    "ETF": etf["ticker"],
                    "z-score": round(etf.get("z_score", 0), 4),
                    "1-in-100yr": round(etf.get("return_level_100yr", 0), 2),
                    "Tail Index (ξ)": round(etf.get("tail_index", 0), 4),
                    "Best Window": etf.get("best_window", "N/A"),
                    "Action": "SELL" if etf.get("z_score", 0) > 0.5 else "REDUCE"
                })
            df_rank = pd.DataFrame(rows).sort_values("z-score", ascending=False)
            st.dataframe(df_rank, use_container_width=True, hide_index=True)

        # Full ranking table
        with st.expander(f"📋 Full ranking — {label} (all ETFs)"):
            rows = []
            for etf in sorted_etfs:
                rows.append({
                    "ETF": etf["ticker"],
                    "z-score": round(etf.get("z_score", 0), 4),
                    "1-in-100yr": round(etf.get("return_level_100yr", 0), 2),
                    "Tail Index (ξ)": round(etf.get("tail_index", 0), 4),
                    "Best Window": etf.get("best_window", "N/A"),
                    "Signal": "BUY" if etf.get("z_score", 0) < -0.5 else ("HOLD" if etf.get("z_score", 0) < 0.5 else "SELL")
                })
            df_rank = pd.DataFrame(rows).sort_values("z-score", ascending=True)
            st.dataframe(df_rank, use_container_width=True, hide_index=True)
        st.divider()

    st.caption(f"Run date: {data1.get('run_date','?')} · "
               "Lowest z-score = Best to Buy · Path signature + EVT · "
               "Cross-sectional z-score · Tail index (ξ)")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 - PER-WINDOW BUYS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🔍 Best ETFs to ADD by Window")

    if not universes2:
        st.warning("Window-level data not found. Re-run trainer.")
        st.stop()

    all_wins = set()
    for ud in universes2.values():
        all_wins.update(ud.get("windows", {}).keys())
    win_options = sorted([int(w) for w in all_wins])

    if not win_options:
        st.error("No window data.")
        st.stop()

    win_labels = {
        63: "63d  (~3 months) — Tactical",
        252: "252d (~1 year) — Core Signal",
        504: "504d (~2 years) — Medium-Term",
        1008: "1008d (~4 years) — Structural",
        2016: "2016d (~8 years) — Secular",
        4032: "4032d (~16 years) — Long-Term",
        4536: "4536d (~18 years) — Full History",
    }

    default_idx = win_options.index(252) if 252 in win_options else 0
    selected_win = st.selectbox(
        "Select lookback window",
        options=win_options,
        index=default_idx,
        format_func=lambda w: win_labels.get(w, f"{w}d"),
    )
    win_key = str(selected_win)

    with st.expander("ℹ️ Window guidance — Which to use?", expanded=False):
        st.markdown("""
| Window | Best For | Trading Use |
|--------|----------|-------------|
| **63d** | Tactical trading | Short-term entries/exits |
| **252d** | Core signal | Primary allocation decisions |
| **504d** | Medium-term | Trend confirmation |
| **1008d** | Structural | Regime identification |
| **2016d+** | Secular | Strategic allocation |

**Rule of thumb:** Use 252d for primary signals, confirm with 63d for momentum.
        """)

    st.markdown(f"### Best Buys at **{win_labels.get(selected_win, f'{selected_win}d')}** window")

    for universe_name in UNIVERSE_ORDER:
        label = UNIVERSE_LABELS.get(universe_name, universe_name)
        uni_data = universes2.get(universe_name, {})
        win_data = uni_data.get("windows", {}).get(win_key)

        st.markdown(f'<div class="uni-title">{label}</div>', unsafe_allow_html=True)

        if not win_data:
            st.info(f"No data for {universe_name} at {selected_win}d.")
            st.divider()
            continue

        # Get full ranking and sort by z-score (lowest = best to buy)
        full_ranking = win_data.get("full_ranking", [])
        
        if not full_ranking:
            st.info(f"No ranking data for {universe_name} at {selected_win}d.")
            st.divider()
            continue

        # Parse ranking data
        parsed_etfs = []
        for item in full_ranking:
            if len(item) >= 4:
                # New format: [ticker, z_score, tail_index, return_level]
                parsed_etfs.append({
                    "ticker": item[0],
                    "z_score": item[1],
                    "tail_index": item[2],
                    "return_level": item[3]
                })
            else:
                # Old format: [ticker, z_score]
                ticker = item[0]
                z_score = item[1]
                # Try to get tail index from tab1
                tab1_data = universes1.get(universe_name, {}).get("full_scores", {}).get(ticker, {})
                parsed_etfs.append({
                    "ticker": ticker,
                    "z_score": z_score,
                    "tail_index": tab1_data.get("tail_index", 0),
                    "return_level": tab1_data.get("return_level_100yr", 0)
                })

        # Sort by z-score (lowest = best to buy)
        sorted_etfs = sorted(parsed_etfs, key=lambda x: x["z_score"])
        
        # Top 3 best buys
        best_buys = sorted_etfs[:3]
        
        # Top 3 worst (to avoid)
        worst_risky = sorted_etfs[-3:][::-1]

        # Show Best Buys
        st.markdown(f'#### 🟢 Top 3 Buys at {selected_win}d', unsafe_allow_html=True)
        
        cols = st.columns(3)
        for idx, etf in enumerate(best_buys):
            ticker = etf["ticker"]
            z_score = etf["z_score"]
            tail_index = etf["tail_index"]
            return_level = etf["return_level"]

            badge = risk_badge(z_score)
            tail_badge_html = tail_badge(tail_index)
            signal = action_signal(z_score)

            with cols[idx]:
                st.markdown(f"""
<div class="win-card-buy">
  <div class="ticker">{ticker}</div>
  <div class="score">z-score = {z_score:+.3f}</div>
  <div class="score">{badge}</div>
  <div class="score">{signal}</div>
  <div class="score">{tail_badge_html}</div>
  <div class="score">1-in-100yr = {return_level:.2f}</div>
  <div class="next-day">window = {selected_win}d · 📅 {ntd}</div>
</div>
""", unsafe_allow_html=True)

        # Show Worst (to avoid)
        with st.expander(f"🔴 {label} — ETFs to AVOID at {selected_win}d"):
            rows = []
            for etf in worst_risky:
                rows.append({
                    "ETF": etf["ticker"],
                    "z-score": round(etf["z_score"], 4),
                    "1-in-100yr": round(etf["return_level"], 2),
                    "Tail Index (ξ)": round(etf["tail_index"], 4),
                    "Action": "SELL" if etf["z_score"] > 0.5 else "REDUCE"
                })
            df_rank = pd.DataFrame(rows).sort_values("z-score", ascending=False)
            st.dataframe(df_rank, use_container_width=True, hide_index=True)

        # Full ranking table
        with st.expander(f"📋 Full ranking — {label} @ {selected_win}d (Lowest → Highest Risk)"):
            rows = []
            for etf in sorted_etfs:
                rows.append({
                    "Rank": len(rows) + 1,
                    "ETF": etf["ticker"],
                    "z-score": round(etf["z_score"], 4),
                    "Tail Index (ξ)": round(etf["tail_index"], 4),
                    "1-in-100yr": round(etf["return_level"], 2),
                    "Signal": "BUY" if etf["z_score"] < -0.5 else ("HOLD" if etf["z_score"] < 0.5 else "SELL")
                })
            df_rank = pd.DataFrame(rows)
            st.dataframe(df_rank, use_container_width=True, hide_index=True)
        st.divider()

    st.caption(f"Window: {selected_win}d · Run date: {data2.get('run_date','?')}")
