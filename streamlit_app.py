import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfFileSystem
from datetime import date, timedelta
import config

st.set_page_config(page_title="Rough Path EVT Engine", layout="wide")

st.markdown("""
<style>
.main-header{font-size:2.3rem;font-weight:700;color:#1a1a2e;margin-bottom:0.2rem}
.sub-header{font-size:1rem;color:#555;margin-bottom:1.5rem}
.uni-title{font-size:1.3rem;font-weight:600;margin-top:1rem;margin-bottom:0.8rem;
           padding-left:0.5rem;border-left:5px solid #e94560}
.hero-card{background:linear-gradient(135deg,#1a1a2e 0%,#16213e 60%,#0f3460 100%);
           color:white;border-radius:16px;padding:1.2rem;margin:0.4rem;text-align:center;
           box-shadow:0 6px 20px rgba(233,69,96,0.3)}
.win-card{background:linear-gradient(135deg,#0f3460 0%,#533483 100%);color:white;
          border-radius:16px;padding:1.2rem;margin:0.4rem;text-align:center;
          box-shadow:0 4px 12px rgba(83,52,131,0.3)}
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
.metric-box{background:#f8f9fa;border-radius:10px;padding:0.8rem;margin:0.3rem 0;
            border-left:4px solid #e94560}
.metric-label{font-size:0.75rem;color:#666;text-transform:uppercase;letter-spacing:0.5px}
.metric-value{font-size:1.1rem;font-weight:700;color:#1a1a2e}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">〜 Rough Path Extreme Value Theory Engine</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Path-signature tail risk · EVT on iterated integrals · '
    '1-in-100-year path-shape events · Cross-sectional z-scores</div>',
    unsafe_allow_html=True)

HF_TOKEN    = config.HF_TOKEN
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
    if xi > 0.3:    return f'<span class="tail-badge">ξ = {xi:.3f} (HEAVY)</span>'
    elif xi > 0.0:  return f'<span class="tail-badge">ξ = {xi:.3f} (MODERATE)</span>'
    else:           return f'<span class="tail-badge">ξ = {xi:.3f} (THIN)</span>'


@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        return [f["name"] for f in fs.ls(f"datasets/{RESULTS_REPO}",
                                          detail=True, recursive=True)
                if f["type"] == "file"]
    except Exception as e:
        return []


def find_latest(files, prefix):
    matches = sorted([f for f in files if f.endswith(".json") and prefix in f], reverse=True)
    return matches[0] if matches else None


@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
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
st.sidebar.markdown("**Macro signals:**")
for col, desc, w, sign in config.MACRO_SIGNALS:
    arrow = "↑risk-on" if sign > 0 else "↑risk-off"
    st.sidebar.markdown(f"  • {col} ({arrow}, w={w:.0%})")

# ── Load data ─────────────────────────────────────────────────────────────────
files     = list_repo_files()
tab1_path = find_latest(files, "rough_evt_")
tab2_path = find_latest(files, "rough_evt_windows_")

if not tab1_path:
    st.error("No results found. Run trainer.py first.")
    st.stop()

data1 = load_json(tab1_path)
if "error" in data1:
    st.error(f"Error loading data: {data1['error']}")
    st.stop()

data2      = load_json(tab2_path) if tab2_path else None
universes1 = data1["universes"]
universes2 = data2["universes"] if data2 and "error" not in data2 else None

st.sidebar.markdown(f"**Run date:** `{data1.get('run_date','?')}`")

tab1, tab2 = st.tabs(["🏆 Tail Risk Summary", "🔍 Per-Window Explorer"])

UNIVERSE_ORDER  = ["FI_COMMODITIES", "EQUITY_SECTORS", "COMBINED"]
UNIVERSE_LABELS = {
    "FI_COMMODITIES": "🏦 FI & Commodities",
    "EQUITY_SECTORS": "📈 Equity Sectors",
    "COMBINED":       "🌐 Combined",
}

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🏆 Tail Risk Summary — Worst-Case Path-Shape Events")

    with st.expander("📖 How Rough Path EVT Works", expanded=True):
        st.markdown("""
**Rough Path Extreme Value Theory (EVT)** applies extreme value theory directly to the **signature of a rough path** — treating an asset's price trajectory as a geometric object, not just a terminal scalar.

| Step | What happens |
|------|-------------|
| 1. Log returns | Convert prices to log-returns |
| 2. Path signature | Compute iterated integrals of the path (depth 3) |
| 3. L∞-norm | Take the maximum absolute value of all signature terms |
| 4. Peaks-Over-Threshold | Select exceedances above the 95th percentile |
| 5. GPD fit | Fit Generalized Pareto Distribution to the tail |
| 6. Return level | Compute 1-in-100-year event for the path shape |

**High z-score + heavy tail (ξ > 0.3)** → ETF has extreme path-shape risk → **CAUTION**

**Low z-score + thin tail (ξ < 0)** → ETF has stable path geometry → **SAFER**

**Tail index (ξ):** ξ > 0 = Fréchet (fat tail) · ξ = 0 = Gumbel · ξ < 0 = Weibull (bounded)
        """)

    ntd = next_trading_day()

    for universe_name in UNIVERSE_ORDER:
        uni_data = universes1.get(universe_name, {})
        top_risky = uni_data.get("top_risky", [])
        if not top_risky:
            continue

        label = UNIVERSE_LABELS.get(universe_name, universe_name)
        st.markdown(f'<div class="uni-title">{label}</div>', unsafe_allow_html=True)

        cols = st.columns(3)
        for idx, etf in enumerate(top_risky):
            ticker = etf["ticker"]
            z_score = etf["z_score"]
            
            # Get full data for this ticker
            full_data = uni_data.get("full_scores", {}).get(ticker, {})
            return_level = full_data.get("return_level_100yr", 0)
            tail_index = full_data.get("tail_index", 0)
            best_window = full_data.get("best_window", "N/A")
            
            badge = risk_badge(z_score)
            tail_badge_html = tail_badge(tail_index)
            
            with cols[idx]:
                st.markdown(f"""
<div class="hero-card">
  <div class="ticker">{ticker}</div>
  <div class="score">z-score = {z_score:+.3f}</div>
  <div class="score">{badge}</div>
  <div class="score">1-in-100yr = {return_level:.4f}</div>
  <div class="score">{tail_badge_html}</div>
  <div class="score">best window = {best_window}d</div>
  <div class="next-day">📅 {ntd}</div>
</div>
""", unsafe_allow_html=True)

        with st.expander(f"📋 Full ranking — {label}"):
            full = uni_data.get("full_scores", {})
            if full:
                rows = []
                for t, info in full.items():
                    rows.append({
                        "ETF": t,
                        "z-score": round(info.get("z_score", 0), 4),
                        "1-in-100yr": round(info.get("return_level_100yr", 0), 4),
                        "Tail Index (ξ)": round(info.get("tail_index", 0), 4),
                        "Best Window (d)": info.get("best_window", "N/A")
                    })
                df_rank = pd.DataFrame(rows).sort_values("z-score", ascending=False)
                st.dataframe(df_rank, use_container_width=True, hide_index=True)
        st.divider()

    st.caption(f"Run date: {data1.get('run_date','?')} · "
               "Path signature + EVT · 1-in-100-year return level · "
               "Cross-sectional z-score · Tail index (ξ)")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🔍 Explore Tail Risk by Window")

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
        63:   "63d  (~3 months)",
        252:  "252d (~1 year)",
        504:  "504d (~2 years)",
        1008: "1008d (~4 years)",
        2016: "2016d (~8 years)",
        4032: "4032d (~16 years)",
        4536: "4536d (~18 years)",
    }

    default_idx  = win_options.index(252) if 252 in win_options else 0
    selected_win = st.selectbox(
        "Select lookback window",
        options=win_options,
        index=default_idx,
        format_func=lambda w: win_labels.get(w, f"{w}d"),
    )
    win_key = str(selected_win)

    with st.expander("ℹ️ Window guidance", expanded=False):
        st.markdown("""
- **63d** — Short-term path-shape risk: captures recent geometric patterns
- **252d** — Annual path risk: recommended primary signal for tail events
- **504d–1008d** — Medium-term structural path regimes
- **2016d+** — Very long-run path geometry (secular cycles)
- **4032d / 4536d** — Full history path shape analysis (2008–present)
        """)

    st.markdown(f"### Tail Risk Rankings at **{win_labels.get(selected_win, f'{selected_win}d')}** window")

    for universe_name in UNIVERSE_ORDER:
        label    = UNIVERSE_LABELS.get(universe_name, universe_name)
        uni_data = universes2.get(universe_name, {})
        win_data = uni_data.get("windows", {}).get(win_key)

        st.markdown(f'<div class="uni-title">{label}</div>', unsafe_allow_html=True)

        if not win_data:
            st.info(f"No data for {universe_name} at {selected_win}d.")
            st.divider()
            continue

        cols = st.columns(3)
        for idx, etf in enumerate(win_data.get("top_risky", [])):
            ticker = etf["ticker"]
            z_score = etf["z_score"]
            badge = risk_badge(z_score)
            
            with cols[idx]:
                st.markdown(f"""
<div class="win-card">
  <div class="ticker">{ticker}</div>
  <div class="score">z-score = {z_score:+.3f}</div>
  <div class="score">{badge}</div>
  <div class="next-day">window = {selected_win}d · 📅 {ntd}</div>
</div>
""", unsafe_allow_html=True)

        with st.expander(f"📋 Full ranking — {label} @ {selected_win}d"):
            rows = win_data.get("full_ranking", [])
            if rows:
                df_win = pd.DataFrame(rows)
                df_win.columns = ["ETF", "z-score"]
                df_win.insert(0, "Rank", range(1, len(df_win) + 1))
                st.dataframe(df_win, use_container_width=True, hide_index=True)
        st.divider()

    st.caption(f"Window: {selected_win}d · Run date: {data2.get('run_date','?')}")
