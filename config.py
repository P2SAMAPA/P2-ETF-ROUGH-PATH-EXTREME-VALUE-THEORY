"""
config.py  —  Configuration for Rough Path EVT Engine
=======================================================

Defines:
  - UNIVERSES: ETF ticker sets
  - MACRO_SIGNALS: macro columns with weights and regime signs
  - SIGNATURE_DEPTH: truncation level of path signature
  - WINDOW_DAYS: rolling windows for signature computation
  - EVT parameters: threshold quantile, return period years
"""

# ── HuggingFace ──────────────────────────────────────────────────────────────

HF_TOKEN = ""  # set via env var HF_TOKEN, or inline for local dev

DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
RESULTS_REPO = "P2SAMAPA/p2-rough-evt-results"  # output repo


# ── ETF Universes ────────────────────────────────────────────────────────────

UNIVERSES = {
    "FI_COMMODITIES": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
    ],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI",
        "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "URA", "SMH",
        "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI",
        "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "URA", "SMH",
        "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
}


# ── Macro Signals ────────────────────────────────────────────────────────────
# Format: (column_name, display_name, weight, regime_sign)
# regime_sign: +1 = risk-on, -1 = risk-off

MACRO_SIGNALS = [
    ("VIX",       "VIX",           0.30, -1.0),  # rising VIX = risk-off
    ("T10Y2Y",    "10Y–2Y Spread", 0.25, +1.0),  # steepening = risk-on
    ("DXY",       "DXY",           0.20, -1.0),  # rising DXY = risk-off
    ("IG_SPREAD", "IG Spread",     0.15, -1.0),  # widening = risk-off
    ("HY_SPREAD", "HY Spread",     0.10, -1.0),  # widening = risk-off
]

# Backward-compatible names for data_manager.py
MACRO_COLS_CORE = ["VIX", "T10Y2Y", "DXY"]
MACRO_COLS_EXTENDED = ["IG_SPREAD", "HY_SPREAD"]


# ── Signature / EVT Configuration ──────────────────────────────────────────

SIGNATURE_DEPTH = 3           # Truncation depth of path signature
WINDOW_DAYS = [63, 252]       # Rolling windows for signature computation
LOOKAHEAD_DAYS = 5            # Forward horizon for path shape

# EVT parameters
EVT_THRESHOLD_QUANTILE = 0.95 # Quantile for POT threshold
RETURN_PERIOD_YEARS = 100     # 1-in-100-year event
MIN_EXCEEDANCES = 10          # Minimum exceedances required for GPD fit
