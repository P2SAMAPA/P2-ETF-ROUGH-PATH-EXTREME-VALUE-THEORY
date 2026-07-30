
---

### 2. `config.py`
```python
"""
Configuration for Rough Path EVT Engine.
"""

# ============================================================
# UNIVERSES
# ============================================================

UNIVERSES = {
    "FI_COMMODITIES": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"
    ],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI",
        "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "URA", "SOXX", "SMH",
        "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE"
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI",
        "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "URA", "SOXX", "SMH",
        "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE"
    ]
}

# ============================================================
# SIGNATURE CONFIGURATION
# ============================================================

SIGNATURE_DEPTH = 3          # Truncation depth of path signature
WINDOW_DAYS = [63, 252]      # Rolling windows for signature computation
LOOKAHEAD_DAYS = 5           # Forward horizon for path shape

# ============================================================
# EVT CONFIGURATION
# ============================================================

EVT_THRESHOLD_QUANTILE = 0.95   # Quantile for POT threshold
RETURN_PERIOD_YEARS = 100       # 1-in-100-year event
MIN_EXCEEDANCES = 10            # Minimum exceedances required for GPD fit

# ============================================================
# MACRO SIGNALS (for potential regime weighting)
# ============================================================

MACRO_COLS_CORE = ["VIX", "T10Y2Y", "DXY", "IG_SPREAD", "HY_SPREAD"]
MACRO_WEIGHTS = {
    "VIX": 0.30,
    "T10Y2Y": 0.25,
    "DXY": 0.20,
    "IG_SPREAD": 0.15,
    "HY_SPREAD": 0.10
}

# ============================================================
# HUGGINGFACE PATHS
# ============================================================

DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
RESULTS_REPO = "P2SAMAPA/p2-rough-evt-results"
