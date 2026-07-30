# P2-ROUGH-EVT

**Rough Path Extreme Value Theory — Path-Signature Tail Risk Engine**

Part of the **P2Quant Engine Suite** · P2SAMAPA

---

## What This Engine Does

This engine applies **Extreme Value Theory (EVT)** directly to the **signature of a rough path** — treating an asset's price trajectory over a window as a geometric object, not just a terminal scalar.

Classical EVT for time series relies on filtering to i.i.d. residuals. This engine defines an extreme value theory directly for the **L∞-norm of the iterated integrals (the signature)** over a sliding window.

### Theory

For a path X: [0,T] → ℝ^d, the **signature** S(X) is the infinite sequence of iterated integrals:
S(X) = (1, ∫dX, ∫∫dX⊗dX, ...)

text

Truncated at depth **m**, the signature is a finite-dimensional feature vector capturing path shape.

The engine computes the **L∞-norm of the truncated signature** over rolling windows:
||S^m(X)||_∞ = max(|S^m_1|, |S^m_2|, ..., |S^m_N|)

text

The **Peaks-Over-Threshold (POT)** method fits a **Generalized Pareto Distribution (GPD)** to the tail:
G_{ξ,σ}(x) = 1 - (1 + ξ·x/σ)^(-1/ξ)

text

Shape parameter **ξ** indicates tail heaviness:
- **ξ > 0**: Fat-tailed (Fréchet) — heavy extreme risk
- **ξ = 0**: Exponential (Gumbel) — moderate extremes
- **ξ < 0**: Bounded (Weibull) — thin tails

The engine answers:
> *"What is the 1-in-100-year event for the entire path-shape of this asset over the next 5 days, as a geometric object, not just the terminal price?"*

### Return Period (1-in-N-year) Calculation
Return Level = u + (σ/ξ) · ((N·ζ_u)^ξ - 1)

text

where:
- u = threshold (95th percentile of signature norms)
- ζ_u = probability of exceeding the threshold
- N = desired return period in years

---

## Universes

| Universe | Tickers |
|----------|---------|
| FI_COMMODITIES | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| EQUITY_SECTORS | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM, IWD, IWO, XLB, XLRE |
| COMBINED | All of the above |

---

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| SIGNATURE_DEPTH | Truncation level of path signature | 3 |
| WINDOW_DAYS | Rolling window for signature computation | 63, 252 |
| EVT_THRESHOLD | Quantile for POT threshold | 0.95 |
| RETURN_PERIOD_YEARS | Return period for tail risk (1-in-N years) | 100 |
| LOOKAHEAD_DAYS | Forward horizon for path shape | 5 |

---

## Repository Structure
P2-ROUGH-EVT/
├── README.md
├── config.py
├── data_manager.py
├── rough_evt.py
├── trainer.py
├── push_results.py
├── streamlit_app.py
├── us_calendar.py
├── requirements.txt
└── .github/
└── workflows/
└── daily.yml

text

---

## Data Flow
HuggingFace Master Dataset
P2SAMAPA/fi-etf-macro-signal-master-data
│
▼
data_manager.py
(prices, macro DFs)
│
▼
rough_evt.py
sliding window → path signature →
L∞-norm → GPD fit → tail risk →
return level (1-in-N years)
│
▼
trainer.py
builds two JSON files:
• rough_evt_YYYY-MM-DD.json
• rough_evt_windows_YYYY-MM-DD.json
│
▼
push_results.py
HfApi.upload_file →
P2SAMAPA/p2-rough-evt-results
│
▼
streamlit_app.py
Tab 1: Tail Risk Summary per ETF
Tab 2: Signature Distribution Explorer

text

---

## Output JSON Schemas

### Tab 1 — `rough_evt_YYYY-MM-DD.json`

```json
{
  "run_date": "2026-07-30",
  "universes": {
    "FI_COMMODITIES": {
      "top_risky": [
        {"ticker": "HYG", "return_level": 0.87, "tail_index": 0.42}
      ],
      "full_risk": {
        "TLT": {
          "return_level_100yr": 0.52,
          "tail_index": 0.18,
          "threshold": 0.34,
          "exceedances": 47,
          "best_window": 252,
          "lookahead": 5
        }
      }
    }
  }
}
Tab 2 — rough_evt_windows_YYYY-MM-DD.json
json
{
  "run_date": "2026-07-30",
  "universes": {
    "FI_COMMODITIES": {
      "windows": {
        "63": {
          "top_risky": [{"ticker": "HYG", "return_level": 0.89}],
          "full_ranking": [
            ["HYG", 0.89],
            ["TLT", 0.52]
          ]
        },
        "252": {
          "top_risky": [{"ticker": "GLD", "return_level": 0.71}],
          "full_ranking": [
            ["GLD", 0.71],
            ["SLV", 0.65]
          ]
        }
      }
    }
  }
}
Setup & Local Run
bash
git clone https://github.com/P2SAMAPA/P2-ROUGH-EVT
cd P2-ROUGH-EVT
pip install -r requirements.txt

export HF_TOKEN=hf_...
python trainer.py
Streamlit Dashboard
bash
streamlit run streamlit_app.py
GitHub Actions
Runs automatically at 00:30 UTC Monday–Saturday via .github/workflows/daily.yml.

Required secret: HF_TOKEN (set in repo Settings → Secrets → Actions).

Key Implementation Notes
No dropna() on all columns — only dropna(subset=MACRO_COLS_CORE) to avoid losing history.

Signature computation uses the signatory library (PyTorch backend) for fast iterated integrals.

GPD fitting uses scipy.stats.genpareto.fit with MLE.

Cross-sectional ranking applied per universe per window so tail risks are comparable.

HfApi.upload_file used for all HuggingFace writes (not HfFileSystem.open).

References
Lyons, T. (1998). Differential equations driven by rough signals. Revista Matemática Iberoamericana.

Chevyrev, I., & Kormilitzin, A. (2016). A primer on the signature method in machine learning. arXiv:1603.03788.

Coles, S. (2001). An Introduction to Statistical Modeling of Extreme Values. Springer.

Pickands, J. (1975). Statistical inference using extreme order statistics. Annals of Statistics.

Pikovsky, A., Rosenblum, M., Kurths, J. (2001). Synchronization: A Universal Concept in Nonlinear Sciences. Cambridge University Press.
