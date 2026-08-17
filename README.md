# Project I-1: Arbitrage-Free Volatility Surface Modeling with SVI and SSVI

A complete, reproducible pipeline that converts raw index option quotes
(SPX or NIFTY style chains) into a smooth, arbitrage-free implied volatility
surface using the raw SVI parametrization per expiry and the SSVI
parametrization across the surface, with explicit calendar and butterfly
arbitrage diagnostics.

## Quick start

```bash
pip install -r requirements.txt

# Synthetic demonstration mode (no external data needed, clearly labeled):
python Arbitrage_Free_Volatility_Surface_SVI_SSVI.py --demo

# Real data mode:
python Arbitrage_Free_Volatility_Surface_SVI_SSVI.py --input your_chain.csv

# Calm vs stressed regime comparison (writes outputs/calm/, outputs/stressed/,
# and a top-level regime_comparison_summary.csv + fig5_regime_comparison.png):
python Arbitrage_Free_Volatility_Surface_SVI_SSVI.py --demo --regime both

# Useful options:
python Arbitrage_Free_Volatility_Surface_SVI_SSVI.py --demo \
    --output-dir outputs --rate 0.03 --seed 42 --regime calm
```

A full synthetic run takes a few seconds and writes all figures and CSV
summaries into the output directory. If no `--input` is given the script
defaults to demo mode and says so in the log. `--regime` selects the
synthetic scenario: `calm` (default, 2023-2024 style), `stressed` (2022
bear-market style), or `both` for a full head-to-head comparison.

## File map

| File | Purpose |
|------|---------|
| `Arbitrage_Free_Volatility_Surface_SVI_SSVI.py` | The complete pipeline: validation, cleaning, parity forwards, Black-76 inversion, SVI and SSVI calibration, arbitrage diagnostics, plots, summaries. |
| `Word_Document_Content.md` | Academic report content: abstract, methodology, application, interpretation, validation, limitations, conclusion. |
| `LinkedIn_Project_Portfolio_Entry.md` | Portfolio write-up with skills and media recommendations. |
| `Plain_English_Project_Notes.md` | Non-technical explanation and interview preparation notes. |
| `README.md` | This file. |
| `requirements.txt` | Python dependencies. |
| `outputs/` (created on run) | All generated artifacts, see below. |

### Generated outputs (demo run)

| File | Content |
|------|---------|
| `outputs/synthetic_option_chain.csv` | The generated raw chain (demo mode only). |
| `outputs/iv_panel.csv` | Cleaned panel of implied vols, k, w, weights. |
| `outputs/forwards.csv` | Parity-implied forwards and parity residuals per expiry. |
| `outputs/svi_slice_summary.csv` | Raw SVI parameters and fit quality per expiry. |
| `outputs/surface_summary.csv` | SSVI parameters and all headline validation metrics. |
| `outputs/butterfly_diagnostics_svi.csv` | Per-slice butterfly diagnostics for raw SVI. |
| `outputs/fig1_svi_slice_fits.png` | Market IVs vs raw SVI fit per expiry. |
| `outputs/fig2_ssvi_surface.png` | 3-D SSVI surface with quotes overlaid. |
| `outputs/fig3_arbitrage_diagnostics.png` | Theta term structure and g(k) butterfly checks. |
| `outputs/fig4_residual_diagnostics.png` | Residual heatmap (expiry x moneyness) plus a per-quote residual scatter, for under/overfitting inspection. |

### Generated outputs (`--regime both` run)

Running with `--regime both` writes a complete, independent copy of the
outputs above into `outputs/calm/` and `outputs/stressed/`, plus two
top-level comparison artifacts:

| File | Content |
|------|---------|
| `outputs/regime_comparison_summary.csv` | Side-by-side calm vs stressed metrics: ATM vol level, SSVI skew and shape parameters, fit RMSE, arbitrage violation counts, parity residual, and an ATM relative bid-ask spread cost proxy. |
| `outputs/fig5_regime_comparison.png` | Four-panel figure: ATM term structure, shortest-expiry smile, median slice RMSE, and quoted cost, calm vs stressed. |

## Input data schema (real mode)

One row per option quote. Column names are matched case-insensitively and
common aliases are accepted (e.g. `strike_price`, `cp_flag`, `oi`).

| Column | Required | Description |
|--------|----------|-------------|
| `expiry` | yes | Calendar date (YYYY-MM-DD) or numeric days to expiry. |
| `strike` | yes | Option strike, positive number. |
| `option_type` | yes | `c`/`p`, also accepts `call`/`put`, `CE`/`PE`. |
| `bid` | yes | Bid price; rows with bid <= 0 are removed. |
| `ask` | yes | Ask price; must satisfy ask >= bid. |
| `open_interest` | recommended | Rows are kept if OI > 0 or volume > 0. |
| `volume` | recommended | See above. |
| `snapshot_date` | optional | Quote date; required to convert calendar expiries to DTE. If absent and expiries are dates, today is assumed and a warning is logged. |

Filters applied, per the project specification: bid > 0, open interest or
volume > 0, 7 < DTE < 365, forward moneyness K/F in [0.8, 1.2], plus
out-of-the-money selection and a minimum time-value guard for stable
volatility inversion.

## Methodology in one paragraph

Forwards come from a put-call parity regression per expiry. Mid prices are
inverted with Black-76 (Brent's method, bisection fallback). Quotes are
mapped to total variance w = sigma_iv^2 T and log-forward moneyness
k = log(K/F). Each expiry is fitted with raw SVI under the constraints
a >= 0, b >= 0, |rho| < 1, sigma > 0, using spread-based weights and an
eight-start strategy against local minima. Because scipy's `least_squares`
only supports box bounds, two further slice-level no-arbitrage conditions,
the Gatheral wing bound b(1+|rho|) <= 4/T and the butterfly condition
g(k) >= 0, are enforced as large penalty residuals added to the objective on
a 121-point coarse grid, so a candidate slice fit that violates them is
pushed back into the feasible region during the fit itself, not just
flagged afterward. The SSVI surface is then built by extracting the
at-the-money total variance per expiry, projecting it onto the
non-decreasing cone (removing calendar arbitrage by construction), and
jointly fitting the global shape parameters. Butterfly freedom is verified
numerically with the Gatheral-Jacquier g(k) >= 0 condition on an 801-point
grid, both for the SSVI surface and, independently, for each raw SVI slice.

## Validation targets and demo results (synthetic data, calm regime)

| Target | Demo result |
|--------|-------------|
| Parity residual < 1 bp on liquid expiries | 0.005 bp median |
| Zero butterfly violations on fine grid (SSVI) | 0 |
| Zero butterfly violations on every fitted raw SVI slice | 0 (10/10 slices) |
| Total variance monotone in maturity | yes, 0 violations |
| Median slice RMSE < 0.5 vol points | 0.085 vol points (range 0.060-0.125) |

Results were stable across seeds 7, 42, and 123 (median slice RMSE between
0.078 and 0.096 vol points, zero arbitrage violations in all runs, at both
the SSVI surface level and every individual raw SVI slice).

## Calm vs stressed regime comparison

`--regime both` runs the identical pipeline on two stylized synthetic
scenarios: `calm` (2023-2024 style: ATM vol around 14%, mild upward-sloping
term structure, moderate skew) and `stressed` (2022 bear-market style: ATM
vol around 30%, backwardated/inverted annualized-vol term structure,
steeper negative skew, and roughly double the quoted bid-ask cost). Both
regimes still satisfy the SSVI calendar precondition (total variance
non-decreasing in maturity) because variance accumulates with time even when
annualized vol falls, which is also true of real inverted term structures.
Headline comparison (see `outputs/regime_comparison_summary.csv`):

| Metric | Calm | Stressed |
|--------|------|----------|
| Average ATM implied vol | 13.8% | 30.0% |
| SSVI skew (rho) | -0.361 | -0.839 |
| Median slice RMSE (vol points) | 0.085 | 0.087 |
| Calendar / butterfly violations (SSVI, all slices) | 0 / 0 | 0 / 0 |
| Parity residual (median, bp) | 0.0047 | 0.0051 |
| Average ATM relative bid-ask spread | 41 bp | 82 bp |

Fit quality and arbitrage-freedom hold up equally well in both regimes; the
main differences are the level of volatility, the steepness of the skew, and
the cost of trading, exactly as expected between a calm and a stressed
market. See `outputs/fig5_regime_comparison.png`.

## Notes and honest caveats

- All figures and CSVs produced by `--demo` are labeled "SYNTHETIC DEMO
  DATA". They demonstrate the pipeline, not real market behavior. The calm
  and stressed regimes are stylized scenarios, not a fit to any specific
  historical date.
- The synthetic ground truth is itself SSVI, so demo fit quality is
  favorable; expect larger errors on real chains, especially in illiquid
  long-dated wings.
- Independently fitted raw SVI slices can in principle admit butterfly
  arbitrage even when each fits its own quotes well; earlier versions of
  this pipeline (before the in-fit penalty constraints described above)
  exhibited exactly that failure mode on one demo slice. The fix is to
  enforce the wing bound and g(k) >= 0 during the fit itself, which is what
  the current pipeline does; the diagnostic still checks every slice
  independently afterward and reports violations honestly rather than
  suppressing them, so a regression would be caught immediately.
- The target universe is 3 years of daily SPX or NIFTY chains, 2022 to 2025.
  Run the script once per snapshot date to build a surface time series;
  the code is deterministic given the same input file.
- Tested with Python 3.13.5, numpy 2.4.2, scipy 1.17.0, pandas 2.2.3,
  matplotlib 3.10.8 (see `requirements.txt` for pinned versions).
