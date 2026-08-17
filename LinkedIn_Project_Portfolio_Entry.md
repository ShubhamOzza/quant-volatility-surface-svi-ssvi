# LinkedIn Project Portfolio Entry

## Title

Arbitrage-Free Volatility Surface Modeling with SVI and SSVI: Full Calibration Pipeline for Index Options

## Description

I built an end-to-end quantitative finance pipeline that turns raw index
option quotes into a smooth, arbitrage-free implied volatility surface, the
same object that trading desks and risk teams rely on every day.

What the project does:

- Cleans and validates real option chain data with auditable liquidity,
  maturity, and moneyness filters (bid > 0, open interest or volume > 0,
  7 < DTE < 365, forward moneyness 0.8 to 1.2).
- Recovers the expiry forward from put-call parity by regression, achieving a
  median parity residual of 0.005 basis points of the forward.
- Inverts option prices into Black-76 implied volatilities with Brent's
  method and a bisection fallback, then works in total variance and
  log-forward moneyness coordinates.
- Calibrates the raw SVI model per expiry with bound-constrained least
  squares, spread-based loss weights, and a multi-start strategy that defends
  against the local minima this calibration is known for.
- Builds a joint SSVI surface with a non-decreasing at-the-money total
  variance term structure, removing calendar spread arbitrage by construction.
- Verifies butterfly arbitrage freedom numerically with the Gatheral-Jacquier
  g(k) condition on an 801-point grid, and enforces the same condition (plus
  the Gatheral wing bound) as an in-fit penalty constraint on every
  individual raw SVI slice, not just the joint surface, so no slice is
  reported as fitted unless it is also arbitrage-free.
- Compares a calm (2023-2024 style, ~14% ATM vol) and a stressed (2022 style,
  ~30% ATM vol, steeper skew, wider spreads) synthetic regime head to head,
  confirming fit quality and zero arbitrage violations hold in both.

Validation highlights on the synthetic benchmark: median slice RMSE of 0.085
volatility points, zero calendar arbitrage violations, zero butterfly
violations on the joint surface and on every individual fitted slice, and
stable results across random seeds and across both market regimes. The
pipeline ships with a clearly labeled synthetic demo mode so anyone can
reproduce the full run in seconds, and the identical code path accepts real
SPX or NIFTY option chain CSVs covering 2022 to 2025.

What I found most interesting: fitting each expiry independently looks fine
until you check for cross-expiry arbitrage. Early in development, one of my
ten demo slices quietly admitted butterfly arbitrage at the parameter
boundary because only box bounds were enforced during the fit. Rather than
just flagging that after the fact, I added the Gatheral wing bound and the
butterfly condition as penalty terms inside the slice-level optimizer
itself, closing the gap at the source. Watching the diagnostic go from
catching a real violation to confirming zero violations, on every slice and
in both a calm and a stressed synthetic market, was the most instructive
part of the project.

## Skills (comma-separated)

Quantitative Finance, Options Pricing, Volatility Surface Modeling, SVI, SSVI, Arbitrage-Free Calibration, Black-76 Model, Put-Call Parity, Implied Volatility, Nonlinear Least Squares, Constrained Optimization, Python, NumPy, SciPy, Pandas, Matplotlib, Derivatives, Risk Management, Numerical Methods, Data Cleaning, Model Validation, Reproducible Research

## Media Recommendations

1. **Hero image**: the 3-D SSVI volatility surface figure
   (`outputs/fig2_ssvi_surface.png`) with market quotes overlaid in red. It is
   the most visually striking output and immediately communicates the project.
2. **Arbitrage diagnostics panel** (`outputs/fig3_arbitrage_diagnostics.png`):
   the monotone total variance curve and the g(k) butterfly condition plot
   tell the arbitrage-free story in one image and signal genuine technical
   depth to reviewers.
3. **Slice fit grid** (`outputs/fig1_svi_slice_fits.png`): ten small panels of
   market quotes versus fitted smiles shows calibration quality at a glance.
4. **Regime comparison** (`outputs/fig5_regime_comparison.png`): the calm vs
   stressed ATM term structure, smile, RMSE, and cost panels in one image,
   showing the pipeline was validated across market conditions, not tuned to
   a single favorable case.
5. **Short screen recording or GIF** of `python Arbitrage_Free_Volatility_Surface_SVI_SSVI.py --demo`
   running end to end, ending on the console validation report. Demonstrates
   reproducibility better than any description.
6. **Link the README and the validation table** from the repository in the
   post text so technical readers can verify the correctness targets.
