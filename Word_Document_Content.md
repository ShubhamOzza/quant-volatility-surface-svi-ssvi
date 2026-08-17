# Constructing Arbitrage-Free Implied Volatility Surfaces with SVI and SSVI: A Calibration Study on Index Option Chains

## Abstract

This project develops and validates a complete pipeline for constructing an
arbitrage-free implied volatility surface from index option quotes, using the
Stochastic Volatility Inspired (SVI) parametrization of Gatheral (2004) for
individual expiry slices and the Surface SVI (SSVI) parametrization of Gatheral
and Jacquier (2014) for the joint surface. Starting from raw bid and ask
quotes, the pipeline enforces liquidity and maturity filters, recovers the
expiry forward from put-call parity by regression, inverts option prices into
Black-76 implied volatilities using Brent's method with a bisection fallback,
and expresses the smile in total implied variance and log-forward moneyness
coordinates. Each expiry is calibrated with bound-constrained nonlinear least
squares, using spread-based loss weights and a multi-start strategy to guard
against local minima. The surface is then re-parameterized in SSVI form with a
non-decreasing at-the-money total variance term structure, which eliminates
calendar spread arbitrage by construction, and butterfly arbitrage is tested on
a fine grid through the Gatheral-Jacquier g(k) condition. On a controlled
synthetic benchmark, the pipeline attains a put-call parity residual of
0.005 basis points of the forward, a median slice root-mean-square error of
0.085 volatility points, zero calendar arbitrage violations, and zero
butterfly violations across the surface grid and every individually fitted
raw SVI slice. The same pipeline is also run on two stylized synthetic
regimes, a calm 2023-2024-style market and a stressed 2022-bear-market-style
market, to demonstrate that fit quality and arbitrage freedom hold up under
both a low-volatility and a high-volatility, steep-skew environment. All
reported figures are produced from synthetic demonstration data and are
clearly labeled as such; the identical code path accepts real SPX or NIFTY
option chain files without modification.

## 1. Introduction

The implied volatility surface is the central object in derivatives pricing,
risk management, and relative value trading. Market makers, quantitative
researchers, and risk teams all require a smooth, arbitrage-free mapping from
strike and maturity to implied volatility, because any arbitrage embedded in
the surface propagates directly into exotic option prices, hedge ratios, and
value-at-risk estimates. Raw market quotes, however, are discrete, noisy, and
subject to bid-ask bounce, so a parametrization step is unavoidable.

This project implements the two parametrizations that have become the
practical industry standard: raw SVI for single expiries and SSVI for the full
surface. The emphasis is not only on fitting quality but on financial
correctness. A surface that interpolates quotes well but admits butterfly or
calendar arbitrage is unusable in production. The pipeline therefore treats
arbitrage diagnostics as first-class outputs rather than afterthoughts, and
every stage includes explicit sanity checks: forwards must be consistent with
put-call parity, implied volatilities must be positive, total variance must be
non-negative and non-decreasing in maturity, and the butterfly condition must
hold on a grid far finer than the observed strike spacing.

The target data set is three years of daily SPX or NIFTY option chains from
2022 to 2025. Because licensed historical option data is not redistributable,
the delivered project runs out of the box on a clearly labeled synthetic
demonstration data set whose ground truth is a known SSVI surface, while the
real-data workflow is fully implemented and documented for the user's own
data files.

## 2. Methodology

### 2.1 Data cleaning and validation

Raw option chain files are validated against a documented schema (expiry,
strike, option type, bid, ask, open interest, volume). Column aliases from
common vendor formats are harmonized automatically. The cleaning stage applies
the following filters, each logged with a row-level accounting so the process
is fully auditable:

- Bid strictly greater than zero.
- Open interest or traded volume strictly greater than zero.
- Days to expiry strictly between 7 and 365.
- Ask at least as large as bid (no locked or crossed markets).
- Forward moneyness K/F between 0.8 and 1.2, applied after forward estimation.

Two further professional filters are applied during panel construction. First,
only out-of-the-money quotes are retained for calibration (calls above the
forward, puts below it), because deep in-the-money option prices are dominated
by intrinsic value and a single tick of price noise translates into a large
implied volatility error. Second, quotes whose price is within one tick of
intrinsic value are discarded as numerically unstable for volatility inversion.

### 2.2 Forward estimation from put-call parity

For each expiry, European put-call parity on the forward states

C(K) - P(K) = exp(-rT) (F - K).

Regressing the call-put mid-price difference on strike across all matched
strikes yields the discount factor as minus the slope and the forward as the
ratio of intercept to minus slope. This regression approach is robust to the
bid-ask bounce of individual strikes because errors average out across the
strike range. The regression residual, expressed in basis points of the
estimated forward, is retained as a data-quality diagnostic; the correctness
target is below one basis point on liquid expiries.

### 2.3 Implied volatility inversion

Option mids are inverted to Black-76 implied volatilities. The Black-76 model
on the parity-implied forward is preferred to Black-Scholes on the spot
because it embeds the market-implied cost of carry directly and requires no
dividend assumption. Inversion uses Brent's method on a wide bracket
(0.01 percent to 500 percent annualized) with a pure bisection fallback, and
prices outside the model-free no-arbitrage band are rejected rather than
forced. The smile is then expressed in total implied variance
w = sigma_iv squared times T and log-forward moneyness k = log(K/F), the
natural coordinates for the SVI family.

### 2.4 Raw SVI calibration per expiry

Each expiry slice is fitted with the raw SVI form

w(k) = a + b ( rho (k - m) + sqrt((k - m)^2 + sigma^2) ),

with parameter constraints a >= 0, b >= 0, |rho| < 1, sigma > 0 enforced
through the box bounds of scipy's trust-region reflective least squares
solver. The level constraint a >= 0 is a slice-level no-arbitrage ingredient,
since total variance can never be negative. The loss is a weighted sum of
squared total-variance residuals, with weights inversely proportional to the
squared relative bid-ask spread of each quote, normalized to mean one per
slice. Quotes with tight, liquid markets therefore anchor the fit, while wide
wing quotes influence it only weakly.

Raw SVI calibration is well known to suffer from local minima and parameter
identifiability issues: combinations of large sigma, large |rho|, and offset
m can describe nearly identical smiles inside the observed strike range. The
pipeline addresses this in three ways. First, a multi-start strategy runs the
optimizer from a data-driven anchor plus randomized perturbations and retains
the best weighted solution. Second, the summary table reports which start won
and whether any parameter sits on a bound, making ill-identified slices
visible rather than hidden. Third, residual diagnostics (fitted minus market
volatility against moneyness) are produced for every slice so systematic
misfit, the signature of underfitting, can be detected visually and
numerically.

A well-identified slice fit is not automatically arbitrage-free: two further
conditions from Gatheral (2004) and Gatheral and Jacquier (2014), the wing
bound b(1+|rho|) <= 4/T and the butterfly condition g(k) >= 0, must also
hold. Because scipy's bound-constrained least-squares solver only supports
box constraints on individual parameters, not joint nonlinear conditions
across all five, both are enforced as large penalty residuals evaluated on a
121-point coarse grid and appended to the objective. A candidate parameter
vector that violates either condition is therefore penalized heavily inside
the fit itself, which pushes the optimizer back into the feasible region
rather than only flagging the violation after the fact. This closes a gap
present in an earlier version of the pipeline, in which one slice fitted
without these penalties admitted a genuine butterfly violation (222 of 801
grid points, minimum g(k) = -1.87); with the penalties in place, every
fitted slice passes the same fine-grid diagnostic with zero violations.

### 2.5 SSVI surface construction

Independently fitted SVI slices are almost never mutually consistent: even
when each slice is butterfly free, nothing prevents total variance from
decreasing in maturity, which is calendar spread arbitrage. The SSVI
parametrization solves this by writing

w(k; theta_t) = theta_t / 2 ( 1 + rho phi(theta_t) k
                              + sqrt((phi(theta_t) k + rho)^2 + 1 - rho^2) ),

where theta_t is the at-the-money total variance at maturity t and phi is a
power-law function phi(theta) = eta / theta^gamma with gamma in (0, 1/2].
By the Gatheral-Jacquier no-arbitrage theorem, a constant-rho SSVI surface
with non-decreasing theta_t and this phi family is free of calendar spread
arbitrage, and is butterfly free whenever theta phi(theta) (1 + |rho|) <= 4,
a condition checked numerically on a fine grid.

The calibration proceeds in two stages. Stage one recovers theta_t per expiry
directly from the raw SVI slices as w_svi(0), a nearly model-free anchor, and
projects the sequence onto the non-decreasing cone with a cumulative maximum,
absorbing violations smaller than fit noise. Stage two holds the monotone
theta sequence fixed and jointly fits the three global shape parameters
(rho, eta, gamma) against the full total-variance panel. Fixing theta first
turns the joint problem into a small, well-conditioned three-parameter fit
and guarantees the calendar condition by construction.

### 2.6 Arbitrage diagnostics

Calendar arbitrage is checked by verifying that model total variance is
non-decreasing in maturity for every point of a fine moneyness grid across
every consecutive maturity pair. Butterfly arbitrage is checked with the
Gatheral-Jacquier condition

g(k) = (1 - k w'(k) / (2 w(k)))^2
       - w'(k)^2 / 4 (1 / w(k) + 1/4) + w''(k) / 2 >= 0,

evaluated with the closed-form SSVI derivatives on an 801-point grid spanning
k from -0.5 to 0.5, far beyond the calibration range. The same diagnostic is
run on each raw SVI slice, since raw SVI is not guaranteed to be butterfly
free even when the in-fit penalty constraints described in Section 2.4 are
active; the diagnostic is kept as an independent, finer-grid check rather
than trusting the fit's own coarse penalty grid, and any violations are
reported honestly rather than suppressed.

## 3. Step-by-Step Application

Running the pipeline on the synthetic demonstration data proceeds as follows:

1. A synthetic option chain is generated from a known SSVI ground truth with
   a gently upward-sloping at-the-money term structure, moderate negative
   skew, realistic bid-ask spreads that widen in the wings, and a small
   number of deliberately invalid rows. The ground truth is labeled synthetic
   throughout.
2. Cleaning removes zero-bid rows, rows without open interest or volume, and
   maturities outside the 7 to 365 day window; the log records 385 raw rows
   reduced to 373 clean rows.
3. Put-call parity regressions recover one forward per expiry with a median
   residual of 0.005 basis points, comfortably below the one basis point
   target.
4. Out-of-the-money selection and Black-76 inversion produce a panel of 138
   implied volatilities across 10 expiries, converted to (k, w) coordinates.
5. Raw SVI calibration with eight multi-starts per slice converges for all
   expiries; per-slice root-mean-square errors range from 0.060 to 0.125
   volatility points, with a median of 0.085.
6. SSVI surface construction yields a monotone theta term structure and
   global parameters rho = -0.361, eta = 0.812, gamma near zero, recovering
   the negative skew of the ground truth with a surface root-mean-square
   error of 0.105 volatility points.
7. Arbitrage diagnostics report zero calendar violations and zero butterfly
   violations across the full surface grid, with a worst g(k) value of 0.834,
   comfortably positive. Every individual raw SVI slice also passes the
   801-point butterfly diagnostic with zero violations.
8. Four figures and four CSV summaries are written to the output folder.

For real data, the user replaces step 1 with a pointer to their own CSV file;
every subsequent step is identical. Running the script once per trading day
across the 2022 to 2025 window produces a daily time series of SVI
parameters, SSVI surfaces, and arbitrage diagnostics suitable for research on
skew dynamics, term structure behavior, and surface arbitrage monitoring.

### 3.1 Calm vs stressed regime comparison

Running the pipeline with `--regime both` repeats the entire workflow above
on two stylized synthetic scenarios that stand in for the two market
environments spanned by the 2022-2025 target window: a calm regime (2023-2024
style, ATM implied vol around 14 percent, mild upward-sloping term structure,
moderate skew) and a stressed regime (2022 bear-market style, ATM implied vol
around 30 percent, a backwardated annualized-vol term structure, steeper
negative skew, and roughly double the quoted bid-ask cost). Total variance
remains non-decreasing in maturity in both regimes, since variance
accumulates with time even where annualized vol itself falls, which mirrors
how real inverted term structures behave and preserves the SSVI calendar
precondition.

| Metric                                     | Calm    | Stressed |
|---------------------------------------------|---------|----------|
| Average ATM implied vol                     | 13.8%   | 30.0%    |
| SSVI skew (rho)                             | -0.361  | -0.839   |
| Median slice RMSE (vol points)              | 0.085   | 0.087    |
| Calendar violations (SSVI)                  | 0       | 0        |
| Butterfly violations (SSVI and all slices)  | 0       | 0        |
| Parity residual, median (bp)                | 0.0047  | 0.0051   |
| Average ATM relative bid-ask spread         | 41 bp   | 82 bp    |

Fit quality and arbitrage freedom are essentially unchanged between the two
regimes; what changes is exactly what should change between a calm and a
stressed market: the level of volatility, the steepness of the negative
skew, and the cost of trading. The full breakdown is written to
`outputs/regime_comparison_summary.csv` and visualized in
`outputs/fig5_regime_comparison.png`.

## 4. Output Interpretation

The pipeline writes the following outputs:

- **svi_slice_summary.csv**: one row per expiry with the five SVI parameters,
  fit RMSE in volatility points, quote count, multi-start accounting, and the
  minimum butterfly g(k) for the slice. Parameters sitting on their bounds
  flag slices where the smile shape is not fully identified by the observed
  strike range.
- **surface_summary.csv**: the SSVI parameters, surface-level RMSE, the
  calendar and butterfly violation counts, and the parity residual
  statistics. This is the single table a reviewer should read first.
- **butterfly_diagnostics_svi.csv**: per-slice butterfly diagnostics for the
  independently fitted raw SVI slices, including the worst grid point.
- **iv_panel.csv and forwards.csv**: the cleaned (k, w) panel and the
  parity-implied forward curve, for full reproducibility.
- **fig1_svi_slice_fits.png**: market implied volatilities against the raw
  SVI fit per expiry, with RMSE in each panel title.
- **fig2_ssvi_surface.png**: the three-dimensional SSVI surface with market
  quotes overlaid.
- **fig3_arbitrage_diagnostics.png**: the theta_t term structure alongside
  the g(k) curves for every slice; the red dashed zero line makes any
  butterfly violation immediately visible.
- **fig4_residual_diagnostics.png**: a two-panel residual diagnostic. The
  left panel is a true heatmap of fitted-minus-market volatility residuals
  binned by expiry and log-forward moneyness; the right panel is the
  underlying per-quote residual scatter, colored by expiry. Small,
  unstructured residuals centered on zero (a uniformly pale heatmap) indicate
  a well-specified fit; a colored band running across the heatmap would
  indicate systematic underfitting, while a fit threading individual noisy
  quotes would indicate overfitting.
- **regime_comparison_summary.csv / fig5_regime_comparison.png** (written by
  `--regime both`): the calm-vs-stressed head-to-head described in Section
  3.1.

## 5. Validation and Results

All validation targets from the project specification are met on the
synthetic benchmark:

| Correctness target                                   | Result          |
|------------------------------------------------------|-----------------|
| Parity residual below 1 bp on liquid expiries        | 0.005 bp median |
| Zero butterfly violations on the fine grid (SSVI)    | 0 violations    |
| Zero butterfly violations on every fitted raw SVI slice | 0 violations (10/10 slices) |
| Total variance monotone in maturity                  | Yes, 0 violations |
| Median slice RMSE under 0.5 vol points               | 0.085 vol points (range 0.060-0.125) |

Two additional observations are worth reporting. First, slice-level butterfly
freedom is not automatic: an earlier version of this pipeline, in which the
raw SVI fit enforced only the box bounds a >= 0, b >= 0, |rho| < 1, sigma > 0
and did not enforce the Gatheral wing bound or the g(k) >= 0 condition
directly, produced one slice (the 35-day expiry) with parameters driven to
the boundary of the admissible region and a genuine butterfly violation on
222 of 801 fine-grid points, minimum g(k) = -1.87. Adding the wing bound and
butterfly condition as penalty terms inside the fit itself, described in
Section 2.4, eliminates the violation without materially changing fit
quality (median slice RMSE is unchanged at 0.085 vol points). This is
precisely the discipline the project specification calls for: enforcing
no-arbitrage constraints rather than merely fitting well and hoping. Second,
a seed-sensitivity check (different synthetic noise draws) leaves all
headline metrics essentially unchanged, confirming that the multi-start
calibration is stable and that the reported fit quality is not an artifact of
one favorable random seed. Third, the calm-vs-stressed regime comparison
(Section 3.1) confirms that both the fit quality and the zero-violation
result hold under a substantially different, higher-volatility, steeper-skew
market regime, not just the single calm scenario used for the headline
numbers above.

## 6. Limitations

Several limitations should be kept in mind. First, the synthetic demonstration
data is generated from an SSVI ground truth, so the strong fit quality partly
reflects model consistency; real SPX and NIFTY surfaces exhibit features such
as short-maturity smile flattening, event-driven kinks around macroeconomic
releases, and discrete dividend effects that no five-parameter slice model
captures fully. Second, constant-rho SSVI cannot represent maturity-dependent
skew decay in full generality; the eSSVI extension with time-varying rho is
the natural next step. Third, the parity-based forward assumes European
exercise and a single discount rate, which is appropriate for index options
but would need adjustment for American-style single-stock options. Fourth,
fit quality on real data will be worse than on the synthetic benchmark,
particularly for illiquid long-dated wings, and the spread-based weights only
partially compensate. Finally, the isotonic projection of theta_t is
conservative: on genuinely inverted term structures (stressed markets) it
flattens the inversion rather than rejecting the surface, which is the safe
choice for pricing but should be monitored as a stress indicator.

## 7. Conclusion

This project delivers a production-grade, fully validated pipeline for
arbitrage-free volatility surface construction. It demonstrates the complete
quantitative workflow from raw quotes to a risk-ready surface: data hygiene,
market-consistent forward estimation, robust implied volatility inversion,
constrained nonlinear calibration with multi-start protection, joint surface
construction with structural no-arbitrage guarantees, and explicit arbitrage
verification on fine grids. That slice-level raw SVI fits can admit butterfly
arbitrage when only box bounds are enforced, and that adding the Gatheral
wing bound and g(k) >= 0 as in-fit penalty constraints closes that gap, is
demonstrated empirically rather than merely asserted; the calm-vs-stressed
comparison further shows the result is not specific to one market regime.
All correctness targets are met on the synthetic benchmark, and the
real-data workflow is ready for SPX or NIFTY chains covering 2022 to 2025.

## References

- Gatheral, J. (2004). A parsimonious arbitrage-free implied volatility
  parameterization with application to the valuation of volatility
  derivatives. Presentation at Global Derivatives and Risk Management, Madrid.
- Gatheral, J. and Jacquier, A. (2014). Arbitrage-free SVI volatility
  surfaces. Quantitative Finance, 14(1), 59-71.
- Black, F. (1976). The pricing of commodity contracts. Journal of Financial
  Economics, 3(1-2), 167-179.
- Gatheral, J. (2006). The Volatility Surface: A Practitioner's Guide. Wiley.
