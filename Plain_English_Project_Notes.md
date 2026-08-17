# Plain English Project Notes: Volatility Surface Modeling with SVI and SSVI

## What we made

We built a Python program that takes a list of option price quotes (buyers'
bids and sellers' asks for calls and puts at many strikes and expiry dates)
and turns it into one smooth, consistent map of implied volatility for every
strike and every expiry. That map is called the volatility surface. The
program guarantees that the surface contains no free-money opportunities
(arbitrage), and it proves this with numerical checks rather than just
assuming it.

Because real historical option data costs money and cannot be shared, the
program can generate its own realistic fake data set, clearly labeled as
synthetic, so anyone can run the whole thing in a few seconds. The exact same
code also accepts real data files.

## The core concept

Every option price implies a volatility: the level of market turbulence that
would justify that price under a standard pricing model. If you plot that
implied volatility against the option's strike, you get a curve called the
smile or skew. If you stack these curves for all expiry dates, you get the
volatility surface.

Two problems arise in practice. First, raw quotes are noisy and discrete, so
we need a mathematical formula that draws a smooth curve through them. That
formula is SVI (Stochastic Volatility Inspired), a five-parameter curve that
has become the industry standard because it is flexible yet well behaved.
Second, fitting each expiry separately can produce curves that contradict
each other: for example, the fitted surface might imply that an option with
more time to expiry is worth less than an otherwise identical shorter one,
which is a free-money inconsistency called calendar arbitrage, or that a
butterfly combination of options has negative value, called butterfly
arbitrage. The SSVI (Surface SVI) formula ties all expiries together with a
shared structure that makes these contradictions impossible, provided one
key quantity, the total variance at the money, never decreases with maturity.
Our program enforces exactly that.

## How we made it, step by step

1. **Clean the quotes.** Throw away quotes with zero bids, zero trading
   interest, expiries under a week or over a year, and strikes too far from
   the current price. Also keep only out-of-the-money options for the fitting
   step, because deep in-the-money option prices are mostly intrinsic value
   and tell you almost nothing about volatility.
2. **Find the forward price.** For each expiry, put-call parity (a
   no-arbitrage identity linking call prices, put prices, and the forward)
   lets us recover the market-implied forward by a simple regression of
   call-minus-put prices on strike. We check the fit is tight, under one
   basis point of error.
3. **Convert prices to volatilities.** Each option's mid price is run
   backwards through the Black-76 pricing model to find the implied
   volatility, using a robust root-finding algorithm (Brent's method, with
   bisection as backup). We then switch to the two coordinates the SVI model
   likes: total variance (volatility squared times time) and log-forward
   moneyness (how far the strike is from the forward, in log terms).
4. **Fit SVI to each expiry.** A nonlinear least squares solver finds the
   best five parameters per expiry. We give more weight to quotes with tight
   bid-ask spreads (they are more trustworthy), enforce the mathematical
   bounds that keep the curve sensible, and run the solver from several
   different starting points because this particular fitting problem has
   local minima that can trap a naive single run.
5. **Fit SSVI to the whole surface.** We read off the at-the-money total
   variance for each expiry, force that sequence to be non-decreasing (this
   is what kills calendar arbitrage), then fit three global shape parameters
   across all quotes at once.
6. **Check for arbitrage explicitly.** On a grid hundreds of times finer
   than the strike spacing, we verify two things: total variance never falls
   as maturity increases (no calendar arbitrage), and a quantity called g(k)
   stays non-negative everywhere (no butterfly arbitrage, using the test
   published by Gatheral and Jacquier).
7. **Report everything.** The program writes summary spreadsheets, a console
   validation report with explicit pass/fail style targets, and four charts:
   per-expiry fits, the 3-D surface, the arbitrage diagnostics, and a
   residual plot used to spot underfitting or overfitting.

## Results on the demo data

On the synthetic benchmark, the parity regression is accurate to 0.005 basis
points, the typical per-expiry fit error is 0.085 volatility points (well
under the 0.5 target), and both arbitrage checks pass with zero violations,
on every individual expiry as well as on the joint surface. Fitting a single
expiry independently is a real, well-known way to accidentally admit
butterfly arbitrage (parameters can drift to an extreme corner that fits the
observed quotes but implies a locally negative butterfly price), so the
program directly penalizes that outcome inside the fit itself, not just
after the fact, and then double-checks every slice on a finer independent
grid to make sure the penalty actually worked.

We also ran the program on two different synthetic markets: a "calm" one
(low volatility, around 14%, like 2023-2024) and a "stressed" one (high
volatility, around 30%, with a steeper downward skew and wider bid-ask
spreads, like the 2022 bear market). Both pass every check with zero
arbitrage violations, and the fit quality barely changes between them, which
is exactly what you would want: the pipeline should work the same way
whether the market is calm or stressed, only the numbers it recovers should
differ.

## Interview-ready explanation

If asked to explain this project in an interview, I would say:

"I built a pipeline that converts raw option quotes into an arbitrage-free
implied volatility surface using SVI and SSVI, which is the industry-standard
parametrization. The pipeline starts with quote hygiene: liquidity filters,
out-of-the-money selection, and parity-implied forwards, because garbage in
means garbage out. I invert prices with Black-76 using Brent's method, then
calibrate raw SVI per expiry with bounded least squares, spread-based
weights, and a multi-start because the SVI objective has local minima. The
key insight is that slice-by-slice fits can be individually excellent yet
mutually arbitrageable, so I re-fit the surface jointly in SSVI form with a
non-decreasing at-the-money total variance term structure, which removes
calendar arbitrage by construction, and I verify butterfly freedom numerically
with the Gatheral-Jacquier g(k) condition on a fine grid. I also enforce the
same butterfly condition and a wing bound directly inside each individual
slice fit as penalty terms, because without that penalty one slice would
happily fit the quotes while quietly admitting arbitrage at the parameter
boundary. On my synthetic benchmark the median slice error was under a tenth
of a volatility point with zero arbitrage violations on every slice and on
the joint surface, and I confirmed the same holds in a deliberately stressed
synthetic regime as well as a calm one, not just one favorable scenario."

Possible follow-up questions and short answers:

- **Why out-of-the-money options only?** Deep in-the-money prices are almost
  all intrinsic value, so a one-tick price error becomes a huge volatility
  error. OTM quotes are where the volatility information lives.
- **Why total variance instead of volatility?** Total variance w = sigma
  squared times T behaves linearly in time, which is what makes the calendar
  no-arbitrage condition simple: w must not decrease in T.
- **What can go wrong in the SVI fit?** Local minima and flat directions:
  different parameter combinations can produce nearly the same curve inside
  the observed strikes, so I use multi-start, bounds, and I report when
  parameters hit their limits.
- **Why SSVI instead of interpolating between SVI slices?** Interpolation
  does not guarantee no-arbitrage. SSVI with monotone theta has a published
  theorem behind it, plus I verify numerically on a fine grid.
- **How would you extend it?** eSSVI with time-varying rho for richer skew
  term structure, daily runs over 2022 to 2025 for a surface time series,
  and staleness detection for real feeds.
- **How did you validate this across market conditions, not just one lucky
  day?** I built two stylized synthetic markets, calm and stressed, with
  very different volatility levels and skew steepness, and ran the identical
  pipeline on both. Fit quality and the zero-arbitrage result held in both,
  which is the point: the method should not be tuned to one regime.
