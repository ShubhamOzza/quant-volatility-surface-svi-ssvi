#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arbitrage-Free Volatility Surface Modeling with SVI and SSVI
=============================================================

This script implements a complete, reproducible pipeline for building an
arbitrage-free implied volatility surface from option chain quotes:

    1. Input validation and quote cleaning (liquidity, maturity, moneyness
       filters as required for index option chains such as SPX or NIFTY).
    2. Forward estimation per expiry from put-call parity.
    3. Black-76 implied volatility inversion using Brent's method with a
       bisection fallback.
    4. Conversion to total implied variance w(k, T) = sigma_iv^2 * T with
       log-forward moneyness k = log(K / F).
    5. Raw SVI calibration per expiry using scipy's least_squares with
       parameter bounds, slice-level sanity constraints, spread-based loss
       weights, and a multi-start strategy to mitigate local minima.
    6. SSVI surface construction across all expiries with a non-decreasing
       at-the-money total variance term structure theta_t, which removes
       calendar spread arbitrage by construction.
    7. Butterfly (strike) arbitrage diagnostics on a fine grid using the
       Gatheral-Jacquier sufficient condition g(k) >= 0.
    8. Fit-quality summaries (RMSE in volatility points), a summary CSV,
       and publication-ready diagnostic figures.

Two input modes are supported:

    * Real mode:   --input path/to/option_chain.csv
    * Demo mode:   --demo   (generates a clearly labeled SYNTHETIC data set;
                   no real market data is used or implied)

The synthetic demonstration mode exists so the full pipeline can be tested
end to end without any external market data. All figures and CSV outputs
produced in demo mode are labeled "SYNTHETIC DEMO DATA".

Author : Quant Master Projects (Project I-1)
License: MIT
"""

# ----------------------------------------------------------------------------
# Standard library imports
# ----------------------------------------------------------------------------
import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field, replace as dataclass_replace
from typing import Dict, List, Optional, Tuple

# ----------------------------------------------------------------------------
# Third-party imports
# ----------------------------------------------------------------------------
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm

import matplotlib
matplotlib.use("Agg")  # headless backend so the script runs on any server
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Logging configuration
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("svi_ssvi")

# Large-but-finite value used for interior numerical safeguards.
_EPS = 1e-12


# ============================================================================
# Configuration
# ============================================================================
@dataclass
class PipelineConfig:
    """All tunable settings for the volatility surface pipeline.

    Attributes
    ----------
    risk_free_rate : float
        Continuously compounded annual risk-free rate used only when a real
        CSV does not provide one. Kept small and explicit so parity-implied
        forwards remain close to data-implied levels.
    dividend_yield : float
        Continuously compounded annual dividend yield. In the reference
        workflow forwards are recovered directly from put-call parity, so
        this is only a fallback for the synthetic generator.
    min_dte, max_dte : int
        Maturity filter in days-to-expiry (7 < DTE < 365 per specification).
    min_moneyness, max_moneyness : float
        Forward moneyness filter K/F in [0.8, 1.2].
    min_strikes_per_slice : int
        Minimum number of cleaned quotes required to calibrate one expiry.
    iv_lower, iv_upper : float
        Search bracket for implied volatility inversion (annualized vol).
    parity_max_residual_bp : float
        Target tolerance for the put-call parity regression residual, in
        basis points of the forward, used in the sanity report.
    ssvi_rho_bound : float
        Hard bound |rho| < 1 used in both SVI and SSVI calibrations.
    n_multistart : int
        Number of initial guesses tried per raw SVI slice fit. The best
        solution by weighted residual sum of squares is retained. This is
        the primary defense against local minima in the calibration.
    n_fine_grid : int
        Number of points in the fine k-grid used for butterfly diagnostics.
    output_dir : str
        Folder where figures and CSV summaries are written.
    random_seed : int
        Seed for the synthetic demo generator (full reproducibility).
    regime : str
        Which synthetic market regime to generate: "calm" (2023-2024 style,
        low ATM vol, upward-sloping term structure, mild skew) or "stressed"
        (2022 bear-market style, high ATM vol, inverted/backwardated
        annualized-vol term structure, steeper negative skew, wider
        spreads). Only affects ``--demo`` mode; see REGIME_PRESETS.
    """
    risk_free_rate: float = 0.03
    dividend_yield: float = 0.015
    min_dte: int = 8
    max_dte: int = 364
    min_moneyness: float = 0.80
    max_moneyness: float = 1.20
    min_strikes_per_slice: int = 6
    iv_lower: float = 1e-4
    iv_upper: float = 5.0
    parity_max_residual_bp: float = 1.0
    ssvi_rho_bound: float = 0.999
    n_multistart: int = 8
    n_fine_grid: int = 801
    output_dir: str = "outputs"
    random_seed: int = 42
    regime: str = "calm"


# ============================================================================
# Black-76 pricing and implied volatility inversion
# ============================================================================
def black76_price(F: float, K: float, T: float, r: float,
                  sigma: float, option_type: str) -> float:
    """Price a European option on a forward using the Black (1976) model.

    Parameters
    ----------
    F : float   Forward price of the underlying for maturity T.
    K : float   Strike.
    T : float   Time to expiry in years (must be positive).
    r : float   Continuously compounded discount rate.
    sigma : float  Black volatility (annualized, must be non-negative).
    option_type : 'c' or 'p'.

    Returns
    -------
    float : Present value of the option.

    Notes
    -----
    Black-76 is used (rather than Black-Scholes on the spot) because the
    forward is recovered directly from put-call parity, which is the market
    consistent approach for index options and avoids any dividend modeling
    assumptions.
    """
    if T <= 0.0:
        # At expiry the price is the intrinsic value.
        return max(F - K, 0.0) if option_type == "c" else max(K - F, 0.0)
    if sigma <= 0.0:
        # Deterministic forward: discounted intrinsic value.
        df = np.exp(-r * T)
        return df * (max(F - K, 0.0) if option_type == "c" else max(K - F, 0.0))

    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    df = np.exp(-r * T)
    if option_type == "c":
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def implied_vol_black76(price: float, F: float, K: float, T: float,
                        r: float, option_type: str,
                        cfg: PipelineConfig) -> float:
    """Invert a Black-76 price to implied volatility.

    Strategy: Brent's method (fast, guaranteed to converge inside a bracket)
    with a bisection-only fallback in the unlikely event Brent fails.
    Prices that fall outside the no-arbitrage band [intrinsic, upper bound]
    return NaN and are dropped downstream.
    """
    df = np.exp(-r * T)
    if option_type == "c":
        intrinsic = df * max(F - K, 0.0)
        upper = df * F
    else:
        intrinsic = df * max(K - F, 0.0)
        upper = df * K

    # A price at (or numerically below) intrinsic implies zero volatility.
    if price <= intrinsic + 1e-10:
        return 0.0
    if price >= upper:
        return np.nan  # price above the model-free upper bound: reject

    def objective(sigma: float) -> float:
        return black76_price(F, K, T, r, sigma, option_type) - price

    try:
        return optimize.brentq(objective, cfg.iv_lower, cfg.iv_upper,
                               xtol=1e-10, rtol=1e-12, maxiter=200)
    except (ValueError, RuntimeError):
        # Bisection fallback: slower but maximally robust.
        lo, hi = cfg.iv_lower, cfg.iv_upper
        f_lo = objective(lo)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            f_mid = objective(mid)
            if f_lo * f_mid <= 0.0:
                hi = mid
            else:
                lo, f_lo = mid, f_mid
            if hi - lo < 1e-10:
                break
        return 0.5 * (lo + hi)


# ============================================================================
# Put-call parity forward estimation
# ============================================================================
def estimate_forward_parity(df_slice: pd.DataFrame, T: float,
                            r: float) -> Tuple[float, float]:
    """Estimate the expiry forward from put-call parity via regression.

    Put-call parity for European options on a forward states

        C(K) - P(K) = exp(-r T) * (F - K)

    so regressing (C - P) on K across all strikes quoted in both calls and
    puts yields  slope = -exp(-rT)  and  intercept = exp(-rT) * F, hence

        F = intercept / (-slope).

    Using mid quotes for the regression makes the estimate robust to the
    bid-ask bounce of individual strikes. Returns (F, parity_rmse_bp), where
    the RMSE is expressed in basis points of F as a data-quality diagnostic.
    """
    calls = df_slice[df_slice["option_type"] == "c"].set_index("strike")
    puts = df_slice[df_slice["option_type"] == "p"].set_index("strike")
    common = calls.index.intersection(puts.index)
    if len(common) < 3:
        # Not enough matched strikes: fall back to median-strike parity pair.
        return np.nan, np.nan

    c_mid = calls.loc[common, "mid"].to_numpy()
    p_mid = puts.loc[common, "mid"].to_numpy()
    K = common.to_numpy(dtype=float)
    y = c_mid - p_mid

    # Ordinary least squares: y = a + b * K  =>  F = a / (-b).
    A = np.vstack([np.ones_like(K), K]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    intercept, slope = coef[0], coef[1]
    if slope >= -1e-6:
        return np.nan, np.nan  # degenerate regression, reject
    F = intercept / (-slope)

    # Residual of the parity identity, in basis points of the forward.
    resid = y - (intercept + slope * K)
    parity_rmse_bp = float(np.sqrt(np.mean(resid ** 2)) / F * 1e4)
    return float(F), parity_rmse_bp


# ============================================================================
# Data loading, validation, and quote cleaning
# ============================================================================
REQUIRED_COLUMNS = {
    "expiry": "Expiry date (YYYY-MM-DD) or DTE in days (numeric)",
    "strike": "Option strike price",
    "option_type": "'c'/'p' (also accepts 'call'/'put', 'CE'/'PE')",
    "bid": "Bid price (must be > 0 after cleaning)",
    "ask": "Ask price (must be >= bid)",
    "open_interest": "Open interest (must be > 0, unless volume > 0)",
    "volume": "Traded volume (must be > 0, unless open_interest > 0)",
}


def normalize_option_type(raw: object) -> Optional[str]:
    """Map the many vendor conventions for call/put flags onto 'c'/'p'."""
    s = str(raw).strip().lower()
    if s in {"c", "call", "ce", "0"}:
        return "c"
    if s in {"p", "put", "pe", "1"}:
        return "p"
    return None


def load_option_chain_csv(path: str) -> pd.DataFrame:
    """Load a real option chain CSV and harmonize column names.

    The loader is deliberately tolerant: column names are lowercased and
    common aliases are mapped onto the required schema. A snapshot date
    column is optional; when present, calendar expiries are converted to
    DTE. Raises ValueError with an actionable message when required columns
    are missing.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Option chain CSV not found: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Alias harmonization for common vendor schemas (OPRA, NSE bhavcopy, ...).
    aliases = {
        "type": "option_type", "cp_flag": "option_type", "optiontype": "option_type",
        "right": "option_type", "call_put": "option_type",
        "strike_price": "strike", "strikeprice": "strike", "k": "strike",
        "exp": "expiry", "expiration": "expiry", "expiry_date": "expiry",
        "expiration_date": "expiry", "maturity": "expiry",
        "oi": "open_interest", "openinterest": "open_interest",
        "vol": "volume", "contracts": "volume",
        "quote_date": "snapshot_date", "date": "snapshot_date",
        "trade_date": "snapshot_date", "asof": "snapshot_date",
    }
    df = df.rename(columns={k: v for k, v in aliases.items()
                            if k in df.columns and v not in df.columns})

    missing = [c for c in ("expiry", "strike", "option_type", "bid", "ask")
               if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns {missing}. Required schema:\n"
            + "\n".join(f"  - {k}: {v}" for k, v in REQUIRED_COLUMNS.items()))
    return df


def compute_dte(df: pd.DataFrame) -> pd.Series:
    """Return days-to-expiry for each row.

    Accepts either numeric DTE values or calendar expiry dates. Calendar
    dates require a snapshot_date column so the elapsed time is well defined.
    """
    exp = df["expiry"]
    if pd.api.types.is_numeric_dtype(exp):
        return exp.astype(float)

    exp_dt = pd.to_datetime(exp, errors="coerce")
    if "snapshot_date" in df.columns:
        snap_dt = pd.to_datetime(df["snapshot_date"], errors="coerce")
    else:
        # Assume the quote snapshot is "today"; logged loudly so the user is
        # aware of the convention.
        log.warning("No snapshot_date column found; assuming snapshot is today.")
        snap_dt = pd.Timestamp.today().normalize()
    return (exp_dt - snap_dt).dt.days.astype(float)


def clean_option_chain(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Apply the full quote cleaning and filtering pipeline.

    Filters, per the project specification:
        * bid > 0
        * open interest > 0 or volume > 0
        * 7 < DTE < 365
        * forward moneyness K/F in [0.8, 1.2]  (applied after forwards are
          estimated, inside ``add_implied_volatilities``)
        * internally consistent markets: ask >= bid, finite prices

    The function is pure (returns a new DataFrame) and logs a row-level
    accounting of every filter so the cleaning step is fully auditable.
    """
    n0 = len(df)
    out = df.copy()

    # Harmonize option types and numeric fields; rows that fail to parse are
    # dropped rather than silently coerced.
    out["option_type"] = out["option_type"].map(normalize_option_type)
    out = out.dropna(subset=["option_type"])
    for col in ("strike", "bid", "ask"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ("open_interest", "volume"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        else:
            out[col] = 0.0
    out = out.dropna(subset=["strike", "bid", "ask"])

    out["dte"] = compute_dte(out)
    out = out.dropna(subset=["dte"])
    n_parsed = len(out)

    # ---- Liquidity and sanity filters --------------------------------------
    out = out[out["bid"] > 0.0]                       # strictly positive bid
    n_bid = len(out)
    out = out[(out["open_interest"] > 0) | (out["volume"] > 0)]
    n_oi = len(out)
    out = out[(out["dte"] > 7) & (out["dte"] < 365)]  # 7 < DTE < 365
    n_dte = len(out)
    out = out[out["ask"] >= out["bid"]]               # locked/crossed markets
    n_crossed = len(out)
    out = out[(out["strike"] > 0) & np.isfinite(out["ask"])]
    n_finite = len(out)

    # Mid quote and half-spread are the calibration inputs downstream.
    out["mid"] = 0.5 * (out["bid"] + out["ask"])
    out["half_spread"] = 0.5 * (out["ask"] - out["bid"])

    log.info(
        "Cleaning: %d raw -> %d parsed -> %d bid>0 -> %d OI/vol>0 -> "
        "%d DTE in (7,365) -> %d uncrossed -> %d finite final rows",
        n0, n_parsed, n_bid, n_oi, n_dte, n_crossed, n_finite)
    if len(out) == 0:
        raise ValueError("No quotes survived the cleaning filters. "
                         "Check the input data against the documented schema.")
    return out.reset_index(drop=True)


def add_implied_volatilities(df: pd.DataFrame, cfg: PipelineConfig
                             ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate forwards, invert implied vols, and build the (k, w) panel.

    Returns
    -------
    panel : DataFrame
        One row per cleaned quote with columns dte, T, strike, option_type,
        mid, half_spread, iv, k (log-forward moneyness), w (total variance),
        and weight (calibration loss weight).
    forward_table : DataFrame
        One row per expiry with the parity-implied forward, the parity RMSE
        in basis points, and the number of quotes kept per slice.
    """
    rows: List[pd.DataFrame] = []
    fwd_rows: List[dict] = []

    for dte, slice_df in df.groupby("dte"):
        T = float(dte) / 365.0
        F, parity_bp = estimate_forward_parity(slice_df, T, cfg.risk_free_rate)
        if not np.isfinite(F) or F <= 0:
            log.warning("DTE %d: parity forward failed; slice skipped.", dte)
            continue

        s = slice_df.copy()
        s["T"] = T
        s["F"] = F

        # Out-of-the-money selection: keep calls with K >= F and puts with
        # K < F. Deep in-the-money quotes are dominated by intrinsic value,
        # carry almost no volatility information, and a one-tick price error
        # translates into a large implied volatility error. This is the
        # standard market practice for volatility surface construction.
        s = s[((s["option_type"] == "c") & (s["strike"] >= F))
              | ((s["option_type"] == "p") & (s["strike"] < F))]

        # Minimum time value guard: discard OTM quotes whose price is within
        # a tick of intrinsic value, since their implied vol is numerically
        # unstable even when they pass the OTM selection.
        df_rate = np.exp(-cfg.risk_free_rate * T)
        intrinsic = np.where(
            s["option_type"] == "c",
            df_rate * np.maximum(F - s["strike"], 0.0),
            df_rate * np.maximum(s["strike"] - F, 0.0))
        s = s[(s["mid"] - intrinsic) > 0.01]

        # Vectorized inversion would be faster, but per-quote inversion keeps
        # the code transparent and robust to individual bad quotes.
        s["iv"] = [
            implied_vol_black76(m, F, K, T, cfg.risk_free_rate, ot, cfg)
            for m, K, ot in zip(s["mid"], s["strike"], s["option_type"])
        ]
        s = s[np.isfinite(s["iv"]) & (s["iv"] > 0.0)]

        # Forward moneyness filter K/F in [0.8, 1.2].
        mny = s["strike"] / F
        s = s[(mny >= cfg.min_moneyness) & (mny <= cfg.max_moneyness)]
        if len(s) < cfg.min_strikes_per_slice:
            log.warning("DTE %d: only %d quotes after moneyness filter "
                        "(min %d); slice skipped.", dte, len(s),
                        cfg.min_strikes_per_slice)
            continue

        s["k"] = np.log(s["strike"] / F)          # log-forward moneyness
        s["w"] = s["iv"] ** 2 * s["T"]            # total implied variance

        # Loss weights favor tight spreads: weight is inversely proportional
        # to the squared bid-ask spread of the mid quote, normalized per
        # slice so weights have mean one within each expiry. Deep OTM options
        # with tiny prices and wide percentage spreads therefore influence
        # the fit less than liquid near-the-money quotes.
        spread_rel = np.maximum(s["half_spread"] / np.maximum(s["mid"], _EPS),
                                1e-3)
        w_raw = 1.0 / spread_rel ** 2
        s["weight"] = w_raw / w_raw.mean()

        rows.append(s)
        fwd_rows.append({
            "dte": float(dte), "T": T, "forward": F,
            "parity_rmse_bp": parity_bp, "n_quotes": len(s),
        })

    if not rows:
        raise ValueError("No expiry slices survived forward estimation and "
                         "the moneyness filter.")
    panel = pd.concat(rows, ignore_index=True)
    forward_table = pd.DataFrame(fwd_rows).sort_values("T").reset_index(drop=True)
    log.info("IV panel built: %d quotes across %d expiries.",
             len(panel), len(forward_table))
    return panel, forward_table


# ============================================================================
# Synthetic demonstration data generator
# ============================================================================
def ssvi_total_variance(k: np.ndarray, theta: float, rho: float,
                        phi: float) -> np.ndarray:
    """SSVI (Gatheral-Jacquier) total variance parametrization.

        w(k; theta_t) = theta_t / 2 * (1 + rho phi k
                                       + sqrt((phi k + rho)^2 + 1 - rho^2))

    where theta_t is the ATM total variance for maturity t and phi is the
    (here constant) volatility-of-variance scaling function. This
    parametrization is used to generate the synthetic ground truth surface.
    """
    return 0.5 * theta * (1.0 + rho * phi * k
                          + np.sqrt((phi * k + rho) ** 2 + 1.0 - rho ** 2))


# Stylized regime presets for the synthetic demonstration generator. These
# are deliberately different, clearly-labeled scenarios (not calibrated to
# any specific historical date) meant to stand in for a calm regime
# (2023-2024 style: low, gently upward-sloping ATM vol) and a stressed
# regime (2022 bear-market style: high ATM vol, an inverted/backwardated
# annualized-vol term structure, steeper negative skew, and wider,
# less-liquid spreads). Total variance theta_t = atm_vol(T)^2 * T remains
# non-decreasing in both regimes (variance accumulates with time even when
# annualized vol falls), so the calendar no-arbitrage precondition for SSVI
# is satisfied in both cases, exactly as it would need to be for a real
# stressed-market surface to be usable.
REGIME_PRESETS: Dict[str, Dict[str, float]] = {
    "calm": dict(
        atm_vol_base=0.13, atm_vol_amplitude=0.02, atm_vol_sign=+1.0,
        rho_true=-0.45, phi_true=0.60,
        spread_base=0.004, spread_wing=0.020,
    ),
    "stressed": dict(
        atm_vol_base=0.32, atm_vol_amplitude=0.05, atm_vol_sign=-1.0,
        rho_true=-0.65, phi_true=0.85,
        spread_base=0.008, spread_wing=0.035,
    ),
}


def generate_synthetic_chain(cfg: PipelineConfig) -> pd.DataFrame:
    """Generate a realistic, clearly labeled synthetic option chain.

    The ground truth is a smooth SSVI surface, with regime-dependent shape
    (see REGIME_PRESETS): "calm" stands in for a 2023-2024-style low-vol
    market with a gently upward-sloping ATM term structure and moderate
    negative skew; "stressed" stands in for a 2022-style bear market with a
    much higher ATM vol level, a backwardated (falling) annualized-vol term
    structure, steeper negative skew, and wider bid-ask spreads reflecting
    thinner liquidity. Option prices are computed with Black-76, then a
    bid-ask spread and microstructure noise are layered on so the
    calibration pipeline is exercised under realistic conditions. A small
    number of deliberately "bad" rows (zero bids, zero open interest,
    absurd maturities) is injected so the cleaning filters can be seen
    working.

    Returns a raw (uncleaned) DataFrame in the same schema as a real CSV.
    """
    rng = np.random.default_rng(cfg.random_seed)
    preset = REGIME_PRESETS[cfg.regime]
    log.info("Generating SYNTHETIC demo option chain (regime=%s, seed=%d).",
             cfg.regime, cfg.random_seed)

    # Ground truth SSVI parameters (regime-dependent stylized values).
    rho_true, phi_true = preset["rho_true"], preset["phi_true"]
    spot = 5000.0
    r, q = cfg.risk_free_rate, cfg.dividend_yield

    # Ten expiries spanning the allowed DTE window.
    dtes = np.array([14, 21, 35, 49, 70, 91, 133, 182, 273, 350], dtype=float)

    # ATM annualized-vol term structure. Calm: upward sloping. Stressed:
    # downward sloping / backwardated (front-month fear premium), but total
    # variance theta_t = atm_vol^2 * T still increases with T in both cases.
    atm_vols = (preset["atm_vol_base"]
                + preset["atm_vol_sign"] * preset["atm_vol_amplitude"]
                * (1.0 - np.exp(-dtes / 180.0)))

    rows = []
    for dte, atm_vol in zip(dtes, atm_vols):
        T = dte / 365.0
        F = spot * np.exp((r - q) * T)
        theta = atm_vol ** 2 * T

        # Strike grid covering forward moneyness roughly 0.75 to 1.25 so the
        # downstream moneyness filter has something to remove.
        moneyness = np.linspace(0.75, 1.25, 21)
        strikes = np.round(F * moneyness / 5.0) * 5.0  # 5-point strike grid
        for K in strikes:
            k = np.log(K / F)
            w = ssvi_total_variance(np.array([k]), theta, rho_true, phi_true)[0]
            iv_true = np.sqrt(max(w, 1e-10) / T)

            # Microstructure noise on the "true" mid, small relative to spread.
            iv_mid = iv_true + rng.normal(0.0, 0.0010)
            iv_mid = max(iv_mid, 1e-4)
            # Relative half-spread: tight ATM, wider in the wings, as in real
            # index option markets; regime-dependent overall liquidity level.
            rel_half_spread = (preset["spread_base"]
                               + preset["spread_wing"] * abs(k) ** 1.5)

            for ot in ("c", "p"):
                mid = black76_price(F, K, T, r, iv_mid, ot)
                if mid <= 0.01:  # worthless far-OTM wing: skip
                    continue
                half = rel_half_spread * mid
                bid, ask = mid - half, mid + half
                oi = float(rng.integers(500, 20000))
                vol = float(rng.integers(10, 5000))
                rows.append(dict(
                    expiry=int(dte), strike=float(K), option_type=ot,
                    bid=round(bid, 2), ask=round(ask, 2),
                    open_interest=oi, volume=vol,
                ))

        # Inject a few dirty rows per expiry to exercise the cleaning filters.
        dirty = dict(expiry=int(dte), strike=float(F), option_type="c",
                     bid=0.0, ask=5.0, open_interest=0.0, volume=0.0)
        rows.append(dirty)
    # A couple of absurd maturities, filtered by the DTE window.
    rows.append(dict(expiry=2, strike=spot, option_type="c", bid=10.0,
                     ask=11.0, open_interest=100, volume=10))
    rows.append(dict(expiry=800, strike=spot, option_type="p", bid=10.0,
                     ask=11.0, open_interest=100, volume=10))

    df = pd.DataFrame(rows)
    log.info("Synthetic chain: %d raw rows across %d expiries.",
             len(df), len(dtes))
    return df


# ============================================================================
# Raw SVI calibration (per expiry slice)
# ============================================================================
def raw_svi(k: np.ndarray, a: float, b: float, rho: float, m: float,
            sigma: float) -> np.ndarray:
    """Raw SVI total variance parametrization (Gatheral, 2004).

        w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

    Parameters
    ----------
    a     : vertical level; total variance at the smile minimum region.
    b     : overall angle of the asymptotes, b >= 0.
    rho   : correlation-like skew parameter, |rho| < 1.
    m     : horizontal translation of the smile minimum.
    sigma : smoothness of the smile vertex, sigma > 0.
    """
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


@dataclass
class SVISliceFit:
    """Container for one calibrated raw SVI slice and its diagnostics."""
    dte: float
    T: float
    forward: float
    params: Dict[str, float]          # a, b, rho, m, sigma
    rmse_vol_points: float            # unweighted RMSE in volatility points
    w_rmse: float                     # RMSE in total variance space
    n_quotes: int
    n_starts_tried: int
    best_start_index: int
    success: bool
    butterfly_min_g: float = np.nan   # filled in by diagnostics


def _svi_initial_guesses(k: np.ndarray, w: np.ndarray,
                         n_multistart: int) -> List[Tuple[float, ...]]:
    """Build a diverse set of initial parameter guesses for the SVI fit.

    A data-driven "anchor" guess is derived from moments of the observed
    smile (level, wing slope, location of the minimum). The remaining
    guesses perturb the anchor across the admissible parameter region so
    that the multi-start explores the landscape and guards against local
    minima, a well-documented practical issue with raw SVI calibration.
    """
    k_min_idx = int(np.argmin(w))
    m_anchor = float(k[k_min_idx])
    a_anchor = float(max(w.min(), 1e-6))
    # Rough wing slope from the most extreme strikes on each side.
    dk = max(k.max() - k.min(), 1e-3)
    b_anchor = float(max((w.max() - w.min()) / dk, 1e-4))

    guesses = [(a_anchor, b_anchor, -0.3, m_anchor, 0.10)]
    rng = np.random.default_rng(7)  # fixed seed: reproducible multistart
    for _ in range(n_multistart - 1):
        guesses.append((
            float(a_anchor * rng.uniform(0.5, 2.0)),
            float(b_anchor * rng.uniform(0.5, 3.0)),
            float(rng.uniform(-0.9, 0.9)),
            float(m_anchor + rng.uniform(-0.05, 0.05)),
            float(rng.uniform(0.02, 0.40)),
        ))
    return guesses


def fit_svi_slice(k: np.ndarray, w: np.ndarray, weights: np.ndarray,
                  iv: np.ndarray, T: float, dte: float, forward: float,
                  cfg: PipelineConfig) -> SVISliceFit:
    """Calibrate raw SVI to one expiry slice with a multi-start strategy.

    The objective is the weighted sum of squared total-variance residuals,
    with weights favoring tight-spread quotes. Parameter bounds enforce the
    SVI admissibility conditions b >= 0, |rho| < 1, sigma > 0, and keep the
    level a non-negative (a necessary slice-level no-arbitrage ingredient,
    since total variance can never be negative).

    Two further no-arbitrage conditions are enforced directly inside the
    fit, not just diagnosed afterward, per the project specification:

      * The Gatheral slice-level wing bound  b * (1 + |rho|) <= 4 / T.
      * The Gatheral-Jacquier butterfly condition g(k) >= 0, checked on a
        coarse penalty grid spanning well beyond the observed strikes.

    scipy's ``least_squares`` only supports box bounds, not general
    nonlinear constraints, so both conditions are added as large penalty
    residuals: any candidate parameter vector that violates them is pushed
    back into the feasible region by squared penalty terms that dominate
    the ordinary fit residuals whenever a violation occurs. This is
    equivalent in spirit to a constrained refit with tighter bounds, without
    needing a second (slower) constrained solver.
    """
    sqrt_w = np.sqrt(weights)

    # Coarse grid used to penalize butterfly violations during fitting. It
    # deliberately spans wider than the fine 801-point diagnostic grid's
    # range so the fit is pushed away from arbitrage before it is measured.
    k_pen = np.linspace(-0.6, 0.6, 121)
    G_MARGIN = 0.02     # safety buffer: penalize g(k) < margin, not just < 0,
                        # since the fine 801-point diagnostic grid is finer
                        # than this penalty grid and could find a worse point
    G_PENALTY = 15.0    # weight on butterfly (g(k) < margin) violations
    BW_PENALTY = 5.0    # weight on the b*(1+|rho|) <= 4/T wing bound

    def residuals(p: np.ndarray) -> np.ndarray:
        a, b, rho, m, sigma = p
        base = sqrt_w * (raw_svi(k, a, b, rho, m, sigma) - w)

        g_pen = raw_svi_g_function(k_pen, dict(a=a, b=b, rho=rho, m=m,
                                               sigma=sigma))
        g_penalty = G_PENALTY * np.maximum(0.0, G_MARGIN - g_pen)

        bw_viol = max(0.0, b * (1.0 + abs(rho)) - 4.0 / T)
        bw_penalty = np.array([BW_PENALTY * bw_viol])

        return np.concatenate([base, g_penalty, bw_penalty])

    # Parameter bounds: a >= 0 (non-negative total variance), b >= 0,
    # |rho| < 1 strictly, m free within data range plus a margin, sigma > 0.
    k_pad = max(0.5, 2.0 * np.max(np.abs(k)))
    lb = np.array([0.0, 0.0, -cfg.ssvi_rho_bound, -k_pad, 1e-4])
    ub = np.array([max(w.max() * 5.0, 1.0), max(w.max() * 20.0, 2.0),
                   cfg.ssvi_rho_bound, k_pad, 5.0])

    best = None
    guesses = _svi_initial_guesses(k, w, cfg.n_multistart)
    for i, p0 in enumerate(guesses):
        p0 = np.clip(np.asarray(p0, dtype=float), lb + 1e-8, ub - 1e-8)
        try:
            sol = optimize.least_squares(
                residuals, p0, bounds=(lb, ub), method="trf",
                x_scale="jac", max_nfev=5000)
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("SVI start %d failed: %s", i, exc)
            continue
        if best is None or sol.cost < best[0]:
            best = (sol.cost, i, sol)

    if best is None:
        raise RuntimeError(f"All SVI multistarts failed for DTE {dte}.")

    _, best_idx, sol = best
    a, b, rho, m, sigma = sol.x
    w_fit = raw_svi(k, a, b, rho, m, sigma)

    # Fit quality in volatility points (the unit practitioners quote).
    iv_fit = np.sqrt(np.maximum(w_fit, _EPS) / T)
    rmse_vol_points = float(np.sqrt(np.mean((iv_fit - iv) ** 2)) * 100.0)
    w_rmse = float(np.sqrt(np.mean((w_fit - w) ** 2)))

    return SVISliceFit(
        dte=float(dte), T=float(T), forward=float(forward),
        params=dict(a=float(a), b=float(b), rho=float(rho),
                    m=float(m), sigma=float(sigma)),
        rmse_vol_points=rmse_vol_points, w_rmse=w_rmse,
        n_quotes=len(k), n_starts_tried=len(guesses),
        best_start_index=int(best_idx), success=True)


# ============================================================================
# SSVI surface construction (joint fit with monotone ATM total variance)
# ============================================================================
def ssvi_phi_powerlaw(theta: np.ndarray, eta: float, gamma: float
                      ) -> np.ndarray:
    """SSVI power-law phi function:  phi(theta) = eta / theta^gamma.

    Heston-like short-maturity behavior requires the power-law form with
    gamma in (0, 1/2]; together with theta non-decreasing this gives a
    surface free of calendar spread arbitrage (Gatheral-Jacquier, 2014,
    Theorem 4.2, for constant rho and this phi family).
    """
    return eta / np.power(np.maximum(theta, _EPS), gamma)


@dataclass
class SSVIFit:
    """Container for the calibrated SSVI surface and its diagnostics."""
    theta: np.ndarray          # ATM total variance per maturity (sorted T)
    T: np.ndarray              # maturities (sorted)
    rho: float
    eta: float
    gamma: float
    rmse_vol_points: float     # surface-level RMSE in volatility points
    monotone_theta: bool       # True if theta is non-decreasing by tolerance
    calendar_violations: int   # number of negative dw/dT grid violations
    butterfly_violations: int  # number of g(k) < 0 grid violations
    butterfly_min_g: float     # worst value of g(k) on the fine grid
    success: bool = True


def fit_ssvi_surface(svi_fits: List[SVISliceFit], panel: pd.DataFrame,
                     cfg: PipelineConfig) -> SSVIFit:
    """Fit the SSVI surface jointly across all expiries.

    Two-stage procedure:

      Stage 1: recover the ATM total variance theta_t per expiry directly
               from each raw SVI slice (theta_t = w_svi(0)), which is a
               stable, nearly model-free anchor, then project the sequence
               onto the non-decreasing cone via isotonic-style cumulative
               maximum with a small tolerance.
      Stage 2: jointly calibrate the global shape parameters (rho, eta,
               gamma) against the observed total variance panel, holding the
               monotone theta sequence fixed. Fixing theta first turns the
               joint problem into a small, well-conditioned 3-parameter fit
               and guarantees by construction that total variance is
               non-decreasing in maturity, i.e. no calendar arbitrage.

    Returns an SSVIFit with arbitrage diagnostics already populated.
    """
    svi_fits = sorted(svi_fits, key=lambda f: f.T)
    T_arr = np.array([f.T for f in svi_fits])

    # --- Stage 1: ATM total variance term structure -------------------------
    theta_raw = np.array([raw_svi(np.array([0.0]), f.params["a"],
                                  f.params["b"], f.params["rho"],
                                  f.params["m"], f.params["sigma"])[0]
                          for f in svi_fits])
    theta_raw = np.maximum(theta_raw, 1e-6)

    # Isotonic projection (non-decreasing): cumulative maximum of a tiny
    # tolerance-adjusted sequence. Violations smaller than fit noise are
    # absorbed without distorting the level.
    theta_mono = np.maximum.accumulate(theta_raw)

    # --- Stage 2: global shape parameters -----------------------------------
    # Map each panel row to its maturity index.
    T_to_idx = {float(T): i for i, T in enumerate(T_arr)}
    idx = panel["T"].map(T_to_idx).to_numpy(dtype=int)
    k_obs = panel["k"].to_numpy()
    w_obs = panel["w"].to_numpy()
    sw = np.sqrt(panel["weight"].to_numpy())
    theta_obs = theta_mono[idx]

    def residuals(p: np.ndarray) -> np.ndarray:
        rho, eta, gamma = p
        phi = ssvi_phi_powerlaw(theta_obs, eta, gamma)
        w_model = ssvi_total_variance(k_obs, theta_obs, rho, phi)
        return sw * (w_model - w_obs)

    # Multi-start over a coarse grid of the skew parameter: the shape fit is
    # only mildly non-convex but a few starts are cheap insurance.
    best = None
    for rho0 in (-0.8, -0.4, 0.0, 0.4):
        p0 = np.array([rho0, 0.5, 0.4])
        sol = optimize.least_squares(
            residuals, p0,
            bounds=([-cfg.ssvi_rho_bound, 1e-3, 1e-3],
                    [cfg.ssvi_rho_bound, 10.0, 0.5]),
            method="trf", x_scale="jac", max_nfev=5000)
        if best is None or sol.cost < best.cost:
            best = sol
    rho, eta, gamma = best.x

    # Surface-level RMSE in volatility points.
    phi_obs = ssvi_phi_powerlaw(theta_obs, eta, gamma)
    w_fit = ssvi_total_variance(k_obs, theta_obs, rho, phi_obs)
    iv_fit = np.sqrt(np.maximum(w_fit, _EPS) / panel["T"].to_numpy())
    rmse_vp = float(np.sqrt(np.mean((iv_fit - panel["iv"].to_numpy()) ** 2))
                    * 100.0)

    fit = SSVIFit(theta=theta_mono, T=T_arr, rho=float(rho), eta=float(eta),
                  gamma=float(gamma), rmse_vol_points=rmse_vp,
                  monotone_theta=bool(np.all(np.diff(theta_mono) >= -1e-10)),
                  calendar_violations=-1, butterfly_violations=-1,
                  butterfly_min_g=np.nan)
    return fit


# ============================================================================
# Arbitrage diagnostics
# ============================================================================
def ssvi_g_function(k: np.ndarray, theta: float, rho: float,
                    phi: float) -> np.ndarray:
    """Gatheral-Jacquier g(k) for one SSVI slice.

    A volatility surface is free of butterfly arbitrage wherever

        g(k) = (1 - k w'(k) / (2 w(k)))^2
               - w'(k)^2 / 4 * (1 / w(k) + 1 / 4)
               + w''(k) / 2  >=  0

    For SSVI the required derivatives have closed forms. Writing
    S = sqrt((phi k + rho)^2 + 1 - rho^2):
        w   = theta/2 * (1 + rho phi k + S)
        w'  = theta/2 * (rho phi + phi (phi k + rho) / S)
        w'' = theta/2 * phi^2 (1 - rho^2) / S^3
    """
    S = np.sqrt((phi * k + rho) ** 2 + 1.0 - rho ** 2)
    w = 0.5 * theta * (1.0 + rho * phi * k + S)
    dw = 0.5 * theta * phi * (rho + (phi * k + rho) / S)
    d2w = 0.5 * theta * phi ** 2 * (1.0 - rho ** 2) / S ** 3
    w = np.maximum(w, _EPS)
    g = ((1.0 - k * dw / (2.0 * w)) ** 2
         - 0.25 * dw ** 2 * (1.0 / w + 0.25)
         + 0.5 * d2w)
    return g


def raw_svi_g_function(k: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    """g(k) for a raw SVI slice, using the closed-form SVI derivatives."""
    a, b, rho, m, sigma = p["a"], p["b"], p["rho"], p["m"], p["sigma"]
    x = k - m
    S = np.sqrt(x ** 2 + sigma ** 2)
    w = a + b * (rho * x + S)
    dw = b * (rho + x / S)
    d2w = b * sigma ** 2 / S ** 3
    w = np.maximum(w, _EPS)
    return ((1.0 - k * dw / (2.0 * w)) ** 2
            - 0.25 * dw ** 2 * (1.0 / w + 0.25)
            + 0.5 * d2w)


def run_butterfly_diagnostics_svi(svi_fits: List[SVISliceFit],
                                  cfg: PipelineConfig) -> pd.DataFrame:
    """Evaluate g(k) on a fine grid for every raw SVI slice.

    Raw SVI slices fitted independently are not guaranteed to be butterfly
    free; the diagnostic quantifies any violations honestly rather than
    hiding them. The minimum of g(k) per slice is stored back on the fit
    objects for the summary table.
    """
    rows = []
    k_grid = np.linspace(-0.5, 0.5, cfg.n_fine_grid)  # ~ [0.61, 1.65] in K/F
    for f in svi_fits:
        g = raw_svi_g_function(k_grid, f.params)
        f.butterfly_min_g = float(g.min())
        n_viol = int(np.sum(g < -1e-10))
        rows.append(dict(dte=f.dte, T=f.T, n_violations=n_viol,
                         min_g=float(g.min()),
                         worst_k=float(k_grid[int(np.argmin(g))])))
    return pd.DataFrame(rows)


def run_surface_arbitrage_diagnostics(ssvi: SSVIFit, cfg: PipelineConfig
                                      ) -> SSVIFit:
    """Calendar and butterfly diagnostics for the SSVI surface.

    Calendar: total variance must be non-decreasing in T for every fixed k.
    We check dw/dT >= 0 on a fine (k, T) grid using finite differences of
    the SSVI surface (theta is monotone by construction, but the check also
    catches pathological phi behavior).

    Butterfly: g(k) >= 0 on a fine k-grid for every maturity.
    """
    k_grid = np.linspace(-0.5, 0.5, cfg.n_fine_grid)

    # --- Calendar check: w(k, T) non-decreasing in T for each k -------------
    cal_viol = 0
    for i in range(len(ssvi.T) - 1):
        t0, t1 = ssvi.T[i], ssvi.T[i + 1]
        th0, th1 = ssvi.theta[i], ssvi.theta[i + 1]
        phi0 = ssvi_phi_powerlaw(np.array([th0]), ssvi.eta, ssvi.gamma)[0]
        phi1 = ssvi_phi_powerlaw(np.array([th1]), ssvi.eta, ssvi.gamma)[0]
        w0 = ssvi_total_variance(k_grid, th0, ssvi.rho, phi0)
        w1 = ssvi_total_variance(k_grid, th1, ssvi.rho, phi1)
        cal_viol += int(np.sum((w1 - w0) < -1e-10))

    # --- Butterfly check: g(k) >= 0 per slice -------------------------------
    fly_viol, min_g = 0, np.inf
    for th in ssvi.theta:
        phi = ssvi_phi_powerlaw(np.array([th]), ssvi.eta, ssvi.gamma)[0]
        g = ssvi_g_function(k_grid, th, ssvi.rho, phi)
        fly_viol += int(np.sum(g < -1e-10))
        min_g = min(min_g, float(g.min()))

    ssvi.calendar_violations = cal_viol
    ssvi.butterfly_violations = fly_viol
    ssvi.butterfly_min_g = float(min_g)
    return ssvi


# ============================================================================
# Visualization
# ============================================================================
def plot_slice_fits(svi_fits: List[SVISliceFit], panel: pd.DataFrame,
                    cfg: PipelineConfig, tag: str) -> str:
    """Grid of per-expiry smile fits: market IVs vs raw SVI fit."""
    n = len(svi_fits)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.0 * nrows),
                             squeeze=False)
    for ax, f in zip(axes.flat, svi_fits):
        s = panel[np.isclose(panel["T"], f.T)]
        k_fine = np.linspace(s["k"].min(), s["k"].max(), 200)
        w_fit = raw_svi(k_fine, f.params["a"], f.params["b"],
                        f.params["rho"], f.params["m"], f.params["sigma"])
        iv_fit = np.sqrt(np.maximum(w_fit, _EPS) / f.T)
        ax.scatter(s["k"], s["iv"] * 100, s=18, alpha=0.7,
                   label="market IV (mid)")
        ax.plot(k_fine, iv_fit * 100, "r-", lw=1.8, label="raw SVI fit")
        ax.set_title(f"DTE {int(f.dte)} | RMSE {f.rmse_vol_points:.3f} vol pts",
                     fontsize=9)
        ax.set_xlabel("log-forward moneyness k")
        ax.set_ylabel("implied vol (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle(f"Raw SVI slice fits - {tag}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(cfg.output_dir, "fig1_svi_slice_fits.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_surface(ssvi: SSVIFit, panel: pd.DataFrame, cfg: PipelineConfig,
                 tag: str) -> str:
    """3-D surface of SSVI implied vols overlaid with market quotes."""
    k_grid = np.linspace(-0.35, 0.35, 120)
    T_grid = np.linspace(ssvi.T.min(), ssvi.T.max(), 60)
    K, Tm = np.meshgrid(k_grid, T_grid)
    theta_grid = np.interp(Tm.ravel(), ssvi.T, ssvi.theta).reshape(Tm.shape)
    phi_grid = ssvi_phi_powerlaw(theta_grid, ssvi.eta, ssvi.gamma)
    W = ssvi_total_variance(K.ravel(), theta_grid.ravel(),
                            np.full(K.size, ssvi.rho),
                            phi_grid.ravel()).reshape(K.shape)
    IV = np.sqrt(np.maximum(W, _EPS) / Tm) * 100.0

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(K, Tm * 365.0, IV, cmap="viridis", alpha=0.85,
                    linewidth=0, antialiased=True)
    ax.scatter(panel["k"], panel["T"] * 365.0, panel["iv"] * 100,
               c="red", s=4, alpha=0.4, label="market quotes")
    ax.set_xlabel("log-forward moneyness k")
    ax.set_ylabel("days to expiry")
    ax.set_zlabel("implied vol (%)")
    ax.set_title(f"SSVI implied volatility surface - {tag}\n"
                 f"rho={ssvi.rho:.3f}, eta={ssvi.eta:.3f}, "
                 f"gamma={ssvi.gamma:.3f}")
    ax.view_init(elev=22, azim=-135)
    fig.tight_layout()
    path = os.path.join(cfg.output_dir, "fig2_ssvi_surface.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_arbitrage_diagnostics(ssvi: SSVIFit, svi_fly: pd.DataFrame,
                               cfg: PipelineConfig, tag: str) -> str:
    """Two-panel diagnostic: theta term structure and g(k) on a fine grid."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # Panel 1: ATM total variance term structure (calendar arbitrage view).
    ax1.plot(ssvi.T * 365.0, ssvi.theta, "o-", color="navy")
    ax1.set_title("ATM total variance $\\theta_t$ (calendar diagnostic)")
    ax1.set_xlabel("days to expiry")
    ax1.set_ylabel("$\\theta_t$")
    ax1.grid(alpha=0.3)
    ax1.text(0.02, 0.95,
             f"monotone non-decreasing: {ssvi.monotone_theta}\n"
             f"calendar grid violations: {ssvi.calendar_violations}",
             transform=ax1.transAxes, va="top", fontsize=9,
             bbox=dict(fc="white", alpha=0.8))

    # Panel 2: g(k) per SSVI slice; butterfly free iff g(k) >= 0 everywhere.
    k_grid = np.linspace(-0.5, 0.5, cfg.n_fine_grid)
    for th, T in zip(ssvi.theta, ssvi.T):
        phi = ssvi_phi_powerlaw(np.array([th]), ssvi.eta, ssvi.gamma)[0]
        g = ssvi_g_function(k_grid, th, ssvi.rho, phi)
        ax2.plot(k_grid, g, lw=1.0,
                 label=f"{int(round(T*365))}d")
    ax2.axhline(0.0, color="red", ls="--", lw=1.2)
    ax2.set_title("Butterfly diagnostic $g(k)$ per SSVI slice "
                  "(must be $\\geq 0$)")
    ax2.set_xlabel("log-forward moneyness k")
    ax2.set_ylabel("$g(k)$")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7, ncol=2, title="expiry")
    fig.suptitle(f"Arbitrage diagnostics - {tag} | butterfly violations: "
                 f"{ssvi.butterfly_violations}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(cfg.output_dir, "fig3_arbitrage_diagnostics.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_residual_diagnostics(svi_fits: List[SVISliceFit],
                              panel: pd.DataFrame, cfg: PipelineConfig,
                              tag: str) -> str:
    """Residual diagnostics: a true (expiry x moneyness) residual heatmap,
    plus a scatter panel of the same residuals against moneyness.

    Underfitting would show as systematic U-shaped or skew-shaped structure
    in the residuals (a colored band running across the heatmap); overfitting
    would show as the fitted curve tracking individual noisy quotes (no
    visible structure, just noise). A well-specified fit shows small,
    unstructured residuals centered on zero in both panels.
    """
    k_bin_edges = np.linspace(-0.45, 0.45, 19)   # 18 moneyness bins
    k_bin_centers = 0.5 * (k_bin_edges[:-1] + k_bin_edges[1:])
    dtes_sorted = sorted({int(f.dte) for f in svi_fits})
    heat = np.full((len(dtes_sorted), len(k_bin_centers)), np.nan)

    fig, (ax_heat, ax_scatter) = plt.subplots(
        1, 2, figsize=(14, 5.5), gridspec_kw=dict(width_ratios=[1.1, 1.0]))

    all_res = []
    for row_idx, f in enumerate(svi_fits):
        s = panel[np.isclose(panel["T"], f.T)]
        w_fit = raw_svi(s["k"].to_numpy(), f.params["a"], f.params["b"],
                        f.params["rho"], f.params["m"], f.params["sigma"])
        iv_fit = np.sqrt(np.maximum(w_fit, _EPS) / f.T)
        res = (iv_fit - s["iv"].to_numpy()) * 100.0
        all_res.append(res)
        ax_scatter.scatter(s["k"], res, s=12, alpha=0.55,
                           label=f"{int(f.dte)}d")

        # Bin residuals into the moneyness grid for this expiry's heatmap row.
        row = dtes_sorted.index(int(f.dte))
        bin_idx = np.digitize(s["k"].to_numpy(), k_bin_edges) - 1
        for b in range(len(k_bin_centers)):
            mask = bin_idx == b
            if mask.any():
                heat[row, b] = float(np.mean(res[mask]))

    all_res = np.concatenate(all_res)
    ax_scatter.axhline(0.0, color="black", lw=1.0)
    ax_scatter.axhline(np.mean(all_res) + 2 * np.std(all_res), color="red",
                       ls=":", lw=1.0)
    ax_scatter.axhline(np.mean(all_res) - 2 * np.std(all_res), color="red",
                       ls=":", lw=1.0)
    ax_scatter.set_title("Residuals by quote (dotted lines: +/- 2 std)")
    ax_scatter.set_xlabel("log-forward moneyness k")
    ax_scatter.set_ylabel("fit IV minus market IV (vol points)")
    ax_scatter.grid(alpha=0.3)
    ax_scatter.legend(fontsize=6, ncol=2, title="expiry")

    vmax = np.nanmax(np.abs(heat)) if np.any(np.isfinite(heat)) else 1.0
    vmax = max(vmax, 1e-6)
    im = ax_heat.imshow(heat, aspect="auto", cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax, origin="lower",
                        extent=[k_bin_edges[0], k_bin_edges[-1],
                                -0.5, len(dtes_sorted) - 0.5])
    ax_heat.set_yticks(range(len(dtes_sorted)))
    ax_heat.set_yticklabels([f"{d}d" for d in dtes_sorted])
    ax_heat.set_xlabel("log-forward moneyness k (binned)")
    ax_heat.set_ylabel("expiry (days to expiry)")
    ax_heat.set_title("Residual heatmap: fit IV minus market IV\n"
                      "(white/blank = no quotes in that bin)")
    cbar = fig.colorbar(im, ax=ax_heat)
    cbar.set_label("vol points")

    fig.suptitle(f"SVI residual diagnostics - {tag}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(cfg.output_dir, "fig4_residual_diagnostics.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ============================================================================
# Summary tables and reporting
# ============================================================================
def build_summary_tables(svi_fits: List[SVISliceFit], ssvi: SSVIFit,
                         forward_table: pd.DataFrame, cfg: PipelineConfig
                         ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble the per-slice and surface-level summary tables."""
    slice_tbl = pd.DataFrame([dict(
        dte=f.dte, T=f.T, forward=f.forward,
        a=f.params["a"], b=f.params["b"], rho=f.params["rho"],
        m=f.params["m"], sigma=f.params["sigma"],
        rmse_vol_points=f.rmse_vol_points, w_rmse=f.w_rmse,
        n_quotes=f.n_quotes, n_starts_tried=f.n_starts_tried,
        best_start_index=f.best_start_index,
        butterfly_min_g=f.butterfly_min_g) for f in svi_fits])

    surface_tbl = pd.DataFrame([dict(
        n_expiries=len(ssvi.T),
        ssvi_rho=ssvi.rho, ssvi_eta=ssvi.eta, ssvi_gamma=ssvi.gamma,
        surface_rmse_vol_points=ssvi.rmse_vol_points,
        median_slice_rmse_vol_points=float(
            slice_tbl["rmse_vol_points"].median()),
        theta_monotone=ssvi.monotone_theta,
        calendar_violations=ssvi.calendar_violations,
        butterfly_violations_ssvi=ssvi.butterfly_violations,
        butterfly_min_g_ssvi=ssvi.butterfly_min_g,
        parity_max_rmse_bp=float(forward_table["parity_rmse_bp"].max()),
        parity_median_rmse_bp=float(forward_table["parity_rmse_bp"].median()),
    )])
    return slice_tbl, surface_tbl


def print_report(forward_table: pd.DataFrame, slice_tbl: pd.DataFrame,
                 surface_tbl: pd.DataFrame, cfg: PipelineConfig,
                 tag: str) -> None:
    """Print a human-readable validation report to the console."""
    med_rmse = surface_tbl["median_slice_rmse_vol_points"].iloc[0]
    par_bp = surface_tbl["parity_median_rmse_bp"].iloc[0]
    cal = int(surface_tbl["calendar_violations"].iloc[0])
    fly = int(surface_tbl["butterfly_violations_ssvi"].iloc[0])

    lines = [
        "",
        "=" * 72,
        f"  VALIDATION REPORT  ({tag})",
        "=" * 72,
        f"  Expiries fitted                    : {len(slice_tbl)}",
        f"  Parity residual (median, bp of F)  : {par_bp:.4f}   "
        f"(target < {cfg.parity_max_residual_bp} bp)",
        f"  Median slice RMSE (vol points)     : {med_rmse:.4f}   "
        "(target < 0.5 on calm synthetic/demo data)",
        f"  Theta non-decreasing in maturity   : "
        f"{bool(surface_tbl['theta_monotone'].iloc[0])}",
        f"  Calendar arbitrage violations      : {cal}   (target 0)",
        f"  Butterfly violations (SSVI, grid)  : {fly}   (target 0)",
        f"  Worst g(k) on fine grid (SSVI)     : "
        f"{surface_tbl['butterfly_min_g_ssvi'].iloc[0]:.6f}",
        "=" * 72,
        "  Per-slice raw SVI results:",
        slice_tbl[["dte", "rmse_vol_points", "a", "b", "rho", "m",
                   "sigma", "butterfly_min_g"]].round(5).to_string(index=False),
        "=" * 72,
        "",
    ]
    print("\n".join(str(x) for x in lines))


# ============================================================================
# Pipeline orchestration and command line interface
# ============================================================================
def run_pipeline(cfg: PipelineConfig, input_csv: Optional[str],
                 demo: bool) -> Dict[str, object]:
    """Execute the full volatility surface pipeline end to end.

    Parameters
    ----------
    cfg : PipelineConfig
        All tunable settings.
    input_csv : str or None
        Path to a real option chain CSV (real-data mode).
    demo : bool
        If True, ignore input_csv and generate synthetic demo data instead.

    Returns
    -------
    dict
        The key intermediate and final objects (forward_table, panel,
        svi_fits, ssvi, slice_tbl, surface_tbl, tag), so callers such as the
        regime-comparison driver can build a cross-run comparison without
        re-parsing CSVs from disk.
    """
    t0 = time.time()
    os.makedirs(cfg.output_dir, exist_ok=True)

    # --- Step 1: data acquisition ------------------------------------------
    if demo:
        tag = f"SYNTHETIC DEMO DATA ({cfg.regime.upper()} REGIME)"
        raw = generate_synthetic_chain(cfg)
        demo_csv = os.path.join(cfg.output_dir, "synthetic_option_chain.csv")
        raw.to_csv(demo_csv, index=False)
        log.info("Synthetic raw chain saved to %s", demo_csv)
    else:
        tag = "REAL MARKET DATA"
        raw = load_option_chain_csv(input_csv)

    # --- Step 2: cleaning and validation ------------------------------------
    clean = clean_option_chain(raw, cfg)

    # --- Step 3: forwards, implied vols, (k, w) panel ------------------------
    panel, forward_table = add_implied_volatilities(clean, cfg)
    panel.to_csv(os.path.join(cfg.output_dir, "iv_panel.csv"), index=False)
    forward_table.to_csv(os.path.join(cfg.output_dir, "forwards.csv"),
                         index=False)

    # --- Step 4: raw SVI calibration per expiry ------------------------------
    svi_fits: List[SVISliceFit] = []
    for _, fwd in forward_table.iterrows():
        s = panel[np.isclose(panel["T"], fwd["T"])]
        fit = fit_svi_slice(
            s["k"].to_numpy(), s["w"].to_numpy(), s["weight"].to_numpy(),
            s["iv"].to_numpy(), float(fwd["T"]), float(fwd["dte"]),
            float(fwd["forward"]), cfg)
        svi_fits.append(fit)
        log.info("DTE %3d | SVI RMSE %6.4f vol pts | params a=%.5f b=%.4f "
                 "rho=%+.3f m=%+.4f sigma=%.4f",
                 int(fwd["dte"]), fit.rmse_vol_points, fit.params["a"],
                 fit.params["b"], fit.params["rho"], fit.params["m"],
                 fit.params["sigma"])

    # --- Step 5: butterfly diagnostics on raw SVI slices ---------------------
    svi_fly_tbl = run_butterfly_diagnostics_svi(svi_fits, cfg)
    svi_fly_tbl.to_csv(os.path.join(cfg.output_dir,
                                    "butterfly_diagnostics_svi.csv"),
                       index=False)
    n_svi_viol = int((svi_fly_tbl["n_violations"] > 0).sum())
    if n_svi_viol:
        log.warning("%d raw SVI slice(s) show butterfly violations on the "
                    "fine grid; this is expected for independently fitted "
                    "slices and is corrected by the SSVI surface.", n_svi_viol)

    # --- Step 6: SSVI surface with monotone ATM total variance ---------------
    ssvi = fit_ssvi_surface(svi_fits, panel, cfg)
    ssvi = run_surface_arbitrage_diagnostics(ssvi, cfg)
    log.info("SSVI: rho=%.4f eta=%.4f gamma=%.4f | surface RMSE %.4f vol pts "
             "| calendar violations %d | butterfly violations %d",
             ssvi.rho, ssvi.eta, ssvi.gamma, ssvi.rmse_vol_points,
             ssvi.calendar_violations, ssvi.butterfly_violations)

    # --- Step 7: summaries, CSVs, figures ------------------------------------
    slice_tbl, surface_tbl = build_summary_tables(svi_fits, ssvi,
                                                  forward_table, cfg)
    slice_tbl.to_csv(os.path.join(cfg.output_dir, "svi_slice_summary.csv"),
                     index=False)
    surface_tbl.to_csv(os.path.join(cfg.output_dir, "surface_summary.csv"),
                       index=False)

    figures = [
        plot_slice_fits(svi_fits, panel, cfg, tag),
        plot_surface(ssvi, panel, cfg, tag),
        plot_arbitrage_diagnostics(ssvi, svi_fly_tbl, cfg, tag),
        plot_residual_diagnostics(svi_fits, panel, cfg, tag),
    ]
    for p in figures:
        log.info("Figure written: %s", p)

    print_report(forward_table, slice_tbl, surface_tbl, cfg, tag)

    # --- Step 8: mathematical sanity assertions ------------------------------
    # Non-negative total variance everywhere on the SSVI grid.
    k_chk = np.linspace(-0.5, 0.5, 201)
    for th in ssvi.theta:
        phi = ssvi_phi_powerlaw(np.array([th]), ssvi.eta, ssvi.gamma)[0]
        w_chk = ssvi_total_variance(k_chk, th, ssvi.rho, phi)
        assert np.all(w_chk > 0), "Non-positive SSVI total variance found."
    assert np.all(panel["iv"] > 0), "Non-positive implied vol in panel."
    assert np.all(panel["w"] >= 0), "Negative total variance in panel."
    assert ssvi.monotone_theta, "ATM total variance not monotone in maturity."

    log.info("Pipeline completed in %.2f s. Outputs in %s/",
             time.time() - t0, cfg.output_dir)

    return dict(tag=tag, forward_table=forward_table, panel=panel,
                svi_fits=svi_fits, ssvi=ssvi, slice_tbl=slice_tbl,
                surface_tbl=surface_tbl, svi_fly_tbl=svi_fly_tbl)


# ============================================================================
# Calm vs. stressed regime comparison
# ============================================================================
def run_regime_comparison(cfg: PipelineConfig, output_root: str) -> None:
    """Run the full pipeline once per regime and report a head-to-head.

    Writes ``outputs/calm/`` and ``outputs/stressed/`` with the complete set
    of per-regime figures and CSVs (each internally consistent and fully
    diagnosed), plus a top-level ``regime_comparison_summary.csv`` and
    ``fig5_regime_comparison.png`` quantifying how fit quality, ATM vol
    level, skew, and quoted cost differ between the two regimes, as required
    by the project specification's calm-vs-stressed comparison.
    """
    os.makedirs(output_root, exist_ok=True)
    results: Dict[str, Dict[str, object]] = {}
    for regime in ("calm", "stressed"):
        regime_cfg = dataclass_replace(cfg, regime=regime,
                                       output_dir=os.path.join(output_root, regime))
        log.info("=== Running %s regime ===", regime.upper())
        results[regime] = run_pipeline(regime_cfg, input_csv=None, demo=True)

    rows = []
    for regime, res in results.items():
        surf = res["surface_tbl"].iloc[0]
        fwd = res["forward_table"]
        panel = res["panel"]
        # ATM-ish average relative half-spread as a liquidity/cost proxy,
        # using the near-the-money quotes (|k| < 0.05) across all expiries.
        atm_panel = panel[panel["k"].abs() < 0.05]
        avg_atm_rel_spread = float(
            (atm_panel["half_spread"] / atm_panel["mid"]).mean() * 1e4)
        short_dte_iv = float(panel.loc[panel["T"].idxmin(), "iv"]) \
            if len(panel) else np.nan
        # Average ATM implied vol per slice (proxy for the level of the
        # regime), using the closest-to-ATM quote in each expiry.
        atm_ivs = []
        for _, fr in fwd.iterrows():
            s = panel[np.isclose(panel["T"], fr["T"])]
            if len(s):
                atm_ivs.append(float(s.iloc[(s["k"].abs()).argmin()]["iv"]))
        rows.append(dict(
            regime=regime,
            n_expiries=int(surf["n_expiries"]),
            avg_atm_iv_pct=float(np.mean(atm_ivs) * 100.0) if atm_ivs else np.nan,
            ssvi_rho=float(surf["ssvi_rho"]),
            ssvi_eta=float(surf["ssvi_eta"]),
            surface_rmse_vol_points=float(surf["surface_rmse_vol_points"]),
            median_slice_rmse_vol_points=float(surf["median_slice_rmse_vol_points"]),
            calendar_violations=int(surf["calendar_violations"]),
            butterfly_violations_ssvi=int(surf["butterfly_violations_ssvi"]),
            butterfly_violations_svi_slices=int(
                (res["svi_fly_tbl"]["n_violations"] > 0).sum()),
            parity_median_rmse_bp=float(surf["parity_median_rmse_bp"]),
            avg_atm_relative_spread_bp=avg_atm_rel_spread,
            shortest_dte_iv_pct=float(short_dte_iv * 100.0)
                                if np.isfinite(short_dte_iv) else np.nan,
        ))
    comparison = pd.DataFrame(rows)
    comp_path = os.path.join(output_root, "regime_comparison_summary.csv")
    comparison.to_csv(comp_path, index=False)
    log.info("Regime comparison written to %s", comp_path)
    print("\nRegime comparison (calm vs stressed):\n" +
          comparison.round(4).to_string(index=False) + "\n")

    # --- Comparison figure: ATM term structure, skew, RMSE, cost ------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    colors = {"calm": "tab:blue", "stressed": "tab:red"}

    ax = axes[0, 0]
    for regime, res in results.items():
        ssvi = res["ssvi"]
        atm_vol = np.sqrt(np.maximum(ssvi.theta, 1e-12) / ssvi.T) * 100.0
        ax.plot(ssvi.T * 365.0, atm_vol, "o-", color=colors[regime],
                label=regime)
    ax.set_title("ATM annualized vol term structure")
    ax.set_xlabel("days to expiry")
    ax.set_ylabel("ATM implied vol (%)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for regime, res in results.items():
        panel = res["panel"]
        s = panel[np.isclose(panel["T"], panel["T"].min())]
        ax.scatter(s["k"], s["iv"] * 100.0, s=14, alpha=0.6,
                   color=colors[regime], label=f"{regime} (shortest expiry)")
    ax.set_title("Shortest-expiry smile: calm vs stressed")
    ax.set_xlabel("log-forward moneyness k")
    ax.set_ylabel("implied vol (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    bar_x = np.arange(2)
    rmse_vals = [comparison.loc[comparison["regime"] == r,
                                "median_slice_rmse_vol_points"].iloc[0]
                for r in ("calm", "stressed")]
    ax.bar(bar_x, rmse_vals, color=[colors["calm"], colors["stressed"]])
    ax.axhline(0.5, color="black", ls="--", lw=1.0, label="0.5 vol pt target")
    ax.set_xticks(bar_x)
    ax.set_xticklabels(["calm", "stressed"])
    ax.set_title("Median slice RMSE by regime")
    ax.set_ylabel("vol points")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 1]
    cost_vals = [comparison.loc[comparison["regime"] == r,
                                "avg_atm_relative_spread_bp"].iloc[0]
                for r in ("calm", "stressed")]
    ax.bar(bar_x, cost_vals, color=[colors["calm"], colors["stressed"]])
    ax.set_xticks(bar_x)
    ax.set_xticklabels(["calm", "stressed"])
    ax.set_title("Average ATM relative bid-ask spread by regime\n"
                 "(transaction cost proxy)")
    ax.set_ylabel("basis points of mid")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Calm (2023-2024 style) vs Stressed (2022 style) "
                 "regime comparison - SYNTHETIC DEMO DATA", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = os.path.join(output_root, "fig5_regime_comparison.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    log.info("Figure written: %s", fig_path)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Command line interface for the pipeline."""
    p = argparse.ArgumentParser(
        description="Arbitrage-free volatility surface modeling with SVI "
                    "and SSVI. Use --demo for a fully synthetic run, or "
                    "--input to point at a real option chain CSV.")
    p.add_argument("--input", type=str, default=None,
                   help="Path to a real option chain CSV. See README.md for "
                        "the required schema.")
    p.add_argument("--demo", action="store_true",
                   help="Run on clearly labeled synthetic demonstration data "
                        "(no real market data used or implied).")
    p.add_argument("--output-dir", type=str, default="outputs",
                   help="Directory for figures and CSV outputs.")
    p.add_argument("--rate", type=float, default=0.03,
                   help="Continuously compounded risk-free rate (fallback).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for the synthetic demo generator.")
    p.add_argument("--regime", type=str, default="calm",
                   choices=["calm", "stressed", "both"],
                   help="Synthetic demo regime: 'calm' (2023-2024 style, "
                        "default), 'stressed' (2022 bear-market style), or "
                        "'both' to run a full calm-vs-stressed comparison "
                        "(writes outputs/calm/, outputs/stressed/, and a "
                        "top-level regime_comparison_summary.csv / figure). "
                        "Only applies in --demo mode.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    cfg = PipelineConfig(output_dir=args.output_dir,
                         risk_free_rate=args.rate,
                         random_seed=args.seed,
                         regime=args.regime if args.regime != "both" else "calm")

    if not args.demo and args.input is None:
        # Default to demo mode when no input is supplied, so the script is
        # runnable out of the box while making the synthetic nature explicit.
        log.warning("No --input supplied; defaulting to --demo "
                    "(synthetic data).")
        args.demo = True
    if not args.demo and args.input:
        log.info("Running on real market data: %s", args.input)

    if args.demo and args.regime == "both":
        run_regime_comparison(cfg, output_root=args.output_dir)
    else:
        run_pipeline(cfg, input_csv=args.input, demo=args.demo)


if __name__ == "__main__":
    sys.exit(main())
