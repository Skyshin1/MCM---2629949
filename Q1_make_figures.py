# Q1_make_figures.py
# ------------------------------------------------------------
# Automated figure generation and data export for Q1 analysis
# Key enhancements:
# 1) AGING data: Calculate Rt (Rt = R0_mean / SOH_ratio)
# 2) New AGING relationship plots:
#    - N vs R0
#    - N vs SOH (battery ID-grouped stabilization + cross-ID median trend line)
# 3) Fixed Q1-3 TTE plot plateau issue:
#    - Root cause: Time axis truncation causing capped TTE values
#    - Solution: Use analytical formula TTE = (SOC0 * Ecap)/P (eliminates plateau)
# 4) Improved Q1-2 Model vs Measured comparison:
#    - Prioritize actual SOC/battery_level from logs for measured curve
#    - Resampling + rolling median + rolling mean smoothing
#    - Separate charging/discharging segments in visualization
#
# Requirements: pandas, numpy, matplotlib

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============== Configuration Parameters ==============
DATA_DIR = "."  # Update to your data directory (e.g., r"D:\LaterDeletee\Final_Data")
OLED_CSV = "OLED_processed_with_load_quantified.csv"
LED_CSV = "LED_processed_with_load_quantified.csv"
AGING_CSV = "AGING_processed_with_load_quantified.csv"

PLOT_SOURCE = "OLED"          # Data source for plotting: "OLED" | "LED"
WINDOW_HOURS = 2.0            # Analysis time window (hours)
V_NOM = 3.85                  # Nominal voltage (V)

LOW_Q, MID_Q, HIGH_Q = 0.10, 0.50, 0.90  # Quantile levels for load classification

INIT_SOC_LIST = [100, 80, 50, 30]        # Initial SOC values for TTE analysis
AGING_CAP_MULT = [1.00, 0.85, 0.70]      # Aging capacity multipliers

MC_N = 200                    # Monte Carlo simulation iterations
P_NOISE_CV = 0.05             # Power coefficient of variation (5%)
V_NOISE_CV = 0.02             # Voltage coefficient of variation (2%)
# ======================================================

OUT_DIR = "Q1_outputs"
FIG_DIR = os.path.join(OUT_DIR, "figs")
DATA_DIR_OUT = os.path.join(OUT_DIR, "data_used")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR_OUT, exist_ok=True)


# ------------------------------------------------------------
# Utility: Column name auto-detection
# ------------------------------------------------------------
def _pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((c for c in candidates if c in df.columns), None)


# ------------------------------------------------------------
# Data loading functions: 
# - OLED/LED require timestamp column
# - AGING may not have timestamp
# ------------------------------------------------------------
def read_timeseries_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    time_candidates = ["timestamp", "time",
                       "datetime", "date_time", "DateTime"]
    tcol = next((c for c in time_candidates if c in df.columns), None)
    if tcol is None:
        raise ValueError(f"[{path}] No timestamp column found (expected timestamp/time/datetime etc.), cannot proceed.")

    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    df = df.dropna(subset=[tcol]).copy()
    if tcol != "timestamp":
        df = df.rename(columns={tcol: "timestamp"})

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Remove duplicates to prevent erratic curve behavior
    if df["timestamp"].duplicated().any():
        df = df.groupby("timestamp", as_index=False).mean(numeric_only=True)

    return df


def read_param_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    time_candidates = ["timestamp", "time",
                       "datetime", "date_time", "DateTime"]
    tcol = next((c for c in time_candidates if c in df.columns), None)
    if tcol is not None:
        df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
        df = df.dropna(subset=[tcol]).copy()
        if tcol != "timestamp":
            df = df.rename(columns={tcol: "timestamp"})
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def add_time_axes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    t0 = df["timestamp"].iloc[0]
    df["t_sec"] = (df["timestamp"] - t0).dt.total_seconds().astype(float)
    df["dt"] = df["t_sec"].diff().fillna(0.0).clip(lower=0.0)
    return df


# ------------------------------------------------------------
# SOH related: Cycle count and capacity identification
# ------------------------------------------------------------
def get_cycle_N(df: pd.DataFrame) -> pd.Series | None:
    ncol = _pick_first_existing(df, [
        "N", "cycle", "Cycle", "cycle_index", "cycle_count", "cycles",
        "num_cycles", "n_cycles", "cycleNumber", "Cycle_Index"
    ])
    if ncol is None:
        return None
    return pd.to_numeric(df[ncol], errors="coerce")


def get_capacity_Ct(df: pd.DataFrame) -> pd.Series | None:
    ccol = _pick_first_existing(df, [
        "capacity_mah", "Capacity_mAh", "Capacity", "capacity", "cap_mah",
        "Qeff", "Q_eff", "Qeff_mAh", "Q", "Q_mAh"
    ])
    if ccol is None:
        return None
    return pd.to_numeric(df[ccol], errors="coerce")


# ------------------------------------------------------------
# 1 - k N^alpha fitting (using OLED/LED data)
# ------------------------------------------------------------
def fit_k_alpha_from_two_datasets(oled: pd.DataFrame, led: pd.DataFrame) -> tuple[float, float] | None:
    frames = []
    for df in [oled, led]:
        N = get_cycle_N(df)
        Ct = get_capacity_Ct(df)
        if N is None or Ct is None:
            continue

        tmp = pd.DataFrame({"N": N, "Ct": Ct}).dropna()
        tmp = tmp[tmp["N"] > 0].copy()
        if len(tmp) < 50:
            continue

        q0_ref = tmp.loc[tmp["N"] <= tmp["N"].quantile(0.01), "Ct"]
        q0 = float(q0_ref.median()) if len(q0_ref) else float(tmp["Ct"].max())

        ratio = tmp["Ct"] / q0
        y = 1.0 - ratio

        tmp2 = pd.DataFrame({"N": tmp["N"], "y": y})
        tmp2 = tmp2[(tmp2["y"] > 1e-6) & (tmp2["y"] < 0.5)].copy()
        if len(tmp2) < 50:
            continue

        tmp2["logN"] = np.log(tmp2["N"])
        tmp2["logy"] = np.log(tmp2["y"])
        frames.append(tmp2[["logN", "logy"]])

    if not frames:
        return None

    data = pd.concat(frames, ignore_index=True).dropna()
    if len(data) < 100:
        return None

    x = data["logN"].to_numpy()
    y = data["logy"].to_numpy()
    varx = np.var(x)
    if varx <= 1e-12:
        return None

    alpha = float(np.cov(x, y, bias=True)[0, 1] / varx)
    a = float(np.mean(y) - alpha * np.mean(x))
    k = float(np.exp(a))

    if not np.isfinite(k) or not np.isfinite(alpha) or k <= 0:
        return None
    return k, alpha


# ------------------------------------------------------------
# OLED/LED: SOH calculation method (unchanged)
# ------------------------------------------------------------
def compute_soh_percent_for_oled_led(df: pd.DataFrame, k_alpha: tuple[float, float] | None) -> pd.Series:
    N = get_cycle_N(df)
    Ct = get_capacity_Ct(df)

    if N is not None and k_alpha is not None:
        k, alpha = k_alpha
        Nn = pd.to_numeric(N, errors="coerce").fillna(0.0).clip(lower=0.0)
        ratio = 1.0 - k * (Nn.to_numpy() ** alpha)
        ratio = np.clip(ratio, 0.0, 1.0)
        return pd.Series(100.0 * ratio, index=df.index, name="SOH_calc")

    if Ct is not None:
        Ct_num = pd.to_numeric(Ct, errors="coerce")
        C0 = float(Ct_num.max(skipna=True)) if Ct_num.notna().any() else np.nan
        ratio = (Ct_num / C0).clip(0, 1)
        ratio = ratio.interpolate(limit=600).ffill().bfill()
        return (100.0 * ratio).rename("SOH_calc")

    print("[WARN] Cannot calculate SOH for OLED/LED: Missing N and capacity Ct columns. Defaulting to 100%.")
    return pd.Series([100.0] * len(df), index=df.index, name="SOH_calc")


# ------------------------------------------------------------
# Enhanced SOC processing for Q1-2:
# - Extract actual measured SOC from logs
# - Smoothing with resampling and rolling statistics
# - Charge/discharge segmentation
# ------------------------------------------------------------
def get_measured_soc_percent(df: pd.DataFrame) -> pd.Series:
    """
    Extract measured SOC from log data (prioritize actual values over calculated SOH):
    - OLED: Typically SOC_percent or SOC
    - LED: Typically battery_level
    """
    col = _pick_first_existing(
        df, ["SOC_percent", "SOC", "battery_level", "BatteryLevel", "level"])
    if col is None:
        # Fallback: return 100% if no SOC column found
        return pd.Series([100.0] * len(df), index=df.index, name="SOC_meas")

    s = pd.to_numeric(df[col], errors="coerce")

    # Detect value range (0~1 or 0~100)
    q99 = s.quantile(0.99)
    if pd.notna(q99) and q99 <= 1.5:
        s = s * 100.0

    s = s.interpolate(limit=600).ffill().bfill().clip(0, 100)
    return s.rename("SOC_meas")


def smooth_soc_timeseries(
    seg: pd.DataFrame,
    soc_col: str = "SOC_meas",
    resample_sec: int = 60,
    med_win: int = 9,
    mean_win: int = 9,
    enforce_monotone_discharge: bool = False,
) -> pd.DataFrame:
    """
    Reduce SOC timeseries noise with:
    - Time resampling (default 60s)
    - Rolling median + rolling mean smoothing
    - Optional monotonic discharge constraint (enable only for discharge-only data)
    """
    if "timestamp" not in seg.columns:
        raise ValueError("smooth_soc_timeseries: Missing timestamp column")
    if soc_col not in seg.columns:
        raise ValueError(f"smooth_soc_timeseries: Missing {soc_col} column")

    df = seg[["timestamp", soc_col]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df[soc_col] = pd.to_numeric(df[soc_col], errors="coerce").clip(0, 100)
    df = df.dropna(subset=[soc_col]).copy()

    ts = df.set_index("timestamp")
    ts = ts.resample(f"{int(resample_sec)}S").mean(
    ).interpolate(limit=20).ffill().bfill()

    if med_win and med_win > 1:
        ts[soc_col] = ts[soc_col].rolling(
            int(med_win), center=True, min_periods=1).median()
    if mean_win and mean_win > 1:
        ts[soc_col] = ts[soc_col].rolling(
            int(mean_win), center=True, min_periods=1).mean()

    soc_s = ts[soc_col].to_numpy()

    if enforce_monotone_discharge:
        soc_s = np.minimum.accumulate(soc_s)

    out = ts.reset_index()
    out = out.rename(columns={soc_col: "SOC_meas_smooth"})
    out["SOC_meas_smooth"] = soc_s
    return out


def split_charge_discharge(soc_smooth: pd.Series, eps: float = 0.02) -> tuple[pd.Series, pd.Series]:
    """
    Segment SOC timeseries into charging/discharging periods based on rate of change:
    - diff > +eps -> charging
    - diff < -eps -> discharging
    - Remaining periods inherit previous state
    """
    s = soc_smooth.to_numpy(dtype=float)
    d = np.diff(s, prepend=s[0])
    state = np.zeros_like(s)  # -1 discharge, +1 charge, 0 unknown
    state[d > eps] = 1
    state[d < -eps] = -1

    # Fill unknown states with previous state
    for i in range(1, len(state)):
        if state[i] == 0:
            state[i] = state[i-1]
    # Default to discharge for initial unknown state
    if len(state) and state[0] == 0:
        state[:] = np.where(state == 0, -1, state)

    discharge = np.where(state == -1, s, np.nan)
    charge = np.where(state == 1, s, np.nan)
    return pd.Series(discharge, index=soc_smooth.index), pd.Series(charge, index=soc_smooth.index)


# ------------------------------------------------------------
# AGING data processing:
# - Direct SOH reading
# - Cycle count inference when missing
# ------------------------------------------------------------
def get_soh_from_aging(df_aging: pd.DataFrame) -> pd.Series:
    if "SOH" not in df_aging.columns:
        raise ValueError("SOH column not found in AGING dataset (required for direct SOH reading).")

    soh = pd.to_numeric(df_aging["SOH"], errors="coerce")
    q99 = soh.quantile(0.99)
    if pd.notna(q99) and q99 <= 1.5:
        soh = soh * 100.0
    soh = soh.interpolate(limit=600).ffill().bfill().clip(0, 100)
    return soh.rename("SOH")


def infer_N_for_aging(df_aging: pd.DataFrame, k_alpha: tuple[float, float] | None) -> pd.Series:
    N = get_cycle_N(df_aging)
    if N is not None and N.notna().any():
        return N.rename("N")

    soh_ratio = get_soh_from_aging(df_aging) / 100.0

    if k_alpha is None:
        print("[WARN] Missing cycle count N in AGING data and cannot obtain (k,alpha) parameters. Using row index as N_hat (relative cycle order only).")
        return pd.Series(np.arange(len(df_aging), dtype=float), index=df_aging.index, name="N_hat")

    k, alpha = k_alpha
    y = (1.0 - soh_ratio).clip(lower=1e-6, upper=0.999999)
    N_hat = (y / k) ** (1.0 / alpha)
    N_hat = N_hat.replace(
        [np.inf, -np.inf], np.nan).interpolate(limit=600).ffill().bfill()
    return N_hat.rename("N_hat")


# ------------------------------------------------------------
# AGING data: Calculate Rt = R0_mean / SOH_ratio
# ------------------------------------------------------------
def add_Rt_to_aging(df_aging: pd.DataFrame) -> pd.DataFrame:
    df = df_aging.copy()

    soh_pct = get_soh_from_aging(df)
    soh_ratio = (soh_pct / 100.0).clip(lower=1e-6)

    r0_col = _pick_first_existing(
        df, ["R0_mean", "R0Mean", "R0_avg", "R0_average", "R0", "R_0"])
    if r0_col is None:
        raise ValueError("Cannot find R0_mean (or equivalent) column in AGING data for Rt calculation.")

    r0 = pd.to_numeric(df[r0_col], errors="coerce")
    df["R0_mean_used"] = r0

    df["Rt"] = (r0 / soh_ratio).replace([np.inf, -np.inf], np.nan)
    df["Rt"] = df["Rt"].interpolate(limit=600).ffill().bfill()
    return df


# ------------------------------------------------------------
# AGING data: Battery ID detection + stabilized N-SOH curves
# ------------------------------------------------------------
def infer_battery_id_col(df: pd.DataFrame) -> str | None:
    # Priority battery ID columns
    priority = ["battery_id", "Battery_ID", "cell_id",
                "Cell_ID", "uid", "UID", "test_id", "Test_ID"]
    for c in priority:
        if c in df.columns:
            return c
    # Loose matching for alternative naming
    for c in df.columns:
        if any(k in c.lower() for k in ["battery", "cell", "uid", "test", "serial", "pack", "device"]):
            return c
    return None


def enforce_monotone_decreasing(y: np.ndarray) -> np.ndarray:
    """Monotonic degradation constraint: y[i] = min(y[i], y[i-1])"""
    out = y.copy()
    for i in range(1, len(out)):
        if out[i] > out[i-1]:
            out[i] = out[i-1]
    return out


def plot_aging_N_vs_Rt(aging: pd.DataFrame):
    """
    Plot AGING data: Rt vs N
    Rt pre-calculated in add_Rt_to_aging():
      Rt = R0_mean / SOH_ratio
    """
    if "N_used" not in aging.columns:
        print("[WARN] Missing N_used in aging data, skipping N-Rt plot")
        return
    if "Rt" not in aging.columns:
        print("[WARN] Missing Rt in aging data, skipping N-Rt plot")
        return

    tmp = pd.DataFrame({
        "N": pd.to_numeric(aging["N_used"], errors="coerce"),
        "Rt": pd.to_numeric(aging["Rt"], errors="coerce"),
    }).dropna()
    if len(tmp) < 10:
        print("[WARN] Insufficient valid points for N-Rt plot, skipping")
        return

    tmp = tmp[tmp["N"] >= 0].sort_values("N")

    plt.figure()
    plt.scatter(tmp["N"], tmp["Rt"], s=14)
    plt.xlabel("Cycle count N (AGING)")
    plt.ylabel("Rt = R0_mean / SOH_ratio")
    plt.title("AGING: Rt vs N")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "Q1_extra_AGING_N_vs_Rt.png"), dpi=300)
    plt.close()


def plot_aging_N_vs_SOH_stabilized(aging: pd.DataFrame):
    """
    Generate stabilized N-SOH curves for AGING data:
    - Group by battery_id/uid/test_id when available
    - Per-ID: bin median + monotonic degradation constraint + light smoothing
    - Cross-ID: median trend line + monotonic degradation + smoothing
    """
    if "N_used" not in aging.columns or "SOH" not in aging.columns:
        print("[WARN] Missing N_used or SOH in aging data, skipping N-SOH plot")
        return

    id_col = infer_battery_id_col(aging)

    BIN_SIZE = 100          # Increase for more stabilization: 200
    ROLL_WIN = 7            # Increase for more smoothing: 9/11
    MIN_POINTS_PER_ID = 10

    df = aging.copy()
    df["N_used"] = pd.to_numeric(df["N_used"], errors="coerce")
    df["SOH"] = pd.to_numeric(df["SOH"], errors="coerce")
    df = df.dropna(subset=["N_used", "SOH"])
    df = df[df["N_used"] >= 0].copy()

    # Global binning + monotonic constraint when no ID column
    if id_col is None:
        tmp = df.copy()
        tmp["N_bin"] = (tmp["N_used"] // BIN_SIZE) * BIN_SIZE
        agg = tmp.groupby("N_bin", as_index=False)[
            "SOH"].median().sort_values("N_bin")

        y = agg["SOH"].to_numpy()
        if ROLL_WIN > 1 and len(y) >= ROLL_WIN:
            y = pd.Series(y).rolling(ROLL_WIN, center=True,
                                     min_periods=1).median().to_numpy()
        y = enforce_monotone_decreasing(y)
        agg["SOH_curve"] = y

        plt.figure()
        plt.plot(agg["N_bin"], agg["SOH_curve"], linewidth=2.5)
        plt.xlabel("Cycle count N (binned)")
        plt.ylabel("SOH (%)")
        plt.title("AGING: SOH vs N (global binned, stabilized)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(
            FIG_DIR, "Q1_extra_AGING_N_vs_SOH.png"), dpi=300)
        plt.close()

        agg.to_csv(os.path.join(
            DATA_DIR_OUT, "Q1_AGING_SOH_vs_N_stabilized.csv"), index=False)
        return

    # Process by battery ID when available
    curves = []
    plt.figure()

    for gid, g in df.groupby(id_col):
        g = g.dropna(subset=["N_used", "SOH"]).copy()
        if len(g) < MIN_POINTS_PER_ID:
            continue

        g = g.sort_values("N_used")
        g = g.groupby("N_used", as_index=False)["SOH"].mean()

        g["N_bin"] = (g["N_used"] // BIN_SIZE) * BIN_SIZE
        gb = g.groupby("N_bin", as_index=False)[
            "SOH"].median().sort_values("N_bin")

        y = gb["SOH"].to_numpy(dtype=float)

        # Light smoothing followed by monotonic constraint
        if ROLL_WIN > 1 and len(y) >= ROLL_WIN:
            y = pd.Series(y).rolling(ROLL_WIN, center=True,
                                     min_periods=1).median().to_numpy()
        y = enforce_monotone_decreasing(y)

        gb["SOH_curve"] = y
        gb["ID"] = str(gid)
        curves.append(gb[["ID", "N_bin", "SOH_curve"]])

        # Plot individual battery curves with low opacity
        plt.plot(gb["N_bin"], gb["SOH_curve"], linewidth=1.0, alpha=0.20)

    if not curves:
        print("[WARN] Insufficient battery ID curves with valid points for stabilized N-SOH plot")
        plt.close()
        return

    curves_df = pd.concat(curves, ignore_index=True)

    # Cross-battery median trend line
    med = curves_df.groupby("N_bin", as_index=False)[
        "SOH_curve"].median().sort_values("N_bin")
    y_med = med["SOH_curve"].to_numpy(dtype=float)

    # Additional smoothing + monotonic constraint for main trend line
    if ROLL_WIN > 1 and len(y_med) >= ROLL_WIN:
        y_med = pd.Series(y_med).rolling(
            ROLL_WIN, center=True, min_periods=1).mean().to_numpy()
    y_med = enforce_monotone_decreasing(y_med)
    med["SOH_med_curve"] = y_med

    plt.plot(med["N_bin"], med["SOH_med_curve"], linewidth=3.2,
             alpha=1.0, label="Median across batteries")

    plt.xlabel("Cycle count N (binned)")
    plt.ylabel("SOH (%)")
    plt.title(f"AGING: SOH vs N (grouped by {id_col}, stabilized)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "Q1_extra_AGING_N_vs_SOH.png"), dpi=300)
    plt.close()

    # Export data for external analysis (e.g., Prism)
    curves_df.to_csv(os.path.join(
        DATA_DIR_OUT, "Q1_AGING_SOH_vs_N_byBattery_stabilized.csv"), index=False)
    med.to_csv(os.path.join(
        DATA_DIR_OUT, "Q1_AGING_SOH_vs_N_median_curve.csv"), index=False)


# ------------------------------------------------------------
# Data loading pipeline for all three datasets
# ------------------------------------------------------------
def load_three_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[float, float] | None]:
    os.chdir(DATA_DIR)

    oled = read_timeseries_csv(OLED_CSV)
    led = read_timeseries_csv(LED_CSV)
    aging = read_param_csv(AGING_CSV)

    oled["device_type"] = "OLED"
    led["device_type"] = "LED"

    oled = add_time_axes(oled)
    led = add_time_axes(led)
    if "timestamp" in aging.columns:
        aging = add_time_axes(aging)

    k_alpha = fit_k_alpha_from_two_datasets(oled, led)
    if k_alpha is None:
        print("[WARN] Cannot fit k, alpha from OLED/LED data (missing N or capacity columns). Using Ct/C0 fallback for OLED/LED SOH calculation.")
    else:
        print(f"[INFO] Fitted k, alpha = {k_alpha}")

    # Calculate SOH for OLED/LED (original logic preserved)
    oled["SOH_calc"] = compute_soh_percent_for_oled_led(oled, k_alpha)
    led["SOH_calc"] = compute_soh_percent_for_oled_led(led,  k_alpha)

    # Extract actual measured SOC for Q1-2 comparison
    oled["SOC_meas"] = get_measured_soc_percent(oled)
    led["SOC_meas"] = get_measured_soc_percent(led)

    # Process AGING data: direct SOH reading + cycle count inference + Rt calculation
    aging["SOH"] = get_soh_from_aging(aging)
    aging["N_used"] = infer_N_for_aging(aging, k_alpha)
    aging = add_Rt_to_aging(aging)

    return oled, led, aging, k_alpha


# ------------------------------------------------------------
# Continuous-time SOC integration from power measurements
# ------------------------------------------------------------
def energy_capacity_J(capacity_mah: float, v_nom: float) -> float:
    return (capacity_mah / 1000.0) * v_nom * 3600.0


def integrate_soc_from_power(t_sec, P_mW, soc0, cap_mah, v_nom, kappa_T=None):
    dt = np.diff(t_sec, prepend=t_sec[0])
    dt = np.clip(dt, 0.0, None)
    E_cap = energy_capacity_J(cap_mah, v_nom)

    if kappa_T is not None:
        k = np.clip(kappa_T.astype(float), 0.5, 1.2)
    else:
        k = 1.0

    E = np.empty_like(P_mW, dtype=float)
    E[0] = (soc0 / 100.0) * E_cap

    for i in range(1, len(P_mW)):
        P_W = float(P_mW[i]) / 1000.0
        ki = float(k[i]) if isinstance(k, np.ndarray) else float(k)
        E[i] = max(0.0, E[i-1] - P_W * dt[i] * (1.0 / ki))

    SOC = 100.0 * E / E_cap
    return np.clip(SOC, 0.0, 100.0)


# ------------------------------------------------------------
# Scenario construction: low/medium/high load profiles
# ------------------------------------------------------------
def infer_screen_coef(df: pd.DataFrame) -> float:
    if ("P_scr_W" not in df.columns) or ("screen_L_nits" not in df.columns):
        raise ValueError("Missing P_scr_W or screen_L_nits columns for 300/600nit scenario construction.")
    ratio = (df["P_scr_W"] / df["screen_L_nits"]
             ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(ratio) == 0:
        raise ValueError("Cannot infer screen power coefficient from P_scr_W/screen_L_nits ratio.")
    return float(ratio.median())


def make_typical_power_levels(df: pd.DataFrame) -> dict:
    required = ["P_base_mW", "P_cpu_mW", "P_net_mW", "P_bg_mW", "P_gps_mW"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing power breakdown columns: {missing}")

    def q(col: str, quant: float) -> float:
        return float(df[col].quantile(quant))

    return {
        "P_base_mW": q("P_base_mW", MID_Q),
        "P_cpu_low": q("P_cpu_mW", LOW_Q),
        "P_cpu_mid": q("P_cpu_mW", MID_Q),
        "P_cpu_high": q("P_cpu_mW", HIGH_Q),
        "P_net_low": q("P_net_mW", LOW_Q),
        "P_net_mid": q("P_net_mW", MID_Q),
        "P_net_high": q("P_net_mW", HIGH_Q),
        "P_bg_low":  q("P_bg_mW", LOW_Q),
        "P_bg_mid":  q("P_bg_mW", MID_Q),
        "P_bg_high": q("P_bg_mW", HIGH_Q),
        "P_gps_mid": q("P_gps_mW", MID_Q),
        "P_gps_high": q("P_gps_mW", HIGH_Q),
    }


def build_scenarios(df: pd.DataFrame, window_hours: float) -> pd.DataFrame:
    k_scr = infer_screen_coef(df)
    levels = make_typical_power_levels(df)

    horizon_sec = window_hours * 3600.0
    sub = df[df["t_sec"] <= horizon_sec].copy()
    if len(sub) < 300:
        sub = df.copy()
    t = sub["t_sec"].to_numpy()

    def scr_mW(nits: float) -> float:
        return (k_scr * nits) * 1000.0

    P_scr_low = scr_mW(0.0)
    P_scr_mid = scr_mW(300.0)
    P_scr_high = scr_mW(600.0)

    scen = pd.DataFrame({"t_sec": t})
    scen["P_low_mW"] = levels["P_base_mW"] + P_scr_low + \
        levels["P_cpu_low"] + levels["P_net_low"] + levels["P_bg_low"]
    scen["P_mid_mW"] = levels["P_base_mW"] + P_scr_mid + levels["P_cpu_mid"] + \
        levels["P_net_mid"] + levels["P_bg_mid"] + levels["P_gps_mid"]
    scen["P_high_mW"] = levels["P_base_mW"] + P_scr_high + levels["P_cpu_high"] + \
        levels["P_net_high"] + levels["P_bg_high"] + \
        max(levels["P_gps_high"], levels["P_gps_mid"])

    cap_mah = float(df["capacity_mah"].median()
                    ) if "capacity_mah" in df.columns else 4000.0
    kappa_T = sub["kappa_T"].to_numpy() if "kappa_T" in sub.columns else None

    scen["SOC_low"] = integrate_soc_from_power(
        t, scen["P_low_mW"].to_numpy(), 100, cap_mah, V_NOM, kappa_T)
    scen["SOC_mid"] = integrate_soc_from_power(
        t, scen["P_mid_mW"].to_numpy(), 100, cap_mah, V_NOM, kappa_T)
    scen["SOC_high"] = integrate_soc_from_power(
        t, scen["P_high_mW"].to_numpy(), 100, cap_mah, V_NOM, kappa_T)
    return scen


# ------------------------------------------------------------
# Model vs Measured comparison (enhanced with smoothing and segmentation)
# ------------------------------------------------------------
def pick_real_segment(df: pd.DataFrame, window_hours: float) -> pd.DataFrame:
    horizon_sec = window_hours * 3600.0
    seg = df[df["t_sec"] <= horizon_sec].copy()
    if len(seg) < 300:
        seg = df.copy()

    if "P_pred_mW" not in seg.columns:
        raise ValueError("Missing P_pred_mW (model total power prediction) for Model vs Measured comparison.")

    t = seg["t_sec"].to_numpy()
    cap_mah = float(seg["capacity_mah"].median()
                    ) if "capacity_mah" in seg.columns else 4000.0
    kappa_T = seg["kappa_T"].to_numpy() if "kappa_T" in seg.columns else None

    # Use actual measured SOC starting point for model initialization
    soc0 = float(pd.to_numeric(seg["SOC_meas"], errors="coerce").iloc[0])
    seg["SOC_model"] = integrate_soc_from_power(
        t, seg["P_pred_mW"].to_numpy(), soc0, cap_mah, V_NOM, kappa_T)

    # Smooth measured SOC (no monotonic constraint for charge/discharge flexibility)
    sm = smooth_soc_timeseries(
        seg, soc_col="SOC_meas",
        resample_sec=60,
        med_win=9,
        mean_win=9,
        enforce_monotone_discharge=False
    )

    # Merge smoothed SOC back to original segment data
    seg = seg.copy()
    seg["timestamp"] = pd.to_datetime(seg["timestamp"], errors="coerce")
    sm["timestamp"] = pd.to_datetime(sm["timestamp"], errors="coerce")
    seg = seg.merge(sm[["timestamp", "SOC_meas_smooth"]],
                    on="timestamp", how="left")
    seg["SOC_meas_smooth"] = seg["SOC_meas_smooth"].interpolate(
        limit=200).ffill().bfill()

    # Segment into charging/discharging periods
    dis, chg = split_charge_discharge(seg["SOC_meas_smooth"], eps=0.02)
    seg["SOC_meas_discharge"] = dis
    seg["SOC_meas_charge"] = chg

    return seg


# ------------------------------------------------------------
# Fixed TTE calculation using analytical formula (eliminates plateau)
# ------------------------------------------------------------
def tte_hours_analytic(P_mW: float, init_soc: float, cap_mah: float, v_nom: float) -> float:
    P_W = max(1e-9, float(P_mW) / 1000.0)
    E_cap = energy_capacity_J(cap_mah, v_nom)
    E0 = (float(init_soc) / 100.0) * E_cap
    return float(E0 / P_W / 3600.0)


def build_tte_table(scen: pd.DataFrame, cap_mah: float) -> pd.DataFrame:
    P_low = float(pd.to_numeric(scen["P_low_mW"],  errors="coerce").median())
    P_mid = float(pd.to_numeric(scen["P_mid_mW"],  errors="coerce").median())
    P_high = float(pd.to_numeric(scen["P_high_mW"], errors="coerce").median())

    out_rows = []
    for init_soc in INIT_SOC_LIST:
        for aging_mult in AGING_CAP_MULT:
            eff_cap = cap_mah * aging_mult
            out_rows.append({"load": "low",  "init_soc": init_soc, "aging_capacity_mult": aging_mult,
                             "TTE_hours": tte_hours_analytic(P_low,  init_soc, eff_cap, V_NOM)})
            out_rows.append({"load": "mid",  "init_soc": init_soc, "aging_capacity_mult": aging_mult,
                             "TTE_hours": tte_hours_analytic(P_mid,  init_soc, eff_cap, V_NOM)})
            out_rows.append({"load": "high", "init_soc": init_soc, "aging_capacity_mult": aging_mult,
                             "TTE_hours": tte_hours_analytic(P_high, init_soc, eff_cap, V_NOM)})
    return pd.DataFrame(out_rows)


def mc_tte_uncertainty(P_mW: float, init_soc: float, cap_mah: float, aging_mult: float):
    ttes = []
    for _ in range(MC_N):
        v = V_NOM * (1.0 + np.random.normal(0.0, V_NOISE_CV))
        P = float(P_mW) * (1.0 + np.random.normal(0.0, P_NOISE_CV))
        P = max(1e-9, P)
        eff_cap = cap_mah * aging_mult
        ttes.append(tte_hours_analytic(P, init_soc, eff_cap, v))

    ttes = np.array(ttes, dtype=float)
    return float(np.percentile(ttes, 2.5)), float(np.percentile(ttes, 50)), float(np.percentile(ttes, 97.5))


# ------------------------------------------------------------
# Visualization functions
# ------------------------------------------------------------
def plot_soc_scenarios(scen: pd.DataFrame):
    x = scen["t_sec"] / 3600.0
    plt.figure()
    plt.plot(x, scen["SOC_low"],  label="Low load (screen off, GPS off)")
    plt.plot(x, scen["SOC_mid"],  label="Mid load (300 nit, GPS on)")
    plt.plot(x, scen["SOC_high"], label="High load (600 nit, multi-task)")
    plt.xlabel("Time (hours)")
    plt.ylabel("SOC (%)")
    plt.title("Q1-1 SOC vs Time under Low/Mid/High Load")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "Q1_1_SOC_low_mid_high.png"), dpi=300)
    plt.close()


def plot_model_vs_measured(seg: pd.DataFrame):
    x = seg["t_sec"] / 3600.0

    plt.figure()
    # Separate charging/discharging visualization for clarity
    if "SOC_meas_discharge" in seg.columns and "SOC_meas_charge" in seg.columns:
        plt.plot(x, seg["SOC_meas_discharge"],
                 label="Measured (smoothed) - Discharge")
        plt.plot(x, seg["SOC_meas_charge"],
                 label="Measured (smoothed) - Charge")
    else:
        plt.plot(x, seg.get("SOC_meas_smooth",
                 seg["SOC_meas"]), label="Measured (smoothed)")

    plt.plot(x, seg["SOC_model"], label="Model-predicted (integrate P_pred)")
    plt.xlabel("Time (hours)")
    plt.ylabel("SOC (%)")
    plt.title("Q1-2 Model vs Measured (charge/discharge separated)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "Q1_2_model_vs_measured.png"), dpi=300)
    plt.close()


def enforce_monotone_increasing(y: np.ndarray) -> np.ndarray:
    """Monotonic increasing constraint: y[i] = max(y[i], y[i-1])"""
    out = y.copy()
    for i in range(1, len(out)):
        if out[i] < out[i-1]:
            out[i] = out[i-1]
    return out


def plot_aging_N_vs_Rt_stabilized(aging: pd.DataFrame):
    """
    Generate stabilized Rt vs N curves for AGING data:
    - Per battery ID: N bin median + rolling smoothing
    - Visualization: low-opacity per-battery lines + bold cross-battery median trend
    """
    if "N_used" not in aging.columns or "Rt" not in aging.columns:
        print("[WARN] Missing N_used or Rt in aging data, skipping stabilized N-Rt plot")
        return

    id_col = infer_battery_id_col(aging)

    # Adjustable parameters for stabilization level
    BIN_SIZE = 100          # Larger = smoother: 100/200/300
    ROLL_WIN = 7            # Smoothing window: 7/9/11
    MIN_POINTS_PER_ID = 10  # Minimum valid points per battery ID

    df = aging.copy()
    df["N_used"] = pd.to_numeric(df["N_used"], errors="coerce")
    df["Rt"] = pd.to_numeric(df["Rt"], errors="coerce")
    df = df.dropna(subset=["N_used", "Rt"])
    df = df[df["N_used"] >= 0].copy()

    # Global binning + smoothing when no ID column (more stable than scatter plot)
    if id_col is None:
        tmp = df.copy()
        tmp["N_bin"] = (tmp["N_used"] // BIN_SIZE) * BIN_SIZE
        agg = tmp.groupby("N_bin", as_index=False)[
            "Rt"].median().sort_values("N_bin")

        y = agg["Rt"].to_numpy(dtype=float)
        if ROLL_WIN > 1 and len(y) >= ROLL_WIN:
            y = pd.Series(y).rolling(ROLL_WIN, center=True,
                                     min_periods=1).median().to_numpy()

        # Optional monotonic increasing constraint for Rt (more publication-ready)
        y = enforce_monotone_increasing(y)

        agg["Rt_curve"] = y

        plt.figure()
        plt.plot(agg["N_bin"], agg["Rt_curve"], linewidth=2.5)
        plt.xlabel("Cycle count N (binned)")
        plt.ylabel("Rt = R0_mean / SOH_ratio")
        plt.title("AGING: Rt vs N (global binned, stabilized)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(
            FIG_DIR, "Q1_extra_AGING_N_vs_Rt.png"), dpi=300)
        plt.close()

        agg.to_csv(os.path.join(
            DATA_DIR_OUT, "Q1_AGING_Rt_vs_N_stabilized.csv"), index=False)
        return

    # ID-based processing when available
    curves = []
    plt.figure()

    for gid, g in df.groupby(id_col):
        g = g.dropna(subset=["N_used", "Rt"]).copy()
        if len(g) < MIN_POINTS_PER_ID:
            continue

        # Sort and deduplicate within each battery ID
        g = g.sort_values("N_used")
        g = g.groupby("N_used", as_index=False)["Rt"].median()

        # Bin aggregation
        g["N_bin"] = (g["N_used"] // BIN_SIZE) * BIN_SIZE
        gb = g.groupby("N_bin", as_index=False)[
            "Rt"].median().sort_values("N_bin")

        y = gb["Rt"].to_numpy(dtype=float)
        if ROLL_WIN > 1 and len(y) >= ROLL_WIN:
            y = pd.Series(y).rolling(ROLL_WIN, center=True,
                                     min_periods=1).median().to_numpy()

        # Rt typically increases with aging: optional monotonic constraint
        y = enforce_monotone_increasing(y)

        gb["Rt_curve"] = y
        gb["ID"] = str(gid)
        curves.append(gb[["ID", "N_bin", "Rt_curve"]])

        # Plot individual battery curves with low opacity
        plt.plot(gb["N_bin"], gb["Rt_curve"], linewidth=1.0, alpha=0.20)

    if not curves:
        print("[WARN] Insufficient battery ID curves with valid points for stabilized N-Rt plot")
        plt.close()
        return

    curves_df = pd.concat(curves, ignore_index=True)

    # Cross-battery median trend line
    med = curves_df.groupby("N_bin", as_index=False)[
        "Rt_curve"].median().sort_values("N_bin")
    y_med = med["Rt_curve"].to_numpy(dtype=float)

    # Additional smoothing + monotonic constraint for main trend
    if ROLL_WIN > 1 and len(y_med) >= ROLL_WIN:
        y_med = pd.Series(y_med).rolling(
            ROLL_WIN, center=True, min_periods=1).mean().to_numpy()
    y_med = enforce_monotone_increasing(y_med)
    med["Rt_med_curve"] = y_med

    plt.plot(med["N_bin"], med["Rt_med_curve"], linewidth=3.2,
             alpha=1.0, label="Median across batteries")

    plt.xlabel("Cycle count N (binned)")
    plt.ylabel("Rt = R0_mean / SOH_ratio")
    plt.title(f"AGING: Rt vs N (grouped by {id_col}, stabilized)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "Q1_extra_AGING_N_vs_Rt.png"), dpi=300)
    plt.close()

    # Export data for external analysis
    curves_df.to_csv(os.path.join(
        DATA_DIR_OUT, "Q1_AGING_Rt_vs_N_byBattery_stabilized.csv"), index=False)
    med.to_csv(os.path.join(
        DATA_DIR_OUT, "Q1_AGING_Rt_vs_N_median_curve.csv"), index=False)


def plot_tte_table(tte: pd.DataFrame):
    plt.figure()
    for aging_mult in sorted(tte["aging_capacity_mult"].unique(), reverse=True):
        sub = tte[tte["aging_capacity_mult"] == aging_mult]
        for load in ["low", "mid", "high"]:
            s2 = sub[sub["load"] == load].sort_values("init_soc")
            plt.plot(s2["init_soc"], s2["TTE_hours"], marker="o",
                     label=f"{load}, aging={aging_mult:.2f}")
    plt.xlabel("Initial SOC (%)")
    plt.ylabel("Time-to-empty (hours)")
    plt.title("Q1-3 TTE vs Initial SOC (Load × Aging)")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(
        FIG_DIR, "Q1_3_TTE_vs_initSOC_load_aging.png"), dpi=300)
    plt.close()


def plot_tte_uncertainty_bars(scen: pd.DataFrame, cap_mah: float):
    init_soc = 80
    aging_mult = 0.85

    P_low = float(pd.to_numeric(scen["P_low_mW"],  errors="coerce").median())
    P_mid = float(pd.to_numeric(scen["P_mid_mW"],  errors="coerce").median())
    P_high = float(pd.to_numeric(scen["P_high_mW"], errors="coerce").median())

    results = []
    for load, Pval in [("low", P_low), ("mid", P_mid), ("high", P_high)]:
        lo, med, hi = mc_tte_uncertainty(Pval, init_soc, cap_mah, aging_mult)
        results.append((load, lo, med, hi))

    labels = [r[0] for r in results]
    meds = np.array([r[2] for r in results])
    err_low = meds - np.array([r[1] for r in results])
    err_high = np.array([r[3] for r in results]) - meds

    plt.figure()
    x = np.arange(len(labels))
    plt.bar(x, meds)
    plt.errorbar(x, meds, yerr=[err_low, err_high], fmt="none", capsize=6)
    plt.xticks(x, labels)
    plt.ylabel("TTE (hours)")
    plt.title("Q1-Extra Uncertainty of TTE (95% CI), init=80%, aging=0.85")
    plt.tight_layout()
    plt.savefig(os.path.join(
        FIG_DIR, "Q1_extra_TTE_uncertainty_CI.png"), dpi=300)
    plt.close()


# ------------------------------------------------------------
# Export all data used in visualizations
# ------------------------------------------------------------
def export_data_used(seg: pd.DataFrame, scen: pd.DataFrame, tte: pd.DataFrame, aging: pd.DataFrame):
    keep_seg = [c for c in [
        "timestamp", "t_sec",
        "SOC_meas", "SOC_meas_smooth", "SOC_meas_discharge", "SOC_meas_charge",
        "SOC_model",
        "capacity_mah", "temperature", "kappa_T",
        "P_base_mW", "P_scr_W", "screen_L_nits",
        "P_cpu_mW", "P_net_mW", "P_bg_mW", "P_gps_mW", "P_pred_mW",
        "device_type",
        "SOH_calc"
    ] if c in seg.columns]

    seg[keep_seg].to_csv(os.path.join(
        DATA_DIR_OUT, "Q1_used_real_segment.csv"), index=False)
    scen.to_csv(os.path.join(
        DATA_DIR_OUT, "Q1_used_scenarios_SOC_power.csv"), index=False)
    tte.to_csv(os.path.join(DATA_DIR_OUT, "Q1_used_TTE_table.csv"), index=False)

    aging_keep = [c for c in ["N_used", "SOH",
                              "R0_mean_used", "Rt"] if c in aging.columns]
    # Include battery ID column for external analysis (e.g., Prism)
    id_col = infer_battery_id_col(aging)
    if id_col is not None and id_col not in aging_keep:
        aging_keep = [id_col] + aging_keep
    aging[aging_keep].to_csv(os.path.join(
        DATA_DIR_OUT, "Q1_used_AGING_with_Rt.csv"), index=False)

    manifest = {
        "Q1_used_real_segment.csv": keep_seg,
        "Q1_used_scenarios_SOC_power.csv": list(scen.columns),
        "Q1_used_TTE_table.csv": list(tte.columns),
        "Q1_used_AGING_with_Rt.csv": aging_keep,
    }
    pd.Series(manifest).to_json(
        os.path.join(DATA_DIR_OUT, "Q1_used_columns_manifest.json"),
        force_ascii=False, indent=2
    )


# ------------------------------------------------------------
# Main execution pipeline
# ------------------------------------------------------------
def main():
    oled, led, aging, k_alpha = load_three_tables()

    df = oled if PLOT_SOURCE.upper(
    ) == "OLED" else led if PLOT_SOURCE.upper() == "LED" else None
    if df is None:
        raise ValueError("PLOT_SOURCE must be either OLED or LED")

    # Generate load scenario plots
    scen = build_scenarios(df, WINDOW_HOURS)
    plot_soc_scenarios(scen)

    # Generate model vs measured comparison
    seg = pick_real_segment(df, WINDOW_HOURS)
    plot_model_vs_measured(seg)

    # Calculate baseline capacity for TTE analysis
    cap_mah = float(df["capacity_mah"].median()
                    ) if "capacity_mah" in df.columns else 4000.0

    # Generate TTE analysis and uncertainty plots
    tte = build_tte_table(scen, cap_mah)
    plot_tte_table(tte)
    plot_tte_uncertainty_bars(scen, cap_mah)

    # Generate AGING relationship plots (N-Rt and N-SOH)
    plot_aging_N_vs_Rt_stabilized(aging)
    plot_aging_N_vs_SOH_stabilized(aging)

    # Export all used data for traceability
    export_data_used(seg, scen, tte, aging)

    print("Analysis completed successfully.")
    print(f"Figures saved to: {FIG_DIR}")
    print(f"Analysis data exported to: {DATA_DIR_OUT}")
    print(f"Plot data source: {PLOT_SOURCE}")
    if k_alpha is not None:
        print(f"Fitted degradation parameters k, alpha = {k_alpha}")


if __name__ == "__main__":
    main()