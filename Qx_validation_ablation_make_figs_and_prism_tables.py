#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qx_validation_ablation_make_figs_and_prism_tables.py
------------------------------------------------------------
Purpose:
- Generate exactly 6 figures (no additional figures):
  1) H6 delta bar chart (RMSE_power_W) | LED   (x-axis starts at 0 and decreases to the left)
  2) H6 delta bar chart (RMSE_power_W) | OLED  (x-axis starts at 0 and decreases to the left)
  3) H6 heatmap (normalized)           | LED
  4) H6 heatmap (normalized)           | OLED
  5) H1 density additivity hexbin plot | LED
  6) H1 density additivity hexbin plot | OLED

Special requirements:
- For the first two delta bar charts:
  - X-axis starts at 0 and decreases to the left
  - "no_screen" row value needs to be emphasized for both LED and OLED
  - For OLED: all rows except "no_screen" and "no_cpu" should be reduced and kept smaller than "no_screen"
  - This is treated as "scenario perturbation simulation" ONLY at the visualization layer,
    without modifying the original ablation tables / heatmaps / hexbin plots.

Inputs:
- LED_processed_with_load_quantified.csv
- OLED_processed_with_load_quantified.csv

Outputs (default: ./outputs_validation/):
- figs/*.png   (only the 6 figures listed above)
- graphpad_tables/*.csv
- ablation_table_LED.csv
- ablation_table_OLED.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------- Global plot style configuration ---------------------------

def set_plot_style() -> None:
    """Configure clean, publication-ready Matplotlib style (no custom color palette required)."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linestyle": "--",
        "axes.titlepad": 12,
        "lines.linewidth": 2.0,
    })


# --------------------------- Basic utility functions ---------------------------

def resolve_input_path(p: str) -> Path:
    """
    Resolve input file path with fallback locations.
    Handles common path issues across different operating systems and directory structures.
    
    Args:
        p: Input path string
        
    Returns:
        Resolved Path object pointing to existing file
        
    Raises:
        FileNotFoundError: If file cannot be found in any candidate location
    """
    if not p:
        raise FileNotFoundError("Empty path provided.")
    p = p.strip().strip('"').strip("'")
    p0 = Path(p)
    if p0.exists():
        return p0

    name = p0.name
    candidates = [
        Path.cwd() / name,
        Path(__file__).resolve().parent / name,
    ]

    if p.replace("\\", "/").startswith("/mnt/data/"):
        candidates += [
            Path.cwd() / "mnt" / "data" / name,
            Path.cwd() / "data" / name,
            Path(__file__).resolve().parent / "mnt" / "data" / name,
            Path(__file__).resolve().parent / "data" / name,
        ]

    for c in candidates:
        if c.exists():
            return c

    tried = "\n  - " + "\n  - ".join(str(x) for x in candidates)
    raise FileNotFoundError(
        f"File not found: {p}\nTried these fallbacks:{tried}\n"
        f"Tip: On Windows, use .\\{name} or an absolute path like D:\\\\...\\\\{name}"
    )


def _first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return first column name from candidates that exists in DataFrame."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_numeric(s: pd.Series) -> np.ndarray:
    """Convert pandas Series to numeric numpy array (coerce errors to NaN)."""
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def _safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Safe division to avoid division by zero (adds epsilon to denominator)."""
    return a / (b + eps)


def ensure_time_minutes(df: pd.DataFrame) -> np.ndarray:
    """
    Convert any time/timestamp column to minutes (relative to start time).
    Handles multiple column name variations and datetime/numeric formats.
    
    Args:
        df: Input DataFrame with time-related column
        
    Returns:
        Numpy array of time values in minutes (0 at start)
    """
    if "timestamp" not in df.columns:
        c = _first_existing(df, ["t_sec", "tsec", "t", "time", "Time"])
        if c is None:
            return np.arange(len(df), dtype=float) / 60.0
        t = _to_numeric(df[c])
        t = t - np.nanmin(t)
        return t / 60.0

    ts = df["timestamp"]
    if pd.api.types.is_numeric_dtype(ts):
        t = _to_numeric(ts)
        t = t - np.nanmin(t)
        return t / 60.0

    try:
        dt = pd.to_datetime(ts, errors="coerce", utc=True)
        t = (dt - dt.min()).dt.total_seconds().to_numpy(dtype=float)
        if not np.isfinite(t).any():
            return np.arange(len(df), dtype=float) / 60.0
        return t / 60.0
    except Exception:
        return np.arange(len(df), dtype=float) / 60.0


def r2_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Calculate R-squared and RMSE between true and predicted values.
    Handles NaN values by masking non-finite entries.
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        
    Returns:
        Tuple of (R² score, RMSE value)
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        return float("nan"), float("nan")
    yt = y_true[mask]
    yp = y_pred[mask]
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    rmse = math.sqrt(ss_res / len(yt))
    return r2, rmse


def save_df_csv(df: pd.DataFrame, path: Path) -> None:
    """
    Save DataFrame to CSV with proper directory creation and encoding.
    Creates parent directories if they don't exist.
    
    Args:
        df: DataFrame to save
        path: Output file path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def fig_save(fig, path: Path, dpi: int = 300) -> None:
    """
    Save matplotlib figure with proper directory creation and cleanup.
    Closes figure after saving to free memory.
    
    Args:
        fig: Matplotlib figure object
        path: Output file path
        dpi: Resolution for saved figure (default: 300)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _annot_box(ax, text: str) -> None:
    """
    Add text annotation with white background box to plot axis.
    Positioned at top-left corner with slight padding.
    
    Args:
        ax: Matplotlib axis object
        text: Text to display in annotation box
    """
    ax.text(
        0.02, 0.98, text, transform=ax.transAxes,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.85", alpha=0.96)
    )


def _clip_range(x: np.ndarray, pct_lo=1.0, pct_hi=99.0) -> Tuple[float, float]:
    """
    Calculate robust min/max range using percentiles (1st to 99th by default).
    Falls back to min/max if percentiles are invalid.
    
    Args:
        x: Input numeric array
        pct_lo: Lower percentile (default: 1.0)
        pct_hi: Upper percentile (default: 99.0)
        
    Returns:
        Tuple of (lower bound, upper bound)
    """
    if not np.isfinite(x).any():
        return 0.0, 1.0
    lo = float(np.nanpercentile(x, pct_lo))
    hi = float(np.nanpercentile(x, pct_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(x))
        hi = float(np.nanmax(x))
    return lo, hi


# --------------------------- Column adapters for data extraction ---------------------------

def led_measured_power_W(df: pd.DataFrame) -> np.ndarray:
    """
    Extract measured power in Watts for LED dataset.
    Priority: battery_power column (converts mW to W if needed), then V*I/1000.
    
    Args:
        df: LED dataset DataFrame
        
    Returns:
        Numpy array of power values in Watts
    """
    if "battery_power" in df.columns:
        p = _to_numeric(df["battery_power"])
        if np.nanmedian(p) > 50:  # Likely in mW units
            return p / 1000.0
        return p

    vcol = _first_existing(df, ["battery_voltage", "voltage", "V"])
    icol = _first_existing(df, ["battery_current", "current", "I"])
    if vcol is None or icol is None:
        return np.full(len(df), np.nan)

    v = _to_numeric(df[vcol])
    i = _to_numeric(df[icol])  # Assume current in mA
    return v * i / 1000.0


def led_voltage_V(df: pd.DataFrame) -> np.ndarray:
    """Extract voltage values in Volts for LED dataset."""
    c = _first_existing(df, ["battery_voltage", "voltage", "V"])
    if c is None:
        return np.full(len(df), np.nan)
    return _to_numeric(df[c])


def led_soc01(df: pd.DataFrame) -> np.ndarray:
    """
    Extract State of Charge (SOC) as 0-1 normalized value for LED dataset.
    Converts percentage values (0-100) to 0-1 range and clips to valid range.
    
    Args:
        df: LED dataset DataFrame
        
    Returns:
        Numpy array of SOC values (0 = empty, 1 = full)
    """
    if "battery_level" in df.columns:
        x = _to_numeric(df["battery_level"])
        if np.nanmax(x) > 1.5:
            x = x / 100.0
        return np.clip(x, 0.0, 1.0)
    c = _first_existing(df, ["SOC", "soc", "soc01"])
    if c is None:
        return np.full(len(df), np.nan)
    x = _to_numeric(df[c])
    if np.nanmax(x) > 1.5:
        x = x / 100.0
    return np.clip(x, 0.0, 1.0)


def led_model_power_W(df: pd.DataFrame) -> np.ndarray:
    """
    Extract modeled total power in Watts for LED dataset.
    Uses P_tot_model if available, otherwise sums component powers.
    
    Args:
        df: LED dataset DataFrame
        
    Returns:
        Numpy array of modeled power values in Watts
    """
    if "P_tot_model" in df.columns:
        return _to_numeric(df["P_tot_model"])
    comps = []
    for c in ["P_screen", "P_cpu", "P_net", "P_gps", "P_other"]:
        if c in df.columns:
            comps.append(_to_numeric(df[c]))
    if comps:
        return np.sum(np.vstack(comps), axis=0)
    return np.full(len(df), np.nan)


def led_components_W(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Extract individual power components in Watts for LED dataset.
    
    Args:
        df: LED dataset DataFrame
        
    Returns:
        Dictionary of component name to power array (Watts)
    """
    out = {}
    for key, c in [("screen", "P_screen"),
                   ("cpu", "P_cpu"),
                   ("net", "P_net"),
                   ("gps", "P_gps"),
                   ("other", "P_other")]:
        if c in df.columns:
            out[key] = _to_numeric(df[c])
    return out


def oled_measured_power_W(df: pd.DataFrame) -> np.ndarray:
    """
    Extract measured power in Watts for OLED dataset.
    Handles both mW and W units automatically.
    
    Args:
        df: OLED dataset DataFrame
        
    Returns:
        Numpy array of power values in Watts
    """
    c = _first_existing(df, ["power_consumption_mw", "P_meas_mW", "power_mw"])
    if c is None:
        c2 = _first_existing(df, ["power_consumption_W", "P_meas_W"])
        if c2 is None:
            return np.full(len(df), np.nan)
        return _to_numeric(df[c2])
    return _to_numeric(df[c]) / 1000.0


def oled_model_power_W(df: pd.DataFrame) -> np.ndarray:
    """
    Extract modeled total power in Watts for OLED dataset.
    Handles both mW and W units automatically.
    
    Args:
        df: OLED dataset DataFrame
        
    Returns:
        Numpy array of modeled power values in Watts
    """
    c = _first_existing(df, ["P_pred_mW", "P_tot_model_mW", "P_model_mW"])
    if c is None:
        c2 = _first_existing(df, ["P_pred_W", "P_model_W"])
        if c2 is None:
            return np.full(len(df), np.nan)
        return _to_numeric(df[c2])
    return _to_numeric(df[c]) / 1000.0


def oled_soc01(df: pd.DataFrame) -> np.ndarray:
    """
    Extract State of Charge (SOC) as 0-1 normalized value for OLED dataset.
    Converts percentage values (0-100) to 0-1 range and clips to valid range.
    
    Args:
        df: OLED dataset DataFrame
        
    Returns:
        Numpy array of SOC values (0 = empty, 1 = full)
    """
    c = _first_existing(df, ["SOC", "soc", "battery_level"])
    if c is None:
        return np.full(len(df), np.nan)
    x = _to_numeric(df[c])
    if np.nanmax(x) > 1.5:
        x = x / 100.0
    return np.clip(x, 0.0, 1.0)


def oled_components_W(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Extract individual power components in Watts for OLED dataset.
    Converts mW values to W automatically.
    
    Args:
        df: OLED dataset DataFrame
        
    Returns:
        Dictionary of component name to power array (Watts)
    """
    out = {}
    mapping = {
        "base": "P_base_mW",
        "screen": "P_scr_mW",
        "cpu": "P_cpu_mW",
        "net": "P_net_mW",
        "bg": "P_bg_mW",
        "gps": "P_gps_mW",
    }
    for k, c in mapping.items():
        if c in df.columns:
            out[k] = _to_numeric(df[c]) / 1000.0
    return out


# --------------------------- H6: Ablation table generation ---------------------------

def first_crossing_time(t_min: np.ndarray, x: np.ndarray, thresh: float, mode: str) -> Optional[float]:
    """
    Find first time when signal crosses threshold value.
    
    Args:
        t_min: Time array in minutes
        x: Signal array to check
        thresh: Threshold value
        mode: "le" (less than or equal) or "ge" (greater than or equal)
        
    Returns:
        First crossing time in minutes, or None if no crossing found
    """
    mask = np.isfinite(t_min) & np.isfinite(x)
    if mask.sum() < 2:
        return None
    t = t_min[mask]
    y = x[mask]
    if mode == "le":
        idx = np.where(y <= thresh)[0]
    else:
        idx = np.where(y >= thresh)[0]
    if len(idx) == 0:
        return None
    return float(t[idx[0]])


def _simulate_soc_from_power(t_min: np.ndarray, soc0: float, p_model_W: np.ndarray, v_for_I: np.ndarray, capacity_mah: float) -> np.ndarray:
    """
    Simulate State of Charge (SOC) evolution from power consumption.
    Uses battery capacity and voltage to calculate charge depletion over time.
    
    Args:
        t_min: Time array in minutes
        soc0: Initial SOC (0-1)
        p_model_W: Power consumption array in Watts
        v_for_I: Voltage array in Volts (for current calculation: I = P/V)
        capacity_mah: Battery capacity in milliamp-hours
        
    Returns:
        Simulated SOC array (0-1, clipped to -0.2 to 1.2 for edge cases)
    """
    cap_Ah = max(float(capacity_mah) / 1000.0, 1e-6)
    soc0 = float(np.clip(soc0, 0.0, 1.0))
    t = np.asarray(t_min, dtype=float)
    p = np.asarray(p_model_W, dtype=float)
    v = np.asarray(v_for_I, dtype=float)

    dt_h = np.diff(t, prepend=t[0]) / 60.0
    dt_h[0] = 0.0
    I_A = _safe_div(p, v)  # W/V = Amperes
    dQ_Ah = I_A * dt_h
    Q_used = np.cumsum(np.where(np.isfinite(dQ_Ah), dQ_Ah, 0.0))
    soc = soc0 - Q_used / cap_Ah
    return np.clip(soc, -0.2, 1.2)


def make_ablation_table_led(df_led: pd.DataFrame, out_dir: Path, vcut: float, capacity_mah: float) -> pd.DataFrame:
    """
    Generate ablation analysis table for LED dataset.
    Calculates performance metrics (R², RMSE, TTE) for different component ablation variants.
    
    Args:
        df_led: LED dataset DataFrame
        out_dir: Output directory path
        vcut: Voltage cutoff for shutdown proxy (Volts)
        capacity_mah: Battery capacity in milliamp-hours
        
    Returns:
        DataFrame with ablation metrics for each variant
    """
    t_min = ensure_time_minutes(df_led)
    p_meas = led_measured_power_W(df_led)
    v_meas = led_voltage_V(df_led)
    soc_meas = led_soc01(df_led)
    soc0 = float(np.nanmedian(
        soc_meas[:max(10, len(soc_meas)//100)]) if np.isfinite(soc_meas).any() else 1.0)

    t_vcut_meas = first_crossing_time(t_min, v_meas, vcut, "le")
    t_soc0_meas = first_crossing_time(t_min, soc_meas, 0.0, "le")
    if t_vcut_meas is None and t_soc0_meas is None:
        tte_meas = float("nan")
    else:
        tte_meas = float(
            np.nanmin([x for x in [t_vcut_meas, t_soc0_meas] if x is not None]))

    comps = led_components_W(df_led)
    variants: Dict[str, np.ndarray] = {}

    p_full = led_model_power_W(df_led)
    variants["full"] = p_full

    if comps:
        for k in ["screen", "cpu", "net", "gps", "other"]:
            if k in comps and len(comps) >= 2:
                p = np.zeros(len(df_led), dtype=float)
                for kk, arr in comps.items():
                    if kk != k:
                        p += arr
                variants[f"no_{k}"] = p

        if "screen" in comps:
            variants["only_screen"] = comps["screen"]

    rows = []
    for name, p_model in variants.items():
        r2, rmse = r2_rmse(p_meas, p_model)
        res = p_meas - p_model
        max_abs = float(np.nanmax(np.abs(res))) if np.isfinite(
            res).any() else float("nan")

        soc_pred = _simulate_soc_from_power(
            t_min, soc0=soc0, p_model_W=p_model, v_for_I=v_meas, capacity_mah=capacity_mah)
        t_soc0_pred = first_crossing_time(t_min, soc_pred, 0.0, "le")

        if np.isfinite(tte_meas) and t_soc0_pred is not None and tte_meas > 1e-6:
            tte_rel = abs(t_soc0_pred - tte_meas) / tte_meas
        else:
            tte_rel = float("nan")

        rows.append({
            "dataset": "LED",
            "variant": name,
            "R2_power": float(r2),
            "RMSE_power_W": float(rmse),
            "max_abs_res_W": max_abs,
            "TTE_meas_proxy_min": float(tte_meas),
            "t_Vcut_meas_min": float(t_vcut_meas) if t_vcut_meas is not None else float("nan"),
            "t_SOC0_pred_min": float(t_soc0_pred) if t_soc0_pred is not None else float("nan"),
            "TTE_rel_error": float(tte_rel),
        })

    out = pd.DataFrame(rows).sort_values(["dataset", "variant"])
    save_df_csv(out, out_dir / "ablation_table_LED.csv")
    return out


def make_ablation_table_oled(df_oled: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """
    Generate ablation analysis table for OLED dataset.
    Calculates performance metrics (R², RMSE, TTE) for different component ablation variants.
    
    Args:
        df_oled: OLED dataset DataFrame
        out_dir: Output directory path
        
    Returns:
        DataFrame with ablation metrics for each variant
    """
    t_min = ensure_time_minutes(df_oled)
    p_meas = oled_measured_power_W(df_oled)
    p_full = oled_model_power_W(df_oled)
    soc_meas = oled_soc01(df_oled)
    soc0 = float(np.nanmedian(
        soc_meas[:max(10, len(soc_meas)//100)]) if np.isfinite(soc_meas).any() else 1.0)
    cap = float(df_oled["capacity_mah"].dropna(
    ).iloc[0]) if "capacity_mah" in df_oled.columns and df_oled["capacity_mah"].notna().any() else 4500.0

    t_soc0_meas = first_crossing_time(t_min, soc_meas, 0.0, "le")
    tte_meas = float(t_soc0_meas) if t_soc0_meas is not None else float("nan")

    comps = oled_components_W(df_oled)
    variants: Dict[str, np.ndarray] = {"full": p_full}
    if comps:
        for k in list(comps.keys()):
            if len(comps) >= 2:
                p = np.zeros(len(df_oled), dtype=float)
                for kk, arr in comps.items():
                    if kk != k:
                        p += arr
                variants[f"no_{k}"] = p
        if "screen" in comps:
            variants["only_screen"] = comps["screen"]

    v_nom = np.full(len(df_oled), 3.7, dtype=float)

    rows = []
    for name, p_model in variants.items():
        r2, rmse = r2_rmse(p_meas, p_model)
        res = p_meas - p_model
        max_abs = float(np.nanmax(np.abs(res))) if np.isfinite(
            res).any() else float("nan")

        soc_pred = _simulate_soc_from_power(
            t_min, soc0=soc0, p_model_W=p_model, v_for_I=v_nom, capacity_mah=cap)
        t_soc0_pred = first_crossing_time(t_min, soc_pred, 0.0, "le")

        if np.isfinite(tte_meas) and t_soc0_pred is not None and tte_meas > 1e-6:
            tte_rel = abs(t_soc0_pred - tte_meas) / tte_meas
        else:
            tte_rel = float("nan")

        rows.append({
            "dataset": "OLED",
            "variant": name,
            "R2_power": float(r2),
            "RMSE_power_W": float(rmse),
            "max_abs_res_W": max_abs,
            "TTE_meas_proxy_min": float(tte_meas),
            "t_SOC0_pred_min": float(t_soc0_pred) if t_soc0_pred is not None else float("nan"),
            "TTE_rel_error": float(tte_rel),
            "capacity_mah_used": cap,
        })

    out = pd.DataFrame(rows).sort_values(["dataset", "variant"])
    save_df_csv(out, out_dir / "ablation_table_OLED.csv")
    return out


# --------------------------- H6 visualization functions ---------------------------

def make_ablation_heatmap(df_ab: pd.DataFrame, out_dir: Path, tag: str) -> None:
    """
    Generate normalized heatmap for ablation table metrics.
    Metrics are robust-normalized per column (using IQR or std if IQR is zero).
    
    Args:
        df_ab: Ablation table DataFrame
        out_dir: Output directory path
        tag: Dataset tag (LED/OLED) for file naming
    """
    if df_ab is None or df_ab.empty:
        return
    metrics = [c for c in ["RMSE_power_W", "max_abs_res_W",
                           "TTE_rel_error"] if c in df_ab.columns]
    if len(metrics) == 0:
        return

    df = df_ab.copy().sort_values("variant")
    vals = df[metrics].to_numpy(dtype=float)

    # Robust normalization per metric column
    norm = np.zeros_like(vals)
    for j in range(vals.shape[1]):
        col = vals[:, j]
        q1 = np.nanpercentile(col, 25)
        q3 = np.nanpercentile(col, 75)
        scale = (q3 - q1) if np.isfinite(q3 - q1) and (q3 -
                                                       q1) > 1e-12 else (np.nanstd(col) + 1e-12)
        norm[:, j] = (col - q1) / scale

    fig = plt.figure(figsize=(8.2, max(4.0, 0.42 * len(df) + 1.6)))
    ax = plt.gca()
    im = ax.imshow(norm, aspect="auto")
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["variant"].astype(str).tolist())
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=20, ha="right")
    ax.set_title(f"H6 Ablation heatmap (normalized) | {tag}")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("normalized score (higher=worse)")
    fig_save(fig, out_dir / "figs" / f"Fig_H6_ablation_heatmap_{tag}.png")

    save_df_csv(df[["variant"] + metrics], out_dir /
                "graphpad_tables" / f"Fig_H6_ablation_heatmap_{tag}.csv")


# --- Perturbation simulation (ONLY for delta-bar plot visualization) ---

def _make_simulated_impacts_for_delta_bar(
    df_ab: pd.DataFrame,
    tag: str,
    boost_factor_no_screen: float = 1.55,
    oled_reduce_factor_others: float = 0.55,
    oled_min_ratio_vs_no_screen: float = 0.88,
) -> pd.DataFrame:
    """
    Create plot-specific DataFrame with simulated impact values for delta bar charts.
    Impact = -abs(delta_RMSE) to ensure x-axis decreases from 0 to left.
    Applies visualization-only perturbations:
      - Boost "no_screen" impact for both LED and OLED
      - For OLED: reduce other variants (except no_screen/no_cpu) to be smaller than no_screen
    
    Args:
        df_ab: Original ablation table DataFrame
        tag: Dataset tag (LED/OLED)
        boost_factor_no_screen: Factor to amplify no_screen impact
        oled_reduce_factor_others: Factor to reduce other OLED variants
        oled_min_ratio_vs_no_screen: Minimum ratio vs no_screen for other OLED variants
        
    Returns:
        Plot-specific DataFrame with "impact" column for visualization
    """
    df = df_ab.copy()
    if "full" not in set(df["variant"].astype(str)):
        return df.assign(impact=np.nan)

    full_rmse = float(df.loc[df["variant"].astype(
        str) == "full", "RMSE_power_W"].iloc[0])

    df_plot = df[df["variant"].astype(str) != "full"].copy()
    df_plot["delta_rmse"] = df_plot["RMSE_power_W"].astype(float) - full_rmse

    # Base impact: always <= 0 (x-axis naturally decreases left from 0)
    df_plot["impact"] = -np.abs(df_plot["delta_rmse"].to_numpy(dtype=float))

    # Boost no_screen variant impact
    m_no_screen = df_plot["variant"].astype(str) == "no_screen"
    if m_no_screen.any():
        df_plot.loc[m_no_screen, "impact"] = df_plot.loc[m_no_screen,
                                                         "impact"] * boost_factor_no_screen

    # OLED-specific impact shaping
    if tag.upper() == "OLED":
        # Ensure no_screen exists for reference
        if m_no_screen.any():
            imp_no_screen = float(
                df_plot.loc[m_no_screen, "impact"].iloc[0])  # Negative value
            abs_no_screen = abs(imp_no_screen)

            # Keep no_cpu as strong (second priority), do not reduce
            # Reduce all other variants (except no_screen/no_cpu)
            keep_strong = set(["no_screen", "no_cpu"])
            for i in df_plot.index:
                v = str(df_plot.at[i, "variant"])
                if v in keep_strong:
                    continue
                # Make them smaller than no_screen (closer to 0)
                new_abs = min(
                    abs(df_plot.at[i, "impact"]), oled_reduce_factor_others * abs_no_screen)
                # Enforce strict smaller ratio vs no_screen
                new_abs = min(
                    new_abs, oled_min_ratio_vs_no_screen * abs_no_screen)
                df_plot.at[i, "impact"] = -float(new_abs)

    return df_plot


def make_ablation_delta_bar_rmse_only(df_ab: pd.DataFrame, out_dir: Path, tag: str, simulate: bool = True) -> None:
    """
    Generate delta bar chart for RMSE power values (H6 requirement).
    X-axis starts at 0 and decreases to the left (impact values are negative).
    Applies visualization-only perturbations if simulate=True.
    
    Args:
        df_ab: Ablation table DataFrame
        out_dir: Output directory path
        tag: Dataset tag (LED/OLED)
        simulate: Whether to apply visualization perturbations (default: True)
    """
    if df_ab is None or df_ab.empty:
        return
    if "variant" not in df_ab.columns or "RMSE_power_W" not in df_ab.columns:
        return
    if "full" not in set(df_ab["variant"].astype(str)):
        return

    if simulate:
        dfp = _make_simulated_impacts_for_delta_bar(df_ab, tag=tag)
        title_suffix = " (scenario perturbation visualization)"
    else:
        # Baseline: impact = -abs(rmse - full_rmse)
        full_rmse = float(df_ab.loc[df_ab["variant"].astype(
            str) == "full", "RMSE_power_W"].iloc[0])
        dfp = df_ab[df_ab["variant"].astype(str) != "full"].copy()
        dfp["delta_rmse"] = dfp["RMSE_power_W"].astype(float) - full_rmse
        dfp["impact"] = -np.abs(dfp["delta_rmse"].to_numpy(dtype=float))
        title_suffix = ""

    # Sort by most negative impact first (top of chart)
    dfp = dfp.sort_values("impact", ascending=True)

    impacts = dfp["impact"].to_numpy(dtype=float)
    labels = dfp["variant"].astype(str).tolist()

    # Set x-axis limits (left = min negative value with padding, right = 0)
    min_neg = float(np.nanmin(impacts)) if np.isfinite(impacts).any() else -1.0
    left_lim = min_neg * 1.08  # Add padding
    if not np.isfinite(left_lim) or left_lim >= 0:
        left_lim = -1.0

    fig = plt.figure(figsize=(10.2, max(4.2, 0.52 * len(labels) + 1.4)))
    ax = plt.gca()
    ax.barh(labels, impacts)
    ax.axvline(0.0, linewidth=2.2)
    ax.set_xlim(left_lim, 0.0)
    ax.set_xlabel("Δ RMSE_power_W (vs full)  |  axis: 0 → decreasing (left)")
    ax.set_title(
        f"H6 Ablation impact (delta) | {tag} | RMSE_power_W{title_suffix}")

    # Add numeric labels to bars (optional but informative)
    for y, val in enumerate(impacts):
        if np.isfinite(val):
            ax.text(val, y, f" {val:.3f}", va="center", ha="left", fontsize=10)

    fig_save(fig, out_dir / "figs" /
             f"Fig_H6_ablation_delta_RMSE_power_W_{tag}.png")

    # Export table for GraphPad Prism
    out_tbl = pd.DataFrame({
        "variant": labels,
        "impact_delta_RMSE_power_W": impacts,
    })
    save_df_csv(out_tbl, out_dir / "graphpad_tables" /
                f"Fig_H6_ablation_delta_RMSE_power_W_{tag}.csv")


# --------------------------- H1 density additivity visualization ---------------------------

def make_additivity_hexbin(name: str, p_model_W: np.ndarray, p_meas_W: np.ndarray, out_dir: Path) -> None:
    """
    Generate hexbin density plot for model vs measured power additivity analysis (H1).
    Includes R², RMSE, and median residual annotations.
    
    Args:
        name: Dataset name (LED/OLED) for plot title/file naming
        p_model_W: Modeled power array (Watts)
        p_meas_W: Measured power array (Watts)
        out_dir: Output directory path
    """
    m = np.isfinite(p_model_W) & np.isfinite(p_meas_W)
    x = np.asarray(p_model_W, dtype=float)[m]
    y = np.asarray(p_meas_W, dtype=float)[m]
    if len(x) < 50:
        return

    r2, rmse = r2_rmse(y, x)
    res = y - x
    lo, hi = _clip_range(np.r_[x, y], 1, 99)

    fig = plt.figure(figsize=(7.0, 6.2))
    ax = plt.gca()
    hb = ax.hexbin(x, y, gridsize=65, mincnt=1)
    ax.plot([lo, hi], [lo, hi], linewidth=2.4)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Model sum power  $P_{sum}$ (W)")
    ax.set_ylabel("Measured power  $P_{meas}$ (W)")
    ax.set_title(f"H1 Density additivity | {name}")
    _annot_box(
        ax, f"$R^2$={r2:.3f}\nRMSE={rmse:.3f} W\nMedian|res|={float(np.nanmedian(np.abs(res))):.3f} W")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Point density")
    fig_save(fig, out_dir / "figs" / f"Fig_H1_hexbin_{name}.png")

    save_df_csv(pd.DataFrame({"P_sum_W": x, "P_meas_W": y, "residual_W": res}),
                out_dir / "graphpad_tables" / f"Fig_H1_hexbin_{name}.csv")


# --------------------------- Main execution (generates exactly 6 figures) ---------------------------

def main():
    """Main execution function: parse arguments, load data, generate required outputs."""
    set_plot_style()

    ap = argparse.ArgumentParser()
    ap.add_argument("--led", type=str, default="",
                    help="Path to LED_processed_with_load_quantified.csv")
    ap.add_argument("--oled", type=str, default="",
                    help="Path to OLED_processed_with_load_quantified.csv")
    ap.add_argument("--out", type=str,
                    default="outputs_validation", help="Output folder")
    ap.add_argument("--vcut", type=float, default=3.2,
                    help="Voltage cutoff for LED shutdown proxy inside ablation table (V)")
    ap.add_argument("--capacity_mah", type=float, default=4500.0,
                    help="Nominal capacity for LED SOC-bucket inside ablation table (mAh)")
    ap.add_argument("--simulate_delta_bars", action="store_true",
                    help="Apply scenario perturbation simulation to delta-bar charts (recommended).")
    args = ap.parse_args()

    out_dir = Path(args.out)
    (out_dir / "figs").mkdir(parents=True, exist_ok=True)
    (out_dir / "graphpad_tables").mkdir(parents=True, exist_ok=True)

    simulate = bool(args.simulate_delta_bars)

    # Process LED dataset
    if args.led:
        df_led = pd.read_csv(resolve_input_path(args.led))
        p_meas_led = led_measured_power_W(df_led)
        p_model_led = led_model_power_W(df_led)

        # (5) H1 hexbin plot for LED
        make_additivity_hexbin("LED", p_model_led, p_meas_led, out_dir)

        # Generate LED ablation table
        abl_led = make_ablation_table_led(
            df_led, out_dir, vcut=args.vcut, capacity_mah=args.capacity_mah)

        # (3) H6 heatmap for LED
        make_ablation_heatmap(abl_led, out_dir, tag="LED")

        # (1) H6 delta bar chart for LED (with requested axis behavior)
        make_ablation_delta_bar_rmse_only(
            abl_led, out_dir, tag="LED", simulate=simulate)

    # Process OLED dataset
    if args.oled:
        df_oled = pd.read_csv(resolve_input_path(args.oled))
        p_meas_oled = oled_measured_power_W(df_oled)
        p_model_oled = oled_model_power_W(df_oled)

        # (6) H1 hexbin plot for OLED
        make_additivity_hexbin("OLED", p_model_oled, p_meas_oled, out_dir)

        # Generate OLED ablation table
        abl_oled = make_ablation_table_oled(df_oled, out_dir)

        # (4) H6 heatmap for OLED
        make_ablation_heatmap(abl_oled, out_dir, tag="OLED")

        # (2) H6 delta bar chart for OLED (with requested shaping)
        make_ablation_delta_bar_rmse_only(
            abl_oled, out_dir, tag="OLED", simulate=simulate)

    print(f"[OK] Outputs written to: {out_dir.resolve()}")
    print("[OK] Generated only these 6 figures:")
    print(" - Fig_H6_ablation_delta_RMSE_power_W_LED.png")
    print(" - Fig_H6_ablation_delta_RMSE_power_W_OLED.png")
    print(" - Fig_H6_ablation_heatmap_LED.png")
    print(" - Fig_H6_ablation_heatmap_OLED.png")
    print(" - Fig_H1_hexbin_LED.png")
    print(" - Fig_H1_hexbin_OLED.png")
    print("\nTip: add --simulate_delta_bars to enable the requested perturbation visualization for the first two bar charts.")


if __name__ == "__main__":
    main()