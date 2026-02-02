# preprocess_phone_9steps_units.py
# Process mobile phone logs with 9-step continuous time modeling preprocessing,
# overwrite original columns with converted units/meanings:
#   1) bright_level: brightness level -> nits (overwrite original column)
#   2) wifi_intensity: RSSI(dBm) -> normalized signal quality q∈[0,1] (overwrite original column)
#   3) wifi_speed / wifi_rx / wifi_tx: Mbps -> bit/s (overwrite original columns)
#   4) mobile_rx / mobile_tx: cumulative byte count -> rate (bytes/s) -> bit/s (overwrite original columns)
#   5) Keep all other columns unchanged (no preservation of pre-conversion columns), output full table
#
# Dependencies: pip install pandas numpy

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class Cfg:
    # Step 0 / Step 2: Time and Resampling
    target_hz: int = 1                 # Target frequency: 1Hz (1 record per second)
    timestamp_col: str = "timestamp"   # Timestamp column name

    # Step 4: Continuization Strategy
    # Fill newly added second-level rows only, keep missing values in original rows unchanged
    continuous_method: str = "piecewise"  # "piecewise" or "linear"

    # bright_level -> nits conversion parameters
    bright_level_max: float = 255.0
    screen_min_nits: float = 2.0
    screen_max_nits: float = 600.0
    bright_gamma: float = 1.0

    # Wi-Fi RSSI(dBm) -> normalized quality q∈[0,1] parameters
    # Typical range: -100 (poor) ~ -50 (excellent)
    rssi_min_dbm: float = -100.0
    rssi_max_dbm: float = -50.0

    # Discharge segment filtering (optional)
    keep_only_collected: bool = False      # Keep only successfully collected rows if True (requires 'collected' column)
    keep_only_discharging: bool = False    # Keep only discharge segments if True (for discharge model training)
    discharge_current_threshold: float = -1e-6  # battery_current < 0 indicates discharging

    # Step 7: Sub-module power consumption inputs (optional, placeholder coefficients for later calibration)
    build_power_inputs: bool = True
    display_type: str = "LCD"  # "LCD" or "OLED"
    lcd_a0: float = 0.2
    lcd_a1: float = 0.002      # W / nit
    oled_a0: float = 0.15
    oled_a1: float = 0.0018
    cpu_idle_w: float = 0.3
    cpu_alpha: float = 0.02    # W / (cpu_usage %)
    cpu_beta: float = 0.08     # W / (freq_GHz)
    net_base_w: float = 0.15
    net_gamma_bits: float = 1e-12  # W / (bit/s) (placeholder magnitude)
    gps_w: float = 0.08
    other_w: float = 0.05


# --------------------------
# Step 2: Time axis alignment utilities
# --------------------------
def infer_epoch_unit(x: pd.Series) -> str:
    """Infer epoch timestamp unit: s / ms / us / ns"""
    v = pd.to_numeric(x, errors="coerce").dropna()
    if v.empty:
        return "s"
    m = float(v.iloc[0])
    if m > 1e17:
        return "ns"
    if m > 1e14:
        return "us"
    if m > 1e11:
        return "ms"
    return "s"


def build_second_resolved_datetime(ts: pd.Series) -> pd.DatetimeIndex:
    """
    Convert timestamp to second-resolved unique datetime index:
    - For numeric epoch values: parse with inferred unit
    - For duplicate minute-level string timestamps: add 0..59 seconds by occurrence order within same minute
    """
    if pd.api.types.is_numeric_dtype(ts):
        unit = infer_epoch_unit(ts)
        dt = pd.to_datetime(ts, unit=unit, errors="coerce")
        if dt.isna().any():
            raise ValueError("Failed to parse epoch timestamp: invalid values exist")
        return pd.DatetimeIndex(dt)

    dt_min = pd.to_datetime(ts, errors="coerce")
    if dt_min.isna().any():
        raise ValueError("Failed to parse string timestamp: invalid values exist")

    dup_rate = dt_min.duplicated().mean()
    if dup_rate > 0.1:
        sec_in_group = dt_min.groupby(dt_min).cumcount()
        dt = dt_min + pd.to_timedelta(sec_in_group, unit="s")
        return pd.DatetimeIndex(dt)

    return pd.DatetimeIndex(dt_min)


# --------------------------
# Step 4: Fill only newly added rows (preserve original rows)
# --------------------------
def fill_only_new_rows(
    df_1hz: pd.DataFrame,
    original_index: pd.DatetimeIndex,
    cols: List[str],
    method: str = "piecewise",
) -> None:
    """
    Continuize specified columns, write results only to newly added rows (index not in original_index):
    - piecewise: forward fill + backward fill
    - linear: time-based interpolation + forward/backward fill
    """
    cols = [c for c in cols if c in df_1hz.columns]
    if not cols:
        return

    is_new = ~df_1hz.index.isin(original_index)

    for c in cols:
        s = df_1hz[c]
        if method == "linear":
            # Time interpolation only works for numeric values, non-numeric remains NaN -> then ffill/bfill
            s2 = pd.to_numeric(s, errors="coerce").interpolate(
                method="time").ffill().bfill()
        else:
            s2 = s.ffill().bfill()

        # Fill only new rows to avoid modifying original data
        df_1hz.loc[is_new, c] = s2.loc[is_new]


# --------------------------
# Step 6: Convert cumulative counter to per-second rate
# --------------------------
def counter_to_rate_per_sec(counter: pd.Series) -> pd.Series:
    """
    Convert cumulative counter to per-second increment (≈ bytes/s):
    - Negative diff values (wrap-around/reset) set to 0
    """
    x = pd.to_numeric(counter, errors="coerce").fillna(
        method="ffill").fillna(0.0)
    d = x.diff()
    d = d.where(d >= 0, 0.0)
    return d  # At 1Hz sampling rate, diff equals bytes per second


# --------------------------
# Main function: 9-step preprocessing pipeline
# --------------------------
def preprocess_9steps(in_csv: str, out_csv: str, cfg: Cfg = Cfg()) -> None:
    # =========================
    # Step 1) Load raw data (preserve column structure)
    # =========================
    df_raw = pd.read_csv(in_csv)
    if cfg.timestamp_col not in df_raw.columns:
        raise ValueError(f"Missing timestamp column: {cfg.timestamp_col}")

    # Optional: Keep only successfully collected rows
    if cfg.keep_only_collected and "collected" in df_raw.columns:
        df_raw = df_raw[df_raw["collected"] == 1].copy()

    # =========================
    # Step 2) Unify time axis (build second-resolved datetime, resample to 1Hz)
    # =========================
    dt = build_second_resolved_datetime(df_raw[cfg.timestamp_col])
    df_raw = df_raw.copy()
    df_raw["__datetime__"] = dt
    df_raw = df_raw.sort_values("__datetime__")
    df_raw = df_raw.set_index("__datetime__")
    df_raw = df_raw[~df_raw.index.duplicated(keep="last")]

    original_index = df_raw.index.copy()

    # Build 1Hz target time axis: fill missing seconds (minimal changes if already 1Hz)
    df_1hz = df_raw.resample("1s").asfreq()

    # Relative time for continuous modeling: t_sec
    t0 = df_1hz.index[0]
    df_1hz["t_sec"] = (df_1hz.index - t0).total_seconds().astype(float)

    # =========================
    # Step 3) Convert calculation columns to numeric type
    # Note: Only convert columns involved in transformations to preserve original data
    # =========================
    numeric_need = [
        # Screen
        "bright_level", "screen_status",
        # Wi-Fi
        "wifi_intensity", "wifi_speed", "wifi_rx", "wifi_tx", "wifi_status",
        # Cellular
        "mobile_rx", "mobile_tx", "mobile_status",
        # CPU
        "cpu_usage", "cpu_temperature",
        "frequency_core0", "frequency_core1", "frequency_core2", "frequency_core3",
        "frequency_core4", "frequency_core5", "frequency_core6", "frequency_core7",
        # GPS
        "gps_activity",
        # Battery
        "battery_current", "battery_voltage", "battery_power",
    ]
    for c in numeric_need:
        if c in df_1hz.columns:
            df_1hz[c] = pd.to_numeric(df_1hz[c], errors="coerce")

    # =========================
    # Step 4) Continuization (fill only new rows, preserve original rows)
    # =========================
    # 4.1 Switch/gear columns: piecewise constant
    step_like = [
        "screen_status", "bright_level",
        "wifi_status", "mobile_status",
        "gps_activity",
    ]
    fill_only_new_rows(df_1hz, original_index, step_like, method="piecewise")

    # 4.2 Continuous sensor columns: linear interpolation or piecewise constant
    cont_like = ["battery_voltage", "cpu_temperature",
                 "wifi_intensity", "wifi_speed", "battery_power"]
    fill_only_new_rows(df_1hz, original_index, cont_like,
                       method=cfg.continuous_method)

    # 4.3 Other columns: fill new rows with piecewise method if needed
    # Note: Only fills inserted rows, no changes to original data
    fill_only_new_rows(df_1hz, original_index, list(
        df_1hz.columns), method="piecewise")

    # =========================
    # Step 5) Optional: Keep only discharge segments (battery_current < 0)
    # =========================
    if cfg.keep_only_discharging and "battery_current" in df_1hz.columns:
        df_1hz = df_1hz[df_1hz["battery_current"] <
                        cfg.discharge_current_threshold].copy()

    # =========================
    # Step 6) Unit unification (overwrite original columns, no preservation of old values)
    # =========================
    # 6.1 bright_level: level -> nits (overwrite)
    if "bright_level" in df_1hz.columns:
        lvl = df_1hz["bright_level"].fillna(0.0).clip(0, cfg.bright_level_max)
        frac = (lvl / cfg.bright_level_max).clip(0, 1)
        df_1hz["bright_level"] = cfg.screen_min_nits + (frac ** cfg.bright_gamma) * (
            cfg.screen_max_nits - cfg.screen_min_nits
        )

    # 6.2 Wi-Fi speed: Mbps -> bit/s (overwrite wifi_speed; apply same to wifi_rx/wifi_tx if exists)
    if "wifi_speed" in df_1hz.columns:
        df_1hz["wifi_speed"] = df_1hz["wifi_speed"].fillna(0.0) * 1e6
    for c in ["wifi_rx", "wifi_tx"]:
        if c in df_1hz.columns:
            # Apply Mbps -> bit/s conversion for link rate values (typical values: 468/650/866)
            df_1hz[c] = pd.to_numeric(
                df_1hz[c], errors="coerce").fillna(0.0) * 1e6

    # 6.3 Cellular traffic: cumulative bytes -> bytes/s -> bit/s (overwrite mobile_rx/mobile_tx)
    # Note: This conversion is correct for cumulative byte counters; 
    # Modify to *8 only if original values are already rate values
    if "mobile_rx" in df_1hz.columns:
        rx_bps = counter_to_rate_per_sec(df_1hz["mobile_rx"])  # bytes/s
        df_1hz["mobile_rx"] = rx_bps * 8.0                     # bit/s
    if "mobile_tx" in df_1hz.columns:
        tx_bps = counter_to_rate_per_sec(df_1hz["mobile_tx"])  # bytes/s
        df_1hz["mobile_tx"] = tx_bps * 8.0                     # bit/s

    # 6.4 Wi-Fi signal strength: RSSI(dBm) -> normalized quality q∈[0,1] (overwrite)
    if "wifi_intensity" in df_1hz.columns:
        rssi = df_1hz["wifi_intensity"].astype(float)
        # Normalization: q = (rssi - min)/(max-min)
        q = (rssi - cfg.rssi_min_dbm) / (cfg.rssi_max_dbm - cfg.rssi_min_dbm)
        q = q.clip(0.0, 1.0).fillna(0.0)

        # Set quality to 0 when Wi-Fi is off (physically meaningful)
        if "wifi_status" in df_1hz.columns:
            wifi_on = (df_1hz["wifi_status"].fillna(0.0) > 0).astype(float)
            q = q * wifi_on

        df_1hz["wifi_intensity"] = q

    # =========================
    # Step 7) Build interpretable inputs (optional): P_screen / P_cpu / P_net / P_gps / P_other / P_tot_model
    # =========================
    if cfg.build_power_inputs:
        # Screen power input (W)
        screen_on = (df_1hz["screen_status"].fillna(0.0) > 0).astype(
            float) if "screen_status" in df_1hz.columns else 0.0
        bright_nits = df_1hz["bright_level"].fillna(
            0.0) if "bright_level" in df_1hz.columns else 0.0
        if cfg.display_type.upper() == "OLED":
            df_1hz["P_screen"] = screen_on * \
                (cfg.oled_a0 + cfg.oled_a1 * bright_nits)
        else:
            df_1hz["P_screen"] = screen_on * \
                (cfg.lcd_a0 + cfg.lcd_a1 * bright_nits)

        # CPU power input (W)
        cpu_usage = df_1hz["cpu_usage"].fillna(
            0.0) if "cpu_usage" in df_1hz.columns else 0.0
        freq_cols = [
            c for c in df_1hz.columns if c.startswith("frequency_core")]
        if freq_cols:
            cpu_freq_ghz = (df_1hz[freq_cols].mean(
                axis=1) / 1000.0).fillna(0.0)
        else:
            cpu_freq_ghz = 0.0
        df_1hz["P_cpu"] = cfg.cpu_idle_w + cfg.cpu_alpha * \
            cpu_usage + cfg.cpu_beta * cpu_freq_ghz

        # Network power input (W)
        # wifi_speed is already in bit/s; wifi_intensity is normalized q∈[0,1]
        wifi_rate_bits = df_1hz["wifi_speed"].fillna(
            0.0) if "wifi_speed" in df_1hz.columns else 0.0
        q_wifi = df_1hz["wifi_intensity"].fillna(
            0.0) if "wifi_intensity" in df_1hz.columns else 0.0
        net_on = 1.0
        if "wifi_status" in df_1hz.columns or "mobile_status" in df_1hz.columns:
            wifi_on = (df_1hz.get("wifi_status", 0.0).fillna(0.0) > 0)
            mobile_on = (df_1hz.get("mobile_status", 0.0).fillna(0.0) > 0)
            net_on = (wifi_on | mobile_on).astype(float)

        # Use (1 - q) to represent "weak signal amplification" (lower q = higher power consumption)
        gain = 1.0 + (1.0 - q_wifi)
        df_1hz["P_net"] = net_on * \
            (cfg.net_base_w + cfg.net_gamma_bits * wifi_rate_bits * gain)

        # GPS power input (W)
        gps_on = (df_1hz.get("gps_activity", 0.0).fillna(
            0.0) > 0).astype(float)
        df_1hz["P_gps"] = gps_on * cfg.gps_w

        # Other baseline power (W)
        df_1hz["P_other"] = cfg.other_w

        # Total modeled power consumption (W)
        df_1hz["P_tot_model"] = df_1hz["P_screen"] + df_1hz["P_cpu"] + \
            df_1hz["P_net"] + df_1hz["P_gps"] + df_1hz["P_other"]

    # =========================
    # Step 8) Battery-side observed power (for validation/calibration)
    # =========================
    if "battery_power" in df_1hz.columns:
        # battery_power is already converted to positive values
        df_1hz["P_batt_discharge_W"] = df_1hz["battery_power"].clip(lower=0.0)
    elif "battery_voltage" in df_1hz.columns and "battery_current" in df_1hz.columns:
        # Note: Divide by 1000 if battery_current is in mA; default unit is A (modify if needed)
        df_1hz["P_batt_discharge_W"] = (-(df_1hz["battery_voltage"]
                                        * df_1hz["battery_current"])).clip(lower=0.0)

    # =========================
    # Step 8.5) Unify battery-related values to positive (overwrite original columns)
    # Note:
    # - Voltage is typically positive, use abs() as safeguard
    # - Current/power: convert to positive magnitude regardless of original sign
    # =========================
    for c in ["battery_voltage", "battery_current", "battery_power"]:
        if c in df_1hz.columns:
            df_1hz[c] = pd.to_numeric(df_1hz[c], errors="coerce").abs()

    # =========================
    # Step 9) Output full table: keep original columns (except overwritten) + new columns
    # =========================
    # Write index time back to timestamp column (with seconds) for unique timestamps
    # Required modification for continuous time modeling
    df_1hz[cfg.timestamp_col] = df_1hz.index.strftime("%Y-%m-%d %H:%M:%S")

    # Export to CSV
    df_1hz.reset_index(drop=True).to_csv(
        out_csv, index=False, encoding="utf-8-sig")


# --------------------------
# Usage example (modify paths only)
# --------------------------
if __name__ == "__main__":
    IN_CSV = r"D:\LaterDeletee\70a09b5174d07fff_20230221_dynamic_processed.csv"
    OUT_CSV = r"D:\LaterDeletee\70a09b5174d07fff_20230221_dynamic_processed_preprocessed_9steps.csv"

    cfg = Cfg(
        target_hz=1,
        continuous_method="piecewise",
        # Brightness mapping (adjust based on device/model)
        screen_max_nits=600.0,
        bright_gamma=1.0,
        # RSSI normalization range
        rssi_min_dbm=-100.0,
        rssi_max_dbm=-50.0,
        # Optional configurations
        keep_only_collected=False,
        keep_only_discharging=False,
        build_power_inputs=True,
        display_type="LCD",  # Change to "OLED" for OLED screens
    )

    preprocess_9steps(IN_CSV, OUT_CSV, cfg)
    print("Saved:", OUT_CSV)