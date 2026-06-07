"""features.py — Ingeniería de features compartida por entrenamiento e inferencia.

Dos problemas:
  • Forecast: predecir minutos de MAÑANA por (usuario, app) con info hasta hoy.
  • Binge: predecir si MAÑANA habrá atracón (día) con info hasta hoy.

Las features de binge están alineadas con la estructura causal del generador
(fin de semana, momentum, uso nocturno previo, impulsividad, racha sin atracón)
para que SHAP exponga triggers legibles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import APPS, OCIO_CATS

FORECAST_FEATURES = [
    "lag_1", "lag_2", "lag_7", "lag_14", "roll_mean_3", "roll_mean_7", "roll_mean_14",
    "roll_std_7", "trend", "user_app_mean", "dow", "is_weekend", "app_code", "cat_code",
    "impulsivity",
]
BINGE_FEATURES = [
    "is_weekend", "dow", "prev_ocio", "roll3_ocio", "roll7_ocio", "roll14_ocio", "momentum",
    "prev_late_night_ratio", "roll3_late_night", "prev_app_switches", "days_since_binge",
    "impulsivity",
]

_APP_CODES = {a: i for i, a in enumerate(APPS)}
_CAT_CODES = {c: i for i, c in enumerate(sorted(set(APPS.values())))}


# ── Forecast ────────────────────────────────────────────────────────────────

def build_forecast_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.sort_values(["user_id", "app", "day"]).copy()
    g = df.groupby(["user_id", "app"], sort=False)["minutes"]
    df["lag_1"] = g.shift(1)
    df["lag_2"] = g.shift(2)
    df["lag_7"] = g.shift(7)
    df["lag_14"] = g.shift(14)
    df["roll_mean_3"] = g.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["roll_mean_7"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    df["roll_mean_14"] = g.transform(lambda s: s.shift(1).rolling(14, min_periods=1).mean())
    df["roll_std_7"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=2).std())
    df["user_app_mean"] = g.transform(lambda s: s.shift(1).expanding().mean())
    df["trend"] = df["roll_mean_3"] - df["roll_mean_7"]
    df["app_code"] = df["app"].map(_APP_CODES)
    df["cat_code"] = df["category"].map(_CAT_CODES)
    df = df.dropna(subset=["lag_1", "lag_7"])
    return df[FORECAST_FEATURES + ["day", "user_id"]], df["minutes"]


# ── Binge (nivel día) ─────────────────────────────────────────────────────────

def _daily_ocio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ocio_min"] = np.where(df["category"].isin(OCIO_CATS), df["minutes"], 0.0)
    daily = df.groupby(["user_id", "day"]).agg(
        ocio_total=("ocio_min", "sum"),
        late_night_ratio=("late_night_ratio", "first"),
        app_switches=("app_switches", "first"),
        is_binge=("is_binge", "first"),
        is_weekend=("is_weekend", "first"),
        dow=("dow", "first"),
        impulsivity=("impulsivity", "first"),
    ).reset_index()
    return daily


def _days_since_binge(binge: pd.Series) -> np.ndarray:
    out = np.empty(len(binge), dtype=float)
    since = 7.0
    for i, b in enumerate(binge.to_numpy()):
        out[i] = since
        since = 0.0 if b == 1 else since + 1.0
    return out


def build_binge_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    daily = _daily_ocio(df).sort_values(["user_id", "day"])
    g = daily.groupby("user_id", sort=False)
    daily["prev_ocio"] = g["ocio_total"].shift(1)
    daily["roll3_ocio"] = g["ocio_total"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    daily["roll7_ocio"] = g["ocio_total"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    daily["roll14_ocio"] = g["ocio_total"].transform(lambda s: s.shift(1).rolling(14, min_periods=1).mean())
    daily["user_ocio_mean"] = g["ocio_total"].transform(lambda s: s.shift(1).expanding().mean())
    daily["momentum"] = daily["roll3_ocio"] / (daily["user_ocio_mean"] + 1e-6) - 1.0
    daily["prev_late_night_ratio"] = g["late_night_ratio"].shift(1)
    daily["roll3_late_night"] = g["late_night_ratio"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    daily["prev_app_switches"] = g["app_switches"].shift(1)
    daily["days_since_binge"] = np.concatenate(
        [_days_since_binge(grp["is_binge"]) for _, grp in g]
    )
    daily = daily.dropna(subset=["prev_ocio", "roll7_ocio", "prev_late_night_ratio"])
    return daily[BINGE_FEATURES], daily["is_binge"]


# ── Inferencia: features a partir del historial reciente de UN usuario ──────--

def forecast_row_from_history(hist: pd.DataFrame, app: str, target_dow: int) -> dict:
    """hist: filas (day, app, minutes, category) de UN usuario. Devuelve la fila
    de features para predecir 'mañana' de `app`."""
    s = hist[hist["app"] == app].sort_values("day")["minutes"]
    cat = APPS[app]
    arr = s.to_numpy(dtype=float)
    g = lambda i: float(arr[-i]) if len(arr) >= i else (float(arr[0]) if len(arr) else 0.0)
    rm3 = float(arr[-3:].mean()) if len(arr) else 0.0
    rm7 = float(arr[-7:].mean()) if len(arr) else 0.0
    rm14 = float(arr[-14:].mean()) if len(arr) else 0.0
    rs7 = float(arr[-7:].std()) if len(arr) >= 2 else 0.0
    return {
        "lag_1": g(1), "lag_2": g(2), "lag_7": g(7), "lag_14": g(14),
        "roll_mean_3": rm3, "roll_mean_7": rm7, "roll_mean_14": rm14,
        "roll_std_7": rs7, "trend": rm3 - rm7,
        "user_app_mean": float(arr.mean()) if len(arr) else 0.0,
        "dow": target_dow, "is_weekend": 1 if target_dow >= 5 else 0,
        "app_code": _APP_CODES[app], "cat_code": _CAT_CODES[cat],
        "impulsivity": float(hist["impulsivity"].iloc[0]) if "impulsivity" in hist and len(hist) else 0.0,
    }
