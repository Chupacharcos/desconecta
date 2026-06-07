"""engine.py — Inferencia del cerebro de Desconecta.

Carga los modelos XGBoost una vez y expone:
  • forecast_user(hist)         → minutos previstos mañana por app + total
  • time_to_limit(app, pred, L) → hora estimada de cruce del límite
  • binge_risk(hist)            → prob. de atracón + triggers (SHAP, legibles)
  • adaptive_limit(hist, goal)  → límite recomendado (glide-path) + explicación
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd
import xgboost as xgb

import features as F
from config import (APPS, BINGE_MODEL, FORECAST_MODEL, META_PATH, OCIO_CATS)

# Etiquetas legibles para los triggers (features del modelo de atracón)
TRIGGER_LABELS = {
    "is_weekend": "Es fin de semana",
    "dow": "Día de la semana",
    "prev_ocio": "Uso de ocio de ayer",
    "roll3_ocio": "Uso de ocio últimos 3 días",
    "roll7_ocio": "Uso de ocio última semana",
    "roll14_ocio": "Tendencia de ocio (2 semanas)",
    "momentum": "Racha de uso al alza",
    "prev_late_night_ratio": "Uso nocturno de ayer",
    "roll3_late_night": "Uso nocturno reciente",
    "prev_app_switches": "Saltas mucho entre apps",
    "days_since_binge": "Tiempo desde el último atracón",
    "impulsivity": "Perfil impulsivo",
}


@lru_cache(maxsize=1)
def _load():
    reg = xgb.XGBRegressor()
    reg.load_model(FORECAST_MODEL)
    clf = xgb.XGBClassifier()
    clf.load_model(BINGE_MODEL)
    meta = json.loads(META_PATH.read_text())
    import shap
    explainer = shap.TreeExplainer(clf)
    return reg, clf, meta, explainer


def ready() -> bool:
    return FORECAST_MODEL.exists() and BINGE_MODEL.exists() and META_PATH.exists()


# ── Forecast ──────────────────────────────────────────────────────────────--

def forecast_user(hist: pd.DataFrame, target_dow: int) -> dict:
    """hist: filas (day, app, minutes, category, impulsivity) de UN usuario."""
    reg, _, _, _ = _load()
    rows, apps = [], list(APPS)
    for app in apps:
        rows.append(F.forecast_row_from_history(hist, app, target_dow))
    X = pd.DataFrame(rows)[F.FORECAST_FEATURES]
    pred = np.clip(reg.predict(X), 0, None)
    per_app = {app: round(float(m), 1) for app, m in zip(apps, pred)}
    total = round(float(sum(per_app.values())), 1)
    ocio = round(float(sum(per_app[a] for a in apps if APPS[a] in OCIO_CATS)), 1)
    return {"per_app": per_app, "total": total, "ocio_total": ocio}


# ── Hora de cruce del límite ────────────────────────────────────────────────

def time_to_limit(app: str, predicted_minutes: float, limit_minutes: float) -> dict:
    """Estima la hora del día en que el uso acumulado de `app` cruza el límite,
    proyectando los minutos previstos sobre el perfil diurno de su categoría."""
    _, _, meta, _ = _load()
    cat = APPS[app]
    diurnal = np.array(meta["diurnal"][cat])
    cum = np.cumsum(diurnal) * predicted_minutes  # minutos acumulados al final de cada hora
    if predicted_minutes <= limit_minutes or limit_minutes <= 0:
        return {"crosses": False, "hour": None, "label": "Por debajo del límite"}
    idx = int(np.argmax(cum >= limit_minutes))  # primera hora que cruza
    # interpolación dentro de la hora para los minutos
    prev = cum[idx - 1] if idx > 0 else 0.0
    frac = (limit_minutes - prev) / max(diurnal[idx] * predicted_minutes, 1e-6)
    clock = idx + float(np.clip(frac, 0, 1))
    hh = int(clock) % 24
    mm = int(round((clock - int(clock)) * 60))
    return {"crosses": True, "hour": round(clock, 2), "label": f"{hh:02d}:{mm:02d}"}


# ── Riesgo de atracón + triggers (SHAP) ──────────────────────────────────────

def _binge_row_from_history(hist: pd.DataFrame, target_dow: int) -> pd.DataFrame:
    h = hist.copy()
    # Rellenar columnas opcionales con defaults razonables (entrada "custom")
    if "late_night_ratio" not in h:
        h["late_night_ratio"] = 0.3
    if "app_switches" not in h:
        h["app_switches"] = 50.0
    if "is_binge" not in h:
        h["is_binge"] = 0
    h["ocio_min"] = np.where(h["category"].isin(OCIO_CATS), h["minutes"], 0.0)
    daily = h.groupby("day").agg(
        ocio_total=("ocio_min", "sum"),
        late_night_ratio=("late_night_ratio", "first"),
        app_switches=("app_switches", "first"),
        is_binge=("is_binge", "first"),
    ).sort_index()
    ocio = daily["ocio_total"].to_numpy(dtype=float)
    lnr = daily["late_night_ratio"].to_numpy(dtype=float)
    sw = daily["app_switches"].to_numpy(dtype=float)
    binge = daily["is_binge"].to_numpy()
    since = 7.0
    for b in binge:
        since = 0.0 if b == 1 else since + 1.0
    row = {
        "is_weekend": 1 if target_dow >= 5 else 0,
        "dow": target_dow,
        "prev_ocio": float(ocio[-1]) if len(ocio) else 0.0,
        "roll3_ocio": float(ocio[-3:].mean()) if len(ocio) else 0.0,
        "roll7_ocio": float(ocio[-7:].mean()) if len(ocio) else 0.0,
        "roll14_ocio": float(ocio[-14:].mean()) if len(ocio) else 0.0,
        "momentum": float(ocio[-3:].mean() / (ocio.mean() + 1e-6) - 1.0) if len(ocio) else 0.0,
        "prev_late_night_ratio": float(lnr[-1]) if len(lnr) else 0.3,
        "roll3_late_night": float(lnr[-3:].mean()) if len(lnr) else 0.3,
        "prev_app_switches": float(sw[-1]) if len(sw) else 50.0,
        "days_since_binge": since,
        "impulsivity": float(hist["impulsivity"].iloc[0]) if "impulsivity" in hist and len(hist) else 0.0,
    }
    return pd.DataFrame([row])[F.BINGE_FEATURES]


def binge_risk(hist: pd.DataFrame, target_dow: int, top_k: int = 4) -> dict:
    _, clf, meta, explainer = _load()
    X = _binge_row_from_history(hist, target_dow)
    proba = float(clf.predict_proba(X)[:, 1][0])
    sv = explainer.shap_values(X)
    sv = np.array(sv).reshape(-1)
    contribs = sorted(zip(F.BINGE_FEATURES, sv), key=lambda t: t[1], reverse=True)
    triggers = [{"factor": TRIGGER_LABELS.get(f, f), "impact": round(float(v), 3)}
                for f, v in contribs if v > 0][:top_k]
    level = "alto" if proba >= 0.5 else ("medio" if proba >= 0.25 else "bajo")
    return {"probability": round(proba, 3), "level": level, "triggers": triggers}


# ── Límite adaptativo (glide-path transparente) ───────────────────────────────

def adaptive_limit(hist: pd.DataFrame, goal_reduction: float = 0.30,
                   horizon_days: int = 28, max_weekly_cut: float = 0.10) -> dict:
    """Recomienda el presupuesto de ocio para mañana descendiendo de forma
    sostenible hacia el objetivo (reducción `goal_reduction`), sin recortes
    bruscos (≤ max_weekly_cut por semana). Transparente y explicable."""
    h = hist.copy()
    h["ocio_min"] = np.where(h["category"].isin(OCIO_CATS), h["minutes"], 0.0)
    daily_ocio = h.groupby("day")["ocio_min"].sum().to_numpy(dtype=float)
    recent = float(np.mean(daily_ocio[-7:])) if len(daily_ocio) else 0.0
    goal_total = recent * (1 - goal_reduction)

    # glide-path lineal hacia el objetivo, capado al recorte semanal sostenible
    step = (recent - goal_total) / max(horizon_days, 1)
    max_daily_cut = recent * (max_weekly_cut / 7)
    step = min(step, max_daily_cut)
    recommended = max(goal_total, recent - step)

    # reparto proporcional al uso reciente por app de ocio
    per_app_recent = {}
    for app, cat in APPS.items():
        if cat in OCIO_CATS:
            s = h[h["app"] == app]["minutes"].to_numpy()
            per_app_recent[app] = float(np.mean(s[-7:])) if len(s) else 0.0
    tot_recent = sum(per_app_recent.values()) + 1e-6
    per_app_limit = {app: round(recommended * (m / tot_recent), 1)
                     for app, m in per_app_recent.items()}

    return {
        "recent_ocio_avg": round(recent, 1),
        "goal_ocio": round(goal_total, 1),
        "recommended_today": round(recommended, 1),
        "per_app": per_app_limit,
        "explanation": (
            f"Tu media reciente de ocio es {recent:.0f} min/día. Para llegar a tu objetivo "
            f"de {goal_total:.0f} min (-{int(goal_reduction*100)}%) en {horizon_days} días sin "
            f"recortes bruscos (≤{int(max_weekly_cut*100)}%/semana), hoy te proponemos "
            f"{recommended:.0f} min en lugar de un límite fijo arbitrario."
        ),
    }
