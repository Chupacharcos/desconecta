"""router.py — Endpoints de Desconecta (cerebro de bienestar digital).

Consumidos por la demo web (proxy Laravel) y por la app Android.
  GET  /demo/desconecta/meta        → métricas de los modelos (transparencia)
  GET  /demo/desconecta/personas    → perfiles de demostración
  POST /demo/desconecta/analyze     → forecast + time-to-limit + riesgo+triggers + límite adaptativo
  POST /demo/desconecta/coach       → intervención del coach LLM
"""
from __future__ import annotations

import json
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import coach as coach_mod
import engine
import personas as personas_mod
from config import APPS, META_PATH, OCIO_CATS

router = APIRouter()

DOW_ES = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]


class AnalyzeReq(BaseModel):
    persona: str = Field(..., description="clave de persona")
    target_dow: int | None = Field(None, ge=0, le=6)
    goal_reduction: float | None = Field(None, ge=0.05, le=0.6)


class UsageRow(BaseModel):
    day: int
    app: str
    minutes: float = Field(..., ge=0)


class AnalyzeCustomReq(BaseModel):
    """Uso real enviado por un cliente (p.ej. la app Android desde UsageStatsManager)."""
    usage: list[UsageRow] = Field(..., min_length=1)
    target_dow: int = Field(..., ge=0, le=6)
    goal_reduction: float = Field(0.3, ge=0.05, le=0.6)
    limits: dict[str, float] | None = None


class CoachReq(BaseModel):
    app: str
    used_min: float
    limit_min: float
    mode: str = "reflect"
    binge_level: str = "medio"
    triggers: list[str] = []
    goal_reduction: float = 0.3
    persona: str = ""


@router.get("/demo/desconecta/meta")
def meta():
    if not META_PATH.exists():
        raise HTTPException(503, "Modelos no entrenados")
    m = json.loads(META_PATH.read_text())
    return {"forecast": m["forecast"], "binge": m["binge"], "apps": APPS}


@router.get("/demo/desconecta/personas")
def personas():
    return {"personas": personas_mod.list_personas()}


@router.post("/demo/desconecta/analyze")
def analyze(req: AnalyzeReq):
    if not engine.ready():
        raise HTTPException(503, "Modelos no entrenados")
    try:
        hist = personas_mod.get_history(req.persona)
    except KeyError:
        raise HTTPException(404, f"Persona desconocida: {req.persona}")

    p = personas_mod.PERSONAS[req.persona]
    goal = req.goal_reduction if req.goal_reduction is not None else p["goal_reduction"]
    target_dow = req.target_dow if req.target_dow is not None else int((hist["day"].max() + 1) % 7)

    fc = engine.forecast_user(hist, target_dow)
    limits = personas_mod.default_limits(hist)
    br = engine.binge_risk(hist, target_dow)
    al = engine.adaptive_limit(hist, goal_reduction=goal)

    apps_out = []
    for app, cat in APPS.items():
        pred = fc["per_app"][app]
        lim = limits.get(app)
        ttl = engine.time_to_limit(app, pred, lim) if lim else {"crosses": False, "label": "Sin límite"}
        apps_out.append({
            "app": app, "category": cat, "is_ocio": cat in OCIO_CATS,
            "forecast_min": pred, "limit_min": lim,
            "crosses": ttl["crosses"], "cross_time": ttl["label"],
            "adaptive_limit_min": al["per_app"].get(app),
        })

    return {
        "persona": {"key": req.persona, **{k: p[k] for k in ("nombre", "descripcion")}},
        "target_dow": target_dow, "target_dow_name": DOW_ES[target_dow],
        "forecast": {"total_min": fc["total"], "ocio_total_min": fc["ocio_total"]},
        "binge": br,
        "adaptive": al,
        "goal_reduction": goal,
        "apps": apps_out,
    }


@router.post("/demo/desconecta/analyze_custom")
def analyze_custom(req: AnalyzeCustomReq):
    """Mismo análisis que /analyze pero sobre uso REAL enviado por el cliente.
    Es el endpoint que consume la app Android (UsageStatsManager → este cerebro)."""
    if not engine.ready():
        raise HTTPException(503, "Modelos no entrenados")
    import pandas as pd
    rows = []
    for u in req.usage:
        if u.app not in APPS:
            continue
        rows.append({"day": u.day, "app": u.app, "minutes": float(u.minutes),
                     "category": APPS[u.app], "impulsivity": 0.0})
    if not rows:
        raise HTTPException(400, "No hay filas de uso con apps reconocidas.")
    hist = pd.DataFrame(rows)

    fc = engine.forecast_user(hist, req.target_dow)
    br = engine.binge_risk(hist, req.target_dow)
    al = engine.adaptive_limit(hist, goal_reduction=req.goal_reduction)

    apps_out = []
    for app, cat in APPS.items():
        pred = fc["per_app"].get(app, 0.0)
        lim = (req.limits or {}).get(app)
        ttl = engine.time_to_limit(app, pred, lim) if lim else {"crosses": False, "label": "Sin límite"}
        apps_out.append({
            "app": app, "category": cat, "is_ocio": cat in OCIO_CATS,
            "forecast_min": pred, "limit_min": lim,
            "crosses": ttl["crosses"], "cross_time": ttl["label"],
            "adaptive_limit_min": al["per_app"].get(app),
        })

    return {
        "target_dow": req.target_dow, "target_dow_name": DOW_ES[req.target_dow],
        "forecast": {"total_min": fc["total"], "ocio_total_min": fc["ocio_total"]},
        "binge": br, "adaptive": al, "goal_reduction": req.goal_reduction, "apps": apps_out,
    }


@router.post("/demo/desconecta/coach")
def coach(req: CoachReq):
    if req.mode not in ("reflect", "negotiate", "block"):
        raise HTTPException(400, "mode debe ser reflect | negotiate | block")
    try:
        out = coach_mod.intervene(
            app=req.app, used_min=req.used_min, limit_min=req.limit_min, mode=req.mode,
            binge_level=req.binge_level, triggers=req.triggers,
            goal_reduction=req.goal_reduction, persona=req.persona,
        )
        return out
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"Coach no disponible: {e}")
