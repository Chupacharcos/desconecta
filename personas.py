"""personas.py — Perfiles de demostración.

Genera, de forma determinista, el historial reciente (30 días) de un usuario
representativo de cada arquetipo, para que la demo web tenga datos realistas sin
que el visitante tenga que subir nada. Cada persona trae también un objetivo y
un límite de ocio "actual" (deliberadamente excedido) para que la intervención
del coach tenga sentido.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from config import APPS, OCIO_CATS
from synth import generate

PERSONAS = {
    "doomscroller": {
        "archetype": "social_heavy",
        "nombre": "Estudiante doomscroller",
        "descripcion": "Pasa la tarde-noche enganchado a Instagram y TikTok; le cuesta dejar el móvil antes de dormir.",
        "goal_reduction": 0.30,
    },
    "gamer": {
        "archetype": "gamer",
        "nombre": "Gamer nocturno",
        "descripcion": "Sesiones largas de juego y YouTube por la noche que comen horas de sueño.",
        "goal_reduction": 0.25,
    },
    "remoto": {
        "archetype": "balanced",
        "nombre": "Trabajador remoto",
        "descripcion": "Mezcla productividad y ocio; el problema es el goteo constante de redes durante la jornada.",
        "goal_reduction": 0.20,
    },
    "productivo": {
        "archetype": "productive",
        "nombre": "Perfil productivo",
        "descripcion": "Uso mayormente productivo; quiere afinar el poco ocio que tiene sin obsesionarse.",
        "goal_reduction": 0.15,
    },
}

RECENT_DAYS = 30


@lru_cache(maxsize=1)
def _dataset() -> pd.DataFrame:
    # Dataset pequeño y determinista solo para las personas de demo
    return generate(n_users=60, n_days=60, seed=7)


@lru_cache(maxsize=8)
def get_history(key: str) -> pd.DataFrame:
    if key not in PERSONAS:
        raise KeyError(key)
    arch = PERSONAS[key]["archetype"]
    df = _dataset()
    uid = int(df[df["archetype"] == arch]["user_id"].iloc[0])
    h = df[df["user_id"] == uid].copy()
    max_day = h["day"].max()
    h = h[h["day"] > max_day - RECENT_DAYS]
    return h.reset_index(drop=True)


def default_limits(hist: pd.DataFrame) -> dict:
    """Límite de ocio 'actual' del usuario = ~70% de su media reciente (para que
    esté siendo excedido y la demo muestre intervención)."""
    h = hist.copy()
    out = {}
    for app, cat in APPS.items():
        if cat in OCIO_CATS:
            s = h[h["app"] == app]["minutes"].to_numpy()
            avg = float(s[-7:].mean()) if len(s) else 0.0
            out[app] = round(avg * 0.7, 0)
    return out


def list_personas() -> list[dict]:
    out = []
    for key, p in PERSONAS.items():
        hist = get_history(key)
        recent_ocio = float(
            hist.assign(o=hist["minutes"].where(hist["category"].isin(OCIO_CATS), 0.0))
            .groupby("day")["o"].sum().tail(7).mean()
        )
        out.append({
            "key": key,
            "nombre": p["nombre"],
            "descripcion": p["descripcion"],
            "goal_reduction": p["goal_reduction"],
            "recent_ocio_avg_min": round(recent_ocio, 0),
        })
    return out
