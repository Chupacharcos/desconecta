"""config.py — Configuración central de Desconecta (bienestar digital con IA).

Backend del "cerebro" compartido por la demo web y la app Android:
forecasting de uso, riesgo de atracón + triggers, límite adaptativo y coach LLM.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# Artefactos del entrenamiento
FORECAST_MODEL = ARTIFACTS / "forecast_xgb.json"
BINGE_MODEL = ARTIFACTS / "binge_xgb.json"
META_PATH = ARTIFACTS / "meta.json"          # columnas, métricas, calibración
DATASET_PATH = ARTIFACTS / "usage.parquet"   # dataset sintético generado

# ── Apps monitorizadas (móvil) y su categoría ───────────────────────────────
# La categoría marca si el uso es "ocio" (limitable) o "productivo".
APPS = {
    "Instagram":    "social",
    "TikTok":       "video",
    "YouTube":      "video",
    "WhatsApp":     "mensajeria",
    "Juegos":       "juegos",
    "Productividad": "productividad",
}
OCIO_CATS = {"social", "video", "juegos"}  # categorías típicamente limitadas

# ── Groq (coach LLM) ────────────────────────────────────────────────────────
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]


def load_groq_key() -> str:
    key = os.getenv("GROQ_API_KEY", "")
    if key:
        return key
    env_path = Path("/var/www/neuralops/.env")
    if env_path.exists():
        try:
            for line in env_path.read_text().splitlines():
                if line.startswith("GROQ_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass
    return ""


GROQ_API_KEY = load_groq_key()

# ── Generación sintética ────────────────────────────────────────────────────
SEED = 42
N_USERS = 600
N_DAYS = 90
SLOTS_PER_DAY = 24  # resolución intradía (1 slot = 1 hora)
