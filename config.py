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
# 2026-09-01: Groq retiró la familia Llama (404 model_not_found) y la demo
# respondía error en cada consulta. Se usan modelos DISTINTOS a los de los
# agentes de NeuralOps (openai/gpt-oss-*): el límite de tokens/minuto es por
# modelo, así que separarlos impide que una tarea de fondo deje sin cuota a un
# visitante. qwen3.6 queda fuera: emite trazas <think> y rompe el JSON.
# 2026-09-22: el segundo eslabón era `groq/compound-mini`, que NO existe en
# el catálogo de la cuenta (404 model_not_found). Desde el barrido del
# 2026-09-01 —cuando Groq retiró la familia Llama— esta cadena no tenía
# respaldo real: al primer 429 del primario, la petición fallaba igual.
# gpt-oss-20b sí existe y es el modelo del chatbot del portfolio: otra demo,
# no un agente de fondo, así que en el peor caso dos demos comparten un
# minuto. Lo vigila neuralops/tests/test_llm_aislamiento.py.
GROQ_MODELS = ["qwen/qwen3.8-27b", "openai/gpt-oss-20b"]


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
