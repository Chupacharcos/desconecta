"""api.py — Servicio FastAPI de Desconecta (puerto 8009).

Cerebro de bienestar digital compartido por la demo web y la app Android:
forecasting de uso, riesgo de atracón + triggers (SHAP), límite adaptativo y
coach LLM. Modelos XGBoost pre-entrenados (artifacts/).
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import engine
from router import router

app = FastAPI(
    title="Desconecta — Bienestar digital con IA",
    description="Forecasting de uso, riesgo de atracón + triggers, límite adaptativo y coach LLM.",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok", "service": "desconecta", "port": 8009, "models_ready": engine.ready()}
