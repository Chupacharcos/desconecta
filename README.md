# Desconecta — Bienestar digital con IA (backend)

El "cerebro" de IA del proyecto **Desconecta**, compartido por la
[demo web](https://adrianmoreno-dev.com/demo/desconecta) y la
[app Android](https://github.com/Chupacharcos/desconecta-android).

Los bloqueadores de pantalla son reactivos, estáticos y fáciles de saltarse.
Desconecta es la capa de inteligencia que les falta:

- **Predictivo** — forecasting de uso (XGBoost) + hora estimada de cruce del límite.
- **Riesgo explicable** — score de atracón (XGBoost) con triggers vía **SHAP**.
- **Adaptativo** — límite con glide-path sostenible hacia el objetivo (no un número fijo).
- **Coach LLM** — interviene (reflexión / negociación / cierre) en vez de un muro.

- **Puerto:** 8009 — `desconecta.service` (lazy)
- **Stack:** Python · FastAPI · XGBoost · SHAP · scikit-learn · pandas · Groq (Llama 3.3 70B)

## Métricas (honestas, datos sintéticos calibrados)

| Modelo | Métrica | Valor |
|--------|---------|-------|
| Forecast de uso (por app) | R² | **0.77** (baseline naïve 0.57) |
| Forecast de uso (total diario) | R² | 0.73 |
| Forecast | MAE / MAPE | 15 min / 25% |
| Riesgo de atracón | AUC / PR-AUC | **0.79** / 0.49 (prevalencia 0.20) |

El forecasting conductual es ruidoso por naturaleza: el uso de una app varía
mucho día a día. Estas cifras son honestas; **no se han limpiado los datos para
inflar métricas**. El atracón es un evento probabilístico, por eso se evalúa
como score de riesgo (AUC), no como clasificación dura.

## Pipeline

```
synth.py   → datos de uso sintéticos con estructura causal del atracón
features.py→ features de forecast (lags, medias móviles) y de riesgo (momentum, nocturno…)
train.py   → entrena XGBoost (forecast + riesgo), split temporal, guarda artifacts/
engine.py  → inferencia: forecast, time-to-limit, riesgo+SHAP, límite adaptativo
coach.py   → coach LLM (Groq) en 3 modos
personas.py→ perfiles de demostración (doomscroller, gamer, remoto, productivo)
router.py  → endpoints HTTP   ·   api.py → app FastAPI
```

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET  | `/health` | healthcheck |
| GET  | `/demo/desconecta/meta` | métricas de los modelos |
| GET  | `/demo/desconecta/personas` | perfiles de demostración |
| POST | `/demo/desconecta/analyze` | análisis de una persona (demo web) |
| POST | `/demo/desconecta/analyze_custom` | análisis sobre uso real (app Android) |
| POST | `/demo/desconecta/coach` | intervención del coach LLM |

## Entrenar / ejecutar

```bash
cd /var/www/desconecta
/var/www/chatbot/venv/bin/python train.py          # genera artifacts/
/var/www/chatbot/venv/bin/uvicorn api:app --host 127.0.0.1 --port 8009
```

Requiere `GROQ_API_KEY` en el entorno o en `/var/www/neuralops/.env` (solo para el coach).
