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

<!-- LOOP-MAP:START (generado por `php artisan project:loop readme` — no editar a mano) -->

## El bucle que cierra

<p align="center"><img src="https://adrianmoreno-dev.com/bucle/desconecta.svg" alt="Mapa del bucle de Desconecta — Bienestar digital con IA" width="900"></p>

**Para** quien quiere bajar su tiempo de pantalla sin muros · **Cada día**

| Etapa | Qué pasa | Quién |
|---|---|---|
| **1. Disparador** | Vuelvo a pasarme del tiempo de pantalla y el bloqueador solo sabe ponerme un muro. | persona |
| **2. Acción** | Predice el uso del día, estima a qué hora cruzaré el límite y calcula el riesgo de atracón con sus disparadores. | software |
| **3. Medición** | La hora estimada del cruce, el score de riesgo y qué lo está empujando, con SHAP. | software |
| **4. Decisión** | Decido si acepto lo que propone el coach, con el límite bajando poco a poco. | persona |

### Lo que no hace

- No bloquea nada: es la capa que decide cuándo merece la pena intervenir, no el muro.
- Las métricas salen de datos sintéticos calibrados, no de usuarios reales.
- No mide el uso por su cuenta: recibe los datos de la app que lo integra.

### Por qué está construido así

- **Límite con glide-path** en vez de un límite fijo de minutos — Un número fijo se salta el primer día malo y ya no vuelves. Bajar poco a poco hacia el objetivo aguanta el mes.
- **Explicar el riesgo con SHAP** en vez de dar solo el número del score — Un riesgo sin motivo no cambia el comportamiento. Saber qué lo empuja permite actuar sobre esa app o esa hora.

<!-- LOOP-MAP:END -->

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
