"""coach.py — Coach de bienestar digital (Groq LLM).

En el momento crítico (cerca o por encima del límite) no pone un muro: genera
una intervención breve, empática y personalizada según el contexto del usuario
(app, exceso, riesgo de atracón, triggers, objetivo). Tres modos:
  • reflect   → una pregunta de reflexión (estilo TCC)
  • negotiate → concede una prórroga "consciente" pidiendo un motivo/compromiso
  • block     → cierre firme pero amable, con micro-acción alternativa
"""
from __future__ import annotations

import json
import re

import httpx

from config import GROQ_API_KEY, GROQ_MODELS, GROQ_URL

SYSTEM = (
    "Eres un coach de bienestar digital empático y directo, formado en hábitos y "
    "terapia cognitivo-conductual. Hablas en español, en segunda persona, sin "
    "culpabilizar ni sermonear. Eres BREVE (máximo 2-3 frases por campo). No "
    "moralizas: ayudas a la persona a decidir con conciencia. Devuelves SOLO un "
    "objeto JSON con las claves: message, question, micro_action. Sin markdown."
)

MODE_HINT = {
    "reflect": "Modo REFLEXIÓN: el mensaje invita a una pausa consciente; 'question' es "
               "una pregunta breve que le haga notar por qué quiere seguir.",
    "negotiate": "Modo NEGOCIACIÓN: concede una prórroga corta (p.ej. 10 min) a cambio de "
                 "un motivo concreto; 'question' pide ese motivo; 'micro_action' propone un límite a la prórroga.",
    "block": "Modo CIERRE: cierra el uso de forma amable y firme; 'micro_action' ofrece una "
             "alternativa concreta de 2 minutos (estirar, agua, respirar, salir del cuarto).",
}


def _groq(system: str, user: str, max_tokens: int = 500) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY no configurada")
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    last = ""
    for model in GROQ_MODELS:
        try:
            r = httpx.post(GROQ_URL, headers=headers, timeout=30, json={
                "model": model,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "temperature": 0.6, "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            })
            if r.status_code == 429:
                last = f"{model} 429"; continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            last = f"{model}: {type(e).__name__}"; continue
    raise RuntimeError(f"Groq no disponible. Último: {last}")


def intervene(*, app: str, used_min: float, limit_min: float, mode: str,
              binge_level: str, triggers: list[str], goal_reduction: float,
              persona: str = "") -> dict:
    over = used_min - limit_min
    pct = (used_min / limit_min * 100) if limit_min > 0 else 100
    trig = ", ".join(triggers[:3]) if triggers else "ninguno destacado"
    ctx = (
        f"App: {app}. Uso hoy: {used_min:.0f} min sobre un límite de {limit_min:.0f} min "
        f"({pct:.0f}% del límite, {'+' if over>=0 else ''}{over:.0f} min). "
        f"Riesgo de atracón: {binge_level}. Triggers detectados: {trig}. "
        f"Objetivo del usuario: reducir su ocio digital un {int(goal_reduction*100)}%. "
        f"{('Perfil: ' + persona + '. ') if persona else ''}"
        f"{MODE_HINT.get(mode, MODE_HINT['reflect'])}"
    )
    raw = _groq(SYSTEM, ctx)
    return _parse(raw, mode)


def _parse(raw: str, mode: str) -> dict:
    txt = re.sub(r"```(?:json)?", "", raw or "").replace("```", "").strip()
    try:
        d = json.loads(txt)
    except json.JSONDecodeError:
        s, e = txt.find("{"), txt.rfind("}")
        d = json.loads(txt[s:e + 1]) if s != -1 and e != -1 else {}
    return {
        "mode": mode,
        "message": str(d.get("message", "")).strip(),
        "question": str(d.get("question", "")).strip(),
        "micro_action": str(d.get("micro_action", "")).strip(),
    }
