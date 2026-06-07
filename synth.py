"""synth.py — Generador de datos de uso de móvil sintéticos, calibrados y con
estructura CAUSAL para que los modelos aprendan señal real.

Calibración (estadísticas públicas de uso de smartphone): ~3-5 h/día totales,
redes sociales ~2 h, picos de tarde-noche, más ocio en fin de semana.

Estructura causal del atracón (para que SHAP exponga triggers legibles):
  P(binge) sube con  → fin de semana, momentum (uso alto últimos 3 días),
                       uso nocturno del día anterior, impulsividad del usuario.
Cuando hay binge, el uso de ocio se infla y se concentra de noche.

Devuelve un DataFrame largo a nivel (user, day, app) con atributos de día.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import APPS, N_DAYS, N_USERS, OCIO_CATS, SEED

# Baseline de minutos/día por app según arquetipo (calibrado a horas reales)
ARCHETYPES = {
    "social_heavy": {"Instagram": 75, "TikTok": 90, "YouTube": 50, "WhatsApp": 45, "Juegos": 20, "Productividad": 30},
    "gamer":        {"Instagram": 25, "TikTok": 40, "YouTube": 60, "WhatsApp": 30, "Juegos": 150, "Productividad": 25},
    "balanced":     {"Instagram": 35, "TikTok": 30, "YouTube": 40, "WhatsApp": 40, "Juegos": 30, "Productividad": 60},
    "productive":   {"Instagram": 20, "TikTok": 15, "YouTube": 30, "WhatsApp": 35, "Juegos": 10, "Productividad": 120},
}
ARCH_WEIGHTS = {"social_heavy": 0.32, "gamer": 0.20, "balanced": 0.30, "productive": 0.18}

# Factor de fin de semana por categoría
WEEKEND_FACTOR = {"social": 1.35, "video": 1.30, "juegos": 1.45, "mensajeria": 1.05,
                  "productividad": 0.45}

# Perfiles diurnos (24 pesos que suman 1) por categoría
def _diurnal(category: str) -> np.ndarray:
    h = np.arange(24)
    if category in ("social", "video"):
        w = np.exp(-((h - 21) ** 2) / 18) + 0.4 * np.exp(-((h - 14) ** 2) / 20)
    elif category == "juegos":
        w = np.exp(-((h - 22) ** 2) / 14) + 0.3 * np.exp(-((h - 16) ** 2) / 25)
    elif category == "mensajeria":
        w = 0.6 + np.exp(-((h - 13) ** 2) / 40) + 0.5 * np.exp(-((h - 20) ** 2) / 30)
    else:  # productividad
        w = np.exp(-((h - 11) ** 2) / 16) + np.exp(-((h - 16) ** 2) / 18)
    w = np.clip(w, 1e-3, None)
    return w / w.sum()


DIURNAL = {cat: _diurnal(cat) for cat in set(APPS.values())}
NIGHT_SLOTS = list(range(0, 6)) + list(range(22, 24))  # franja "nocturna"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(n_users: int = N_USERS, n_days: int = N_DAYS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cats = list(set(APPS.values()))
    night_mass = {cat: DIURNAL[cat][NIGHT_SLOTS].sum() for cat in cats}

    rows = []
    arch_names = list(ARCHETYPES)
    arch_p = np.array([ARCH_WEIGHTS[a] for a in arch_names])
    arch_p /= arch_p.sum()

    for uid in range(n_users):
        arch = rng.choice(arch_names, p=arch_p)
        impuls = float(np.clip(rng.normal(0.0, 1.0), -2.5, 2.5))  # rasgo de impulsividad
        # baseline individual = arquetipo * factor personal (lognormal)
        pf = rng.lognormal(0.0, 0.25)
        base = {app: ARCHETYPES[arch][app] * pf for app in APPS}

        # estado dinámico
        recent_ocio = np.array([sum(base[a] for a in APPS if APPS[a] in OCIO_CATS)] * 3, float)
        prev_night_ratio = 0.3
        days_since_binge = 7

        # tendencia suave del hábito (mejora o empeora)
        trend_slope = rng.normal(0.0, 0.0025)

        for d in range(n_days):
            dow = d % 7
            is_weekend = 1 if dow >= 5 else 0
            trend = 1.0 + trend_slope * d

            # --- probabilidad de atracón (estructura causal) ---
            momentum = (recent_ocio.mean() / (sum(base[a] for a in APPS if APPS[a] in OCIO_CATS) + 1e-6)) - 1.0
            logit = (-2.0
                     + 0.95 * is_weekend
                     + 1.1 * momentum
                     + 1.1 * (prev_night_ratio - 0.3)
                     + 0.5 * impuls
                     - 0.04 * days_since_binge)
            p_binge = float(np.clip(_sigmoid(np.array(logit)), 0.0, 0.85))
            is_binge = int(rng.random() < p_binge)

            day_ocio_total = 0.0
            night_ocio = 0.0
            per_app_today = {}
            for app, cat in APPS.items():
                wf = WEEKEND_FACTOR[cat] if is_weekend else 1.0
                mins = base[app] * wf * trend * rng.lognormal(0.0, 0.22)
                # el atracón infla ocio y concentra de noche
                nr = night_mass[cat]
                if is_binge and cat in OCIO_CATS:
                    boost = 1.0 + rng.uniform(0.5, 1.2)
                    mins *= boost
                    nr = min(0.95, nr + 0.22)
                mins = float(max(0.0, mins))
                per_app_today[app] = (mins, cat, nr)
                if cat in OCIO_CATS:
                    day_ocio_total += mins
                    night_ocio += mins * nr

            late_night_ratio = float(night_ocio / (day_ocio_total + 1e-6))
            app_switches = int(max(0, rng.normal(40 + 60 * (day_ocio_total / 300), 12)))

            for app, (mins, cat, nr) in per_app_today.items():
                rows.append((uid, arch, round(impuls, 3), d, dow, is_weekend, app, cat,
                             round(mins, 1), round(late_night_ratio, 3), app_switches, is_binge))

            # actualizar estado dinámico
            recent_ocio = np.roll(recent_ocio, -1)
            recent_ocio[-1] = day_ocio_total
            prev_night_ratio = late_night_ratio
            days_since_binge = 0 if is_binge else days_since_binge + 1

    df = pd.DataFrame(rows, columns=[
        "user_id", "archetype", "impulsivity", "day", "dow", "is_weekend",
        "app", "category", "minutes", "late_night_ratio", "app_switches", "is_binge",
    ])
    return df


def diurnal_profile(category: str) -> np.ndarray:
    """Perfil diurno (24 pesos) usado para reconstruir la trayectoria intradía
    y calcular la hora estimada de cruce del límite."""
    return DIURNAL[category]


if __name__ == "__main__":
    df = generate()
    print("filas:", len(df), "| usuarios:", df.user_id.nunique(), "| días:", df.day.nunique())
    print("uso medio diario total (min):",
          round(df.groupby(["user_id", "day"]).minutes.sum().mean(), 1))
    print("tasa de días con atracón:", round(df.drop_duplicates(['user_id','day']).is_binge.mean(), 3))
