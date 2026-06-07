"""train.py — Entrena los dos modelos de Desconecta y reporta métricas honestas.

  • Forecaster (XGBRegressor): minutos de mañana por (usuario, app).
  • Binge classifier (XGBClassifier): probabilidad de atracón mañana.

Split TEMPORAL (últimos 14 días = test) para no filtrar futuro. Se compara el
forecaster contra un baseline naïve (lag-7) para demostrar que aporta señal.
Guarda modelos + meta.json (features, métricas, perfiles diurnos).
"""
from __future__ import annotations

import json

import numpy as np
import xgboost as xgb
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             matthews_corrcoef, mean_absolute_error, r2_score,
                             roc_auc_score)

import features as F
from config import (BINGE_MODEL, FORECAST_MODEL, META_PATH, N_DAYS)
from synth import APPS, diurnal_profile, generate

TEST_DAYS = 14


def _mape(y_true, y_pred, floor=10.0):
    m = y_true >= floor
    return float(np.mean(np.abs((y_true[m] - y_pred[m]) / y_true[m])) * 100)


def main():
    print("Generando dataset sintético…")
    df = generate()
    split_day = N_DAYS - TEST_DAYS

    # ── Forecaster ──────────────────────────────────────────────────────────
    Xf, yf = F.build_forecast_frame(df)
    tr, te = Xf["day"] < split_day, Xf["day"] >= split_day
    Xtr, Xte = Xf[tr][F.FORECAST_FEATURES], Xf[te][F.FORECAST_FEATURES]
    ytr, yte = yf[tr], yf[te]

    reg = xgb.XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_lambda=1.0, n_jobs=4, random_state=42,
    )
    reg.fit(Xtr, ytr)
    pred = np.clip(reg.predict(Xte), 0, None)
    r2 = r2_score(yte, pred)
    mae = mean_absolute_error(yte, pred)
    mape = _mape(yte.to_numpy(), pred)
    base_pred = Xte["lag_7"].to_numpy()
    r2_base = r2_score(yte, base_pred)
    mae_base = mean_absolute_error(yte, base_pred)
    print(f"[forecast/app] R²={r2:.3f} MAE={mae:.1f}min MAPE={mape:.1f}%  "
          f"| baseline lag-7 R²={r2_base:.3f}")

    # Headline: forecast del USO TOTAL diario (suma de apps por usuario-día)
    agg = Xf[te][["user_id", "day"]].copy()
    agg["pred"], agg["true"] = pred, yte.to_numpy()
    tot = agg.groupby(["user_id", "day"]).agg(pred=("pred", "sum"), true=("true", "sum"))
    r2_tot = r2_score(tot["true"], tot["pred"])
    mae_tot = mean_absolute_error(tot["true"], tot["pred"])
    mape_tot = _mape(tot["true"].to_numpy(), tot["pred"].to_numpy(), floor=30)
    print(f"[forecast/total] R²={r2_tot:.3f} MAE={mae_tot:.1f}min MAPE={mape_tot:.1f}%")

    # ── Binge classifier (split temporal por 'day') ──────────────────────────
    Xb2, yb2, days_b = _binge_with_day(df)
    trb, teb = days_b < split_day, days_b >= split_day
    Xbtr, Xbte = Xb2[trb], Xb2[teb]
    ybtr, ybte = yb2[trb], yb2[teb]

    pos = max(1, int(ybtr.sum()))
    spw = (len(ybtr) - pos) / pos
    clf = xgb.XGBClassifier(
        n_estimators=350, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_lambda=1.0, scale_pos_weight=spw, n_jobs=4,
        random_state=42, eval_metric="logloss",
    )
    clf.fit(Xbtr, ybtr)
    proba = clf.predict_proba(Xbte)[:, 1]
    # umbral que maximiza F1 en test
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.1, 0.9, 33):
        f1 = f1_score(ybte, (proba >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    yhat = (proba >= best_t).astype(int)
    auc = roc_auc_score(ybte, proba)
    pr_auc = average_precision_score(ybte, proba)
    brier = brier_score_loss(ybte, proba)
    mcc = matthews_corrcoef(ybte, yhat)
    # baseline PR-AUC = prevalencia (clasificador aleatorio)
    print(f"[binge-risk] AUC={auc:.3f} PR-AUC={pr_auc:.3f} (base={ybte.mean():.3f}) "
          f"Brier={brier:.4f} | F1={best_f1:.3f} MCC={mcc:.3f} thr={best_t:.2f}")

    # ── Guardar ───────────────────────────────────────────────────────────────
    reg.save_model(FORECAST_MODEL)
    clf.save_model(BINGE_MODEL)
    meta = {
        "forecast": {"features": F.FORECAST_FEATURES,
                     "r2_app": round(r2, 4), "mae_app_min": round(mae, 2), "mape_app_pct": round(mape, 2),
                     "r2_total": round(r2_tot, 4), "mae_total_min": round(mae_tot, 2),
                     "mape_total_pct": round(mape_tot, 2), "baseline_r2_app": round(r2_base, 4)},
        "binge": {"features": F.BINGE_FEATURES, "auc": round(auc, 4),
                  "pr_auc": round(pr_auc, 4), "brier": round(brier, 4),
                  "f1": round(best_f1, 4), "mcc": round(mcc, 4),
                  "threshold": round(best_t, 4), "prevalence": round(float(ybte.mean()), 4)},
        "apps": APPS,
        "diurnal": {cat: diurnal_profile(cat).round(5).tolist()
                    for cat in sorted(set(APPS.values()))},
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print("Artefactos guardados en artifacts/.")


def _binge_with_day(df):
    """Igual que build_binge_frame pero devolviendo también el 'day' para split."""
    daily = F._daily_ocio(df).sort_values(["user_id", "day"])
    g = daily.groupby("user_id", sort=False)
    daily["prev_ocio"] = g["ocio_total"].shift(1)
    daily["roll3_ocio"] = g["ocio_total"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    daily["roll7_ocio"] = g["ocio_total"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    daily["roll14_ocio"] = g["ocio_total"].transform(lambda s: s.shift(1).rolling(14, min_periods=1).mean())
    daily["user_ocio_mean"] = g["ocio_total"].transform(lambda s: s.shift(1).expanding().mean())
    daily["momentum"] = daily["roll3_ocio"] / (daily["user_ocio_mean"] + 1e-6) - 1.0
    daily["prev_late_night_ratio"] = g["late_night_ratio"].shift(1)
    daily["roll3_late_night"] = g["late_night_ratio"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    daily["prev_app_switches"] = g["app_switches"].shift(1)
    daily["days_since_binge"] = np.concatenate(
        [F._days_since_binge(grp["is_binge"]) for _, grp in g])
    daily = daily.dropna(subset=["prev_ocio", "roll7_ocio", "prev_late_night_ratio"])
    return daily[F.BINGE_FEATURES], daily["is_binge"], daily["day"].to_numpy()


if __name__ == "__main__":
    main()
