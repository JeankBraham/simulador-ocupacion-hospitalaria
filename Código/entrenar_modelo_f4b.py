"""
entrenar_modelo_f4b.py
======================
Simulador Inteligente de Ocupacion Hospitalaria
Fase 4-B — Entrenamiento y evaluacion del modelo predictivo

Autor  : Juan Camilo Garcia Braham
Curso  : IA en Salud · Maestria IA y CD · UTP
Anno   : 2026
Stack  : Python 3.12 · numpy 2.4 · pandas 3.0 · scikit-learn 1.8 · joblib 1.5

Decisiones de diseno activas
------------------------------
D_F4B_006  Baseline: LinearRegression sin regularizacion
D_F4B_007  Alternativo: RandomForestRegressor(n_estimators=100, random_state=42)
D_F4B_008  Stack ampliado con scikit-learn 1.8 y joblib 1.5
D_F4B_009  Hiperparametros de RF fijos sin busqueda exhaustiva (PMV academico)
D_F3_001   Semilla global SEED=42 — heredada del proyecto

Split: 80% entrenamiento / 20% validacion estratificado por escenario
Metrica principal: RMSE  |  Metrica secundaria: MAE
Criterio de seleccion: menor RMSE; desempate por MAE; empate (<1pp) -> modelo lineal
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # backend sin GUI para entorno headless
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

SEED: int = 42                  # D_F3_001 — semilla global
TEST_SIZE: float = 0.20         # 80/20 split
DATASET_PATH: str = "/home/claude/dataset_f4b.parquet"

# Features definitivas (D_F4B_001..005)
FEATURE_COLS: list[str] = [
    "O_t", "E_t", "P_t", "C_t", "I_t",
    "llegadas_t", "altas_t", "traslados_t", "cola_espera_t",
    "O_lag1", "O_lag2", "O_lag3", "O_lag4",
    "escenario_alta_demanda", "escenario_crisis",
]
TARGET_COL: str = "target_O_t4"

# Colores consistentes con el estilo de reportes anteriores del proyecto
COLOR_LINEAL = "#4A9EFF"     # azul
COLOR_RF     = "#FF6B6B"     # rojo suave
COLOR_IDEAL  = "#95D44A"     # verde
COLOR_FONDO  = "#1E1E2E"
COLOR_TEXTO  = "#CDD6F4"
COLOR_GRID   = "#313244"


# ─────────────────────────────────────────────────────────────────────────────
# CARGA Y SPLIT DEL DATASET
# ─────────────────────────────────────────────────────────────────────────────

def cargar_y_dividir(ruta: str = DATASET_PATH) -> tuple:
    """Carga el dataset y realiza el split 80/20 estratificado por escenario.

    La estratificacion garantiza que los tres escenarios esten representados
    proporcionalmente en train y test. Retorna:
        X_train, X_test, y_train, y_test, df (dataset completo)
    """
    df = pd.read_parquet(ruta)

    X = df[FEATURE_COLS].values.astype(float)
    y = df[TARGET_COL].values.astype(float)

    # Estratificacion por escenario — garantiza representacion balanceada
    estratos = df["escenario"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=estratos,
    )

    return X_train, X_test, y_train, y_test, df


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCCION DE PIPELINES
# ─────────────────────────────────────────────────────────────────────────────

def construir_pipeline_lineal() -> Pipeline:
    """D_F4B_006 — Pipeline: StandardScaler + LinearRegression.

    El escalado es necesario para que los coeficientes del modelo lineal
    sean comparables entre features con escalas distintas (O_t en [0,100]
    vs llegadas_t en [0,~8] por tick Poisson con lambda=5).
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", LinearRegression()),
    ])


def construir_pipeline_rf() -> Pipeline:
    """D_F4B_007 — Pipeline: RandomForestRegressor.

    RF no requiere escalado (invariante a transformaciones monotónas de features),
    pero se incluye dentro del Pipeline para consistencia de interfaz.
    StandardScaler con with_std=False aplica solo centrado, sin efecto real en RF.
    """
    return Pipeline([
        ("scaler", StandardScaler(with_mean=False, with_std=False)),  # pass-through
        ("modelo", RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            random_state=SEED,   # D_F3_001
            n_jobs=-1,
        )),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO Y METRICAS
# ─────────────────────────────────────────────────────────────────────────────

def calcular_metricas(y_true: np.ndarray, y_pred: np.ndarray,
                      nombre: str) -> dict:
    """Calcula RMSE y MAE. Retorna dict con metricas y nombre del modelo."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    return {
        "modelo": nombre,
        "rmse":   round(rmse, 4),
        "mae":    round(mae,  4),
        "n_test": len(y_true),
    }


def seleccionar_modelo_ganador(m_lineal: dict, m_rf: dict) -> str:
    """D_F4B_006/007 — Criterio de seleccion documentado.

    Regla: menor RMSE. Si |RMSE_lineal - RMSE_rf| < 1.0 pp -> modelo lineal
    (navaja de Occam). Retorna 'lineal' o 'rf'.
    """
    diff = abs(m_lineal["rmse"] - m_rf["rmse"])
    if diff < 1.0:
        return "lineal"   # empate practico -> parsimonia
    return "lineal" if m_lineal["rmse"] < m_rf["rmse"] else "rf"


# ─────────────────────────────────────────────────────────────────────────────
# FIGURAS DE EVALUACION
# ─────────────────────────────────────────────────────────────────────────────

def _estilo_oscuro() -> None:
    """Aplica estilo oscuro consistente con reportes anteriores del proyecto."""
    plt.rcParams.update({
        "figure.facecolor":  COLOR_FONDO,
        "axes.facecolor":    COLOR_FONDO,
        "axes.edgecolor":    COLOR_GRID,
        "axes.labelcolor":   COLOR_TEXTO,
        "xtick.color":       COLOR_TEXTO,
        "ytick.color":       COLOR_TEXTO,
        "text.color":        COLOR_TEXTO,
        "grid.color":        COLOR_GRID,
        "grid.linestyle":    "--",
        "grid.alpha":        0.5,
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.titleweight":  "bold",
    })


def generar_figura_prediccion(
    y_test: np.ndarray,
    pred_lineal: np.ndarray,
    pred_rf: np.ndarray,
    metricas_lineal: dict,
    metricas_rf: dict,
    ganador: str,
    ruta: str = "/home/claude/fig_prediccion_f4b.png",
) -> str:
    """Genera figura 2x2 con scatter y residuos para ambos modelos."""
    _estilo_oscuro()
    fig = plt.figure(figsize=(14, 10), facecolor=COLOR_FONDO)
    fig.suptitle(
        "F4-B — Evaluacion del Modelo Predictivo  (O_{t+4} — horizonte 1 hora)",
        fontsize=13, fontweight="bold", color=COLOR_TEXTO, y=0.98
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    lim = (0, 105)

    # ── Panel 1: Scatter lineal ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_test, pred_lineal, alpha=0.35, s=14,
                color=COLOR_LINEAL, label="predicciones")
    ax1.plot(lim, lim, color=COLOR_IDEAL, lw=1.4, ls="--", label="ideal")
    ax1.set_xlim(lim); ax1.set_ylim(lim)
    ax1.set_xlabel("O real (%)"); ax1.set_ylabel("O predicho (%)")
    marca = " [GANADOR]" if ganador == "lineal" else ""
    ax1.set_title(
        f"Regresion Lineal{marca}\n"
        f"RMSE={metricas_lineal['rmse']:.2f}  MAE={metricas_lineal['mae']:.2f}"
    )
    ax1.legend(fontsize=8)
    ax1.grid(True)

    # ── Panel 2: Scatter RF ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(y_test, pred_rf, alpha=0.35, s=14,
                color=COLOR_RF, label="predicciones")
    ax2.plot(lim, lim, color=COLOR_IDEAL, lw=1.4, ls="--", label="ideal")
    ax2.set_xlim(lim); ax2.set_ylim(lim)
    ax2.set_xlabel("O real (%)"); ax2.set_ylabel("O predicho (%)")
    marca = " [GANADOR]" if ganador == "rf" else ""
    ax2.set_title(
        f"Random Forest{marca}\n"
        f"RMSE={metricas_rf['rmse']:.2f}  MAE={metricas_rf['mae']:.2f}"
    )
    ax2.legend(fontsize=8)
    ax2.grid(True)

    # ── Panel 3: Residuos lineal ──────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    residuos_lineal = y_test - pred_lineal
    ax3.scatter(pred_lineal, residuos_lineal, alpha=0.35, s=14, color=COLOR_LINEAL)
    ax3.axhline(0, color=COLOR_IDEAL, lw=1.4, ls="--")
    ax3.set_xlabel("O predicho (%)"); ax3.set_ylabel("Residuo (pp)")
    ax3.set_title("Residuos — Regresion Lineal")
    ax3.grid(True)

    # ── Panel 4: Residuos RF ──────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    residuos_rf = y_test - pred_rf
    ax4.scatter(pred_rf, residuos_rf, alpha=0.35, s=14, color=COLOR_RF)
    ax4.axhline(0, color=COLOR_IDEAL, lw=1.4, ls="--")
    ax4.set_xlabel("O predicho (%)"); ax4.set_ylabel("Residuo (pp)")
    ax4.set_title("Residuos — Random Forest")
    ax4.grid(True)

    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor=COLOR_FONDO)
    plt.close(fig)
    return ruta


def generar_figura_importancia(
    pipeline_rf: Pipeline,
    ruta: str = "/home/claude/fig_importancia_f4b.png",
) -> str:
    """Genera figura de importancia de features del Random Forest."""
    _estilo_oscuro()
    modelo_rf = pipeline_rf.named_steps["modelo"]
    importancias = modelo_rf.feature_importances_
    indices = np.argsort(importancias)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLOR_FONDO)
    ax.set_facecolor(COLOR_FONDO)

    colores = [COLOR_RF if FEATURE_COLS[i].startswith("O_lag") or
               FEATURE_COLS[i] == "O_t" else COLOR_LINEAL
               for i in indices]

    bars = ax.barh(
        [FEATURE_COLS[i] for i in indices[::-1]],
        importancias[indices[::-1]],
        color=colores[::-1],
        edgecolor="none",
        height=0.65,
    )
    ax.set_xlabel("Importancia (Gini)", color=COLOR_TEXTO)
    ax.set_title(
        "Importancia de Features — Random Forest\n"
        "Azul: componentes del indicador I y llegadas  |  Rojo: lags de O",
        color=COLOR_TEXTO, fontsize=11
    )
    ax.grid(True, axis="x")
    ax.tick_params(colors=COLOR_TEXTO)

    for bar, imp in zip(bars[::-1], importancias[indices]):
        ax.text(imp + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{imp:.3f}", va="center", fontsize=8, color=COLOR_TEXTO)

    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor=COLOR_FONDO)
    plt.close(fig)
    return ruta


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def entrenar_y_evaluar(
    dataset_path: str = DATASET_PATH,
    directorio_salida: str = "/home/claude",
    verbose: bool = True,
) -> dict:
    """Ejecuta el pipeline completo de entrenamiento y evaluacion.

    Retorna un dict con metricas, rutas de artefactos y modelo seleccionado.
    """
    if verbose:
        print("=" * 60)
        print("F4-B — ENTRENAMIENTO Y EVALUACION DEL MODELO PREDICTIVO")
        print(f"Dataset: {dataset_path}")
        print(f"Semilla: SEED={SEED}  |  Split: {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}")
        print("=" * 60)

    # ── 1. Cargar y dividir ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, df = cargar_y_dividir(dataset_path)
    if verbose:
        print(f"\n[1] Dataset cargado: {len(df)} filas totales")
        print(f"    Train: {len(X_train)}  |  Test: {len(X_test)}")
        print(f"    Features: {len(FEATURE_COLS)}")

    # ── 2. Construir pipelines ────────────────────────────────────────────────
    pipe_lineal = construir_pipeline_lineal()
    pipe_rf     = construir_pipeline_rf()

    # ── 3. Entrenar ───────────────────────────────────────────────────────────
    if verbose:
        print("\n[2] Entrenando modelos...")

    pipe_lineal.fit(X_train, y_train)
    if verbose:
        print("    Regresion Lineal: OK")

    pipe_rf.fit(X_train, y_train)
    if verbose:
        print("    Random Forest:    OK")

    # ── 4. Predicciones y metricas ────────────────────────────────────────────
    pred_lineal = pipe_lineal.predict(X_test)
    pred_rf     = pipe_rf.predict(X_test)

    metricas_lineal = calcular_metricas(y_test, pred_lineal, "Regresion Lineal")
    metricas_rf     = calcular_metricas(y_test, pred_rf,     "Random Forest")

    if verbose:
        print("\n[3] Metricas en set de validacion (20%):")
        print(f"    {'Modelo':<25} {'RMSE':>8} {'MAE':>8}")
        print(f"    {'-'*41}")
        print(f"    {'Regresion Lineal':<25} {metricas_lineal['rmse']:>8.4f} {metricas_lineal['mae']:>8.4f}")
        print(f"    {'Random Forest':<25} {metricas_rf['rmse']:>8.4f} {metricas_rf['mae']:>8.4f}")

    # ── 5. Seleccion del modelo ganador ───────────────────────────────────────
    ganador = seleccionar_modelo_ganador(metricas_lineal, metricas_rf)
    pipe_ganador = pipe_lineal if ganador == "lineal" else pipe_rf
    nombre_ganador = "Regresion Lineal" if ganador == "lineal" else "Random Forest"

    if verbose:
        diff = abs(metricas_lineal["rmse"] - metricas_rf["rmse"])
        print(f"\n[4] Seleccion del modelo:")
        print(f"    Diferencia RMSE: {diff:.4f} pp")
        if diff < 1.0:
            print(f"    Diferencia < 1.0 pp -> parsimonia: GANADOR = Regresion Lineal")
        else:
            print(f"    Diferencia >= 1.0 pp -> GANADOR = {nombre_ganador}")

    # ── 6. Coeficientes / importancias ────────────────────────────────────────
    modelo_lineal_obj = pipe_lineal.named_steps["modelo"]
    coeficientes = {
        feat: round(float(coef), 6)
        for feat, coef in zip(FEATURE_COLS, modelo_lineal_obj.coef_)
    }
    intercepto = round(float(modelo_lineal_obj.intercept_), 6)

    modelo_rf_obj = pipe_rf.named_steps["modelo"]
    importancias = {
        feat: round(float(imp), 6)
        for feat, imp in zip(FEATURE_COLS, modelo_rf_obj.feature_importances_)
    }

    if verbose:
        print("\n[5] Coeficientes Regresion Lineal (top-5 por |valor|):")
        top5 = sorted(coeficientes.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        for feat, coef in top5:
            print(f"    {feat:<25} {coef:>+.4f}")
        print(f"    Intercepto: {intercepto:>+.4f}")

        print("\n[6] Importancia features Random Forest (top-5):")
        top5_rf = sorted(importancias.items(), key=lambda x: x[1], reverse=True)[:5]
        for feat, imp in top5_rf:
            print(f"    {feat:<25} {imp:.4f}")

    # ── 7. Guardar figuras ────────────────────────────────────────────────────
    ruta_fig_pred = os.path.join(directorio_salida, "fig_prediccion_f4b.png")
    ruta_fig_imp  = os.path.join(directorio_salida, "fig_importancia_f4b.png")

    generar_figura_prediccion(
        y_test, pred_lineal, pred_rf,
        metricas_lineal, metricas_rf, ganador,
        ruta=ruta_fig_pred,
    )
    generar_figura_importancia(pipe_rf, ruta=ruta_fig_imp)

    if verbose:
        print(f"\n[7] Figuras guardadas:")
        print(f"    {ruta_fig_pred}")
        print(f"    {ruta_fig_imp}")

    # ── 8. Serializar modelos (.pkl) ──────────────────────────────────────────
    ruta_pkl_lineal  = os.path.join(directorio_salida, "modelo_lineal_f4b.pkl")
    ruta_pkl_rf      = os.path.join(directorio_salida, "modelo_rf_f4b.pkl")
    ruta_pkl_ganador = os.path.join(directorio_salida, "modelo_final_f4b.pkl")

    joblib.dump(pipe_lineal,  ruta_pkl_lineal)
    joblib.dump(pipe_rf,      ruta_pkl_rf)
    joblib.dump(pipe_ganador, ruta_pkl_ganador)

    if verbose:
        print(f"\n[8] Modelos serializados (.pkl):")
        print(f"    {ruta_pkl_lineal}")
        print(f"    {ruta_pkl_rf}")
        print(f"    {ruta_pkl_ganador}  <- modelo final (F6)")

    # ── 9. Guardar metricas en JSON ───────────────────────────────────────────
    ruta_metricas = os.path.join(directorio_salida, "metricas_f4b.json")
    resultado = {
        "modelo_lineal":  metricas_lineal,
        "modelo_rf":      metricas_rf,
        "modelo_ganador": ganador,
        "nombre_ganador": nombre_ganador,
        "coeficientes_lineal": coeficientes,
        "intercepto_lineal":   intercepto,
        "importancias_rf":     importancias,
        "features":       FEATURE_COLS,
        "target":         TARGET_COL,
        "seed":           SEED,
        "test_size":      TEST_SIZE,
        "n_train":        len(X_train),
        "n_test":         len(X_test),
        "rutas": {
            "dataset":        dataset_path,
            "pkl_lineal":     ruta_pkl_lineal,
            "pkl_rf":         ruta_pkl_rf,
            "pkl_ganador":    ruta_pkl_ganador,
            "fig_prediccion": ruta_fig_pred,
            "fig_importancia": ruta_fig_imp,
        }
    }

    with open(ruta_metricas, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"    Metricas JSON: {ruta_metricas}")
        print("\n" + "=" * 60)
        print(f"RESULTADO FINAL: modelo ganador = {nombre_ganador.upper()}")
        print(f"  RMSE = {resultado[f'modelo_{ganador}']['rmse']:.4f} pp")
        print(f"  MAE  = {resultado[f'modelo_{ganador}']['mae']:.4f} pp")
        print("=" * 60)

    return resultado


if __name__ == "__main__":
    resultado = entrenar_y_evaluar(verbose=True)
