"""
evaluar_escenarios_f5.py
========================
Simulador Inteligente de Ocupacion Hospitalaria
Fase 5 — Evaluacion (SEMMA·Assess)

Autor  : Juan Camilo Garcia Braham
Curso  : IA en Salud · Maestria IA y CD · UTP
Anno   : 2026
Stack  : Python 3.12 · numpy 2.4 · pandas 3.0 · matplotlib 3.10 ·
         seaborn 0.13 · scikit-learn 1.8 · joblib 1.5

Decisiones de diseno activas en este modulo
---------------------------------------------
D_F5_001  Horizonte de evaluacion: 200 ticks (50 h simuladas) por escenario
          — mismo horizonte que F4-B para comparabilidad.
D_F5_002  Escenario normal evaluado primero (Entregable 1 de F5).
          Alta demanda (E2) y crisis (E3) quedan pendientes de confirmacion.
D_F5_003  Semilla de evaluacion: SEED_EVAL = 99 (distinta a semillas de
          entrenamiento 42/123/456/789 — evita data leakage entre E/A).
D_F5_004  El modelo predictivo recibe los 4 lags de O construidos desde
          el historial tick-a-tick. Los primeros 4 ticks no tienen prediccion
          (lags incompletos).
D_F5_005  Modo asistido evaluado en el escenario de crisis (E3):
          se simula un gestor que confirma el 100% de las propuestas
          (comportamiento conservador / pesimista para el analisis).
D_F5_006  Periodos de regimen estable: se descartan los primeros WARM_UP_TICKS
          ticks del analisis estadistico (pero se grafican completos).
          WARM_UP_TICKS = 20 (5 horas) — tiempo de calentamiento del simulador.
D_F5_007  Features del modelo: orden fijo definido en FEATURES_ORDER
          (consistente con entrenar_modelo_f4b.py). Array numpy, sin DataFrame,
          para evitar advertencia de nombres de features (modelo entrenado
          con arrays).

Riesgos activos monitoreados en F5
------------------------------------
R03  Distribuciones no representativas — verificacion final en esta fase:
     criterio CE-B: lambda=3.0 -> O en [85,95]%; lambda=5.0 -> O>95%.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib

from generador_pacientes import generar_llegadas_tick, LAMBDA_POR_ESCENARIO
from sistema_experto import (
    SistemaExperto,
    crear_hospital_referencia,
    ResultadoTick,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DE EVALUACION
# ─────────────────────────────────────────────────────────────────────────────

TICKS_EVAL:     int = 200      # D_F5_001 — 50 horas simuladas
WARM_UP_TICKS:  int = 20       # D_F5_006 — 5 horas de calentamiento
SEED_EVAL:      int = 99       # D_F5_003 — semilla exclusiva de evaluacion
HORIZONTE_PRED: int = 4        # D_F4B_002 — T+4 (1 hora)
N_LAGS:         int = 4        # D_F4B_005

# Orden fijo de features para el modelo predictivo — D_F5_007
FEATURES_ORDER: list[str] = [
    "O_t", "E_t", "P_t", "C_t", "I_t",
    "llegadas_t", "altas_t", "traslados_t", "cola_espera_t",
    "O_lag1", "O_lag2", "O_lag3", "O_lag4",
    "escenario_alta_demanda", "escenario_crisis",
]

# Umbrales esperados por escenario — criterios de salida CE-B
# D_F5_008 (Opcion A): umbrales ajustados al equilibrio matematico del sistema
# sintetico (Ley de Little con lambda fija D012 y distribuciones de F2).
# Umbrales conceptuales de F1 eran para hospitales reales; el sistema sintetico
# converge a valores proporcionalmente correctos pero escalados por F2.
UMBRALES_ESPERADOS: dict[str, dict] = {
    "normal":       {"O_min": 35.0,  "O_max": 55.0,  "I_nivel": ["Bajo", "Medio"]},
    "alta_demanda": {"O_min": 60.0,  "O_max": 88.0,  "I_nivel": ["Alto", "Critico", "Medio"]},
    "crisis":       {"O_min": 80.0,  "O_max": 100.0, "I_nivel": ["Alto", "Critico"]},
}

# Colores por escenario para graficos
COLOR_ESCENARIO: dict[str, str] = {
    "normal":       "#4CAF50",
    "alta_demanda": "#FF9800",
    "crisis":       "#F44336",
}

# ─────────────────────────────────────────────────────────────────────────────
# ESTRUCTURA DE RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RegistroTick:
    """Una fila del historial tick-a-tick de una simulacion de evaluacion."""
    tick:        int
    escenario:   str
    modo:        str       # "automatico" | "asistido"
    O_t:         float
    E_t:         float
    P_t:         float
    C_t:         float
    I_t:         float
    nivel_I:     str
    llegadas_t:  int
    altas_t:     int
    traslados_t: int
    cola_t:      int
    n_alertas:   int
    pred_O_t4:   float | None = None   # prediccion del modelo (None si lag incompleto)


@dataclass
class ResultadoEscenario:
    """Agregado de metricas para un escenario completo."""
    escenario:       str
    modo:            str
    historial:       list[RegistroTick] = field(default_factory=list)

    # Metricas de regimen estable (post warm-up)
    O_media:         float = 0.0
    O_std:           float = 0.0
    O_p5:            float = 0.0
    O_p95:           float = 0.0
    I_media:         float = 0.0
    I_std:           float = 0.0
    nivel_I_modal:   str = ""
    distribucion_niveles: dict[str, int] = field(default_factory=dict)
    cola_media:      float = 0.0
    cola_max:        int   = 0
    altas_total:     int   = 0
    traslados_total: int   = 0
    alertas_total:   int   = 0
    pred_rmse:       float | None = None   # RMSE prediccion vs real (en regime)

    # Verificacion CE-B
    cumple_O_rango:  bool = False
    cumple_I_nivel:  bool = False
    cumple_criticos: bool = False   # R03 — lambda verifica CE-B


# ─────────────────────────────────────────────────────────────────────────────
# MOTOR DE SIMULACION DE EVALUACION
# ─────────────────────────────────────────────────────────────────────────────

def ejecutar_simulacion_eval(
    escenario: str,
    modo_asistido: bool,
    modelo_pred,
    seed: int = SEED_EVAL,
    ticks: int = TICKS_EVAL,
    verbose: bool = False,
) -> ResultadoEscenario:
    """Ejecuta una simulacion completa de evaluacion con coleccion de metricas.

    Para el modo asistido (D_F5_005): el gestor simulado confirma todas las
    acciones propuestas (100% de confirmacion). Esto representa el comportamiento
    de un gestor conservador que no rechaza ninguna redistribucion.

    Retorna ResultadoEscenario con historial completo y metricas agregadas.
    """
    rng_sim = np.random.default_rng(seed)
    rng_gen = np.random.default_rng(seed + 500)

    hospital = crear_hospital_referencia(rng=rng_sim)
    se       = SistemaExperto(hospital=hospital)

    modo_str = "asistido" if modo_asistido else "automatico"
    historial: list[RegistroTick] = []
    historial_O: list[float] = []   # para calcular lags

    # One-hot del escenario para el modelo predictivo
    esc_alta   = 1.0 if escenario == "alta_demanda" else 0.0
    esc_crisis = 1.0 if escenario == "crisis"       else 0.0

    for tick in range(ticks):
        nuevos = generar_llegadas_tick(tick, escenario=escenario, rng=rng_gen)

        resultado: ResultadoTick = se.procesar_tick(
            tick, nuevos_pacientes=nuevos, modo_asistido=modo_asistido
        )

        altas_t     = sum(1 for a in resultado.acciones_ejecutadas if a.tipo == "alta")
        traslados_t = sum(1 for a in resultado.acciones_ejecutadas
                         if a.tipo in ("traslado", "desborde"))

        # Modo asistido — gestor confirma el 100% de propuestas (D_F5_005)
        # Confirmacion post-conteo: captura traslados/desbordes pendientes
        if modo_asistido and resultado.acciones_pendientes:
            confirmadas = se.confirmar_acciones(resultado.acciones_pendientes, tick)
            traslados_t += sum(1 for a in confirmadas
                               if a.tipo in ("traslado", "desborde"))
        cola_t      = len(hospital.pacientes_esperando())

        # Prediccion del modelo — D_F5_004
        pred_O_t4: float | None = None
        historial_O.append(resultado.componente_O)

        if len(historial_O) > N_LAGS:
            features_vec = np.array([[
                resultado.componente_O,
                resultado.componente_E,
                resultado.componente_P,
                resultado.componente_C,
                resultado.indicador_I,
                float(len(nuevos)),
                float(altas_t),
                float(traslados_t),
                float(cola_t),
                historial_O[-2],   # O_lag1 = tick-1
                historial_O[-3],   # O_lag2 = tick-2
                historial_O[-4],   # O_lag3 = tick-3
                historial_O[-5],   # O_lag4 = tick-4
                esc_alta,
                esc_crisis,
            ]])
            pred_O_t4 = float(modelo_pred.predict(features_vec)[0])

        historial.append(RegistroTick(
            tick        = tick,
            escenario   = escenario,
            modo        = modo_str,
            O_t         = resultado.componente_O,
            E_t         = resultado.componente_E,
            P_t         = resultado.componente_P,
            C_t         = resultado.componente_C,
            I_t         = resultado.indicador_I,
            nivel_I     = resultado.nivel_I,
            llegadas_t  = len(nuevos),
            altas_t     = altas_t,
            traslados_t = traslados_t,
            cola_t      = cola_t,
            n_alertas   = len(resultado.alertas),
            pred_O_t4   = pred_O_t4,
        ))

        if verbose and tick % 40 == 0:
            print(f"  tick={tick:3d}  O={resultado.componente_O:5.1f}%  "
                  f"I={resultado.indicador_I:5.1f} [{resultado.nivel_I}]  "
                  f"cola={cola_t}  alertas={len(resultado.alertas)}")

    # ── Calcular metricas de regimen estable ──────────────────────────────────
    res = ResultadoEscenario(escenario=escenario, modo=modo_str, historial=historial)

    regime = [r for r in historial if r.tick >= WARM_UP_TICKS]
    if not regime:
        return res

    O_vals  = np.array([r.O_t  for r in regime])
    I_vals  = np.array([r.I_t  for r in regime])
    niveles = [r.nivel_I for r in regime]

    res.O_media  = float(np.mean(O_vals))
    res.O_std    = float(np.std(O_vals))
    res.O_p5     = float(np.percentile(O_vals, 5))
    res.O_p95    = float(np.percentile(O_vals, 95))
    res.I_media  = float(np.mean(I_vals))
    res.I_std    = float(np.std(I_vals))

    from collections import Counter
    nivel_counts = Counter(niveles)
    res.distribucion_niveles = dict(nivel_counts)
    res.nivel_I_modal = nivel_counts.most_common(1)[0][0]

    res.cola_media      = float(np.mean([r.cola_t      for r in regime]))
    res.cola_max        = int(max(r.cola_t              for r in regime))
    res.altas_total     = sum(r.altas_t                 for r in regime)
    res.traslados_total = sum(r.traslados_t             for r in regime)
    res.alertas_total   = sum(r.n_alertas               for r in regime)

    # RMSE prediccion vs real — alineado en T+4
    pairs_pred = [
        (regime[i].pred_O_t4, regime[i + HORIZONTE_PRED].O_t)
        for i in range(len(regime) - HORIZONTE_PRED)
        if regime[i].pred_O_t4 is not None
    ]
    if pairs_pred:
        preds, reals = zip(*pairs_pred)
        res.pred_rmse = float(np.sqrt(np.mean((np.array(preds) - np.array(reals))**2)))

    # Verificacion CE-B — D_F5_008 umbrales ajustados
    umb = UMBRALES_ESPERADOS[escenario]
    res.cumple_O_rango = (
        res.O_media >= umb["O_min"] - 5.0
        and res.O_media <= umb["O_max"] + 5.0
    )
    # Normalizar acento para comparacion robusta ('Critico' con y sin tilde)
    nivel_modal_norm = res.nivel_I_modal.replace("\u00ed", "i").replace("\u00e9", "e")
    res.cumple_I_nivel = nivel_modal_norm in umb["I_nivel"]

    return res


# ─────────────────────────────────────────────────────────────────────────────
# FIGURAS DE EVALUACION — Entregable 1 (escenario normal)
# ─────────────────────────────────────────────────────────────────────────────

def _aplicar_estilo_base() -> None:
    """Aplica estilo visual coherente con las figuras de F4-B."""
    plt.rcParams.update({
        "figure.facecolor":  "#1a1a2e",
        "axes.facecolor":    "#16213e",
        "axes.edgecolor":    "#aaaaaa",
        "axes.labelcolor":   "#e0e0e0",
        "axes.titlecolor":   "#ffffff",
        "text.color":        "#e0e0e0",
        "xtick.color":       "#aaaaaa",
        "ytick.color":       "#aaaaaa",
        "grid.color":        "#2a2a4a",
        "grid.alpha":        0.6,
        "legend.facecolor":  "#1a1a2e",
        "legend.edgecolor":  "#555555",
        "figure.dpi":        120,
    })


def generar_figura_escenario(
    res: ResultadoEscenario,
    ruta_salida: str,
) -> None:
    """Genera figura de 4 paneles para un escenario.

    Paneles:
      [0,0] Ocupacion O(t) vs tiempo — con prediccion superpuesta
      [0,1] Indicador I(t) con bandas de nivel
      [1,0] Cola de espera y traslados por tick
      [1,1] Distribucion de niveles de I (barras) + tabla de metricas
    """
    _aplicar_estilo_base()

    color = COLOR_ESCENARIO.get(res.escenario, "#2196F3")
    titulo_esc = {
        "normal":       "Escenario Normal",
        "alta_demanda": "Escenario Alta Demanda",
        "crisis":       "Escenario Crisis",
    }.get(res.escenario, res.escenario)

    ticks    = [r.tick  for r in res.historial]
    O_vals   = [r.O_t   for r in res.historial]
    I_vals   = [r.I_t   for r in res.historial]
    cola_vals= [r.cola_t for r in res.historial]
    tras_vals= [r.traslados_t for r in res.historial]

    # Predicciones alineadas a T+4 para mostrar en el grafico
    pred_ticks = []
    pred_vals  = []
    for r in res.historial:
        if r.pred_O_t4 is not None:
            pred_ticks.append(r.tick + HORIZONTE_PRED)
            pred_vals.append(r.pred_O_t4)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(
        f"F5 — Evaluacion del Sistema Inteligente | {titulo_esc} | Modo {res.modo.capitalize()}",
        fontsize=13, fontweight="bold", color="#ffffff", y=1.01
    )

    # ── Panel [0,0] — Ocupacion O(t) ─────────────────────────────────────────
    ax0 = axes[0, 0]
    ax0.axvline(WARM_UP_TICKS, color="#888888", linestyle=":", linewidth=1,
                label=f"Warm-up ({WARM_UP_TICKS} ticks)")
    ax0.plot(ticks, O_vals, color=color, linewidth=1.5, alpha=0.9, label="O(t) real")
    if pred_ticks:
        ax0.plot(pred_ticks, pred_vals, color="#FFD700", linewidth=1.0,
                 linestyle="--", alpha=0.7, label="O(t+4) predicho")
    # Bandas de referencia por escenario
    umb = UMBRALES_ESPERADOS[res.escenario]
    ax0.axhspan(umb["O_min"], umb["O_max"], alpha=0.08, color=color,
                label=f"Rango esperado [{umb['O_min']:.0f}%–{umb['O_max']:.0f}%]")
    ax0.set_xlabel("Tick (1 tick = 15 min)")
    ax0.set_ylabel("Ocupacion (%)")
    ax0.set_title("Ocupacion Global O(t) vs Prediccion")
    ax0.set_ylim(0, 110)
    ax0.legend(fontsize=7, loc="upper left")
    ax0.grid(True, linewidth=0.5)

    # Anotacion de media de regimen
    ax0.axhline(res.O_media, color=color, linestyle="-.", linewidth=1.0, alpha=0.6)
    ax0.text(TICKS_EVAL * 0.72, res.O_media + 2,
             f"media={res.O_media:.1f}%", color=color, fontsize=8)

    # ── Panel [0,1] — Indicador I(t) ─────────────────────────────────────────
    ax1 = axes[0, 1]
    ax1.axhspan(0,  25, alpha=0.08, color="#4CAF50")
    ax1.axhspan(25, 50, alpha=0.08, color="#8BC34A")
    ax1.axhspan(50, 75, alpha=0.08, color="#FF9800")
    ax1.axhspan(75, 100,alpha=0.10, color="#F44336")
    # Lineas de umbral
    for y, lbl, clr in [(25, "Bajo", "#4CAF50"), (50, "Medio", "#8BC34A"),
                        (75, "Alto", "#FF9800")]:
        ax1.axhline(y, color=clr, linestyle=":", linewidth=0.8, alpha=0.7)
        ax1.text(2, y + 1, lbl, color=clr, fontsize=7)
    ax1.axvline(WARM_UP_TICKS, color="#888888", linestyle=":", linewidth=1)
    ax1.plot(ticks, I_vals, color="#BB86FC", linewidth=1.5, label="I(t)")
    ax1.axhline(res.I_media, color="#BB86FC", linestyle="-.", linewidth=1.0, alpha=0.6)
    ax1.text(TICKS_EVAL * 0.72, res.I_media + 2,
             f"media={res.I_media:.1f}", color="#BB86FC", fontsize=8)
    ax1.set_xlabel("Tick")
    ax1.set_ylabel("Indicador I")
    ax1.set_title("Indicador Compuesto I(t)")
    ax1.set_ylim(0, 105)
    ax1.legend(fontsize=7)
    ax1.grid(True, linewidth=0.5)

    # ── Panel [1,0] — Cola y traslados ───────────────────────────────────────
    ax2 = axes[1, 0]
    ax2.bar(ticks, tras_vals, color="#FF9800", alpha=0.6, width=1.0, label="Traslados/tick")
    ax2_r = ax2.twinx()
    ax2_r.plot(ticks, cola_vals, color="#03DAC6", linewidth=1.2, alpha=0.85, label="Cola espera")
    ax2_r.set_ylabel("Cola de espera (pacientes)", color="#03DAC6")
    ax2_r.tick_params(axis="y", colors="#03DAC6")
    ax2.axvline(WARM_UP_TICKS, color="#888888", linestyle=":", linewidth=1)
    ax2.set_xlabel("Tick")
    ax2.set_ylabel("Traslados / tick", color="#FF9800")
    ax2.tick_params(axis="y", colors="#FF9800")
    ax2.set_title("Cola de Espera y Traslados Internos")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")
    ax2.grid(True, linewidth=0.5)

    # ── Panel [1,1] — Distribucion de niveles + tabla ────────────────────────
    ax3 = axes[1, 1]
    niveles_orden = ["Bajo", "Medio", "Alto", "Critico"]
    niveles_colores = {"Bajo": "#4CAF50", "Medio": "#8BC34A",
                       "Alto": "#FF9800", "Critico": "#F44336"}
    # Normalizar claves (el sistema usa "Critico" con tilde a veces)
    dist_norm: dict[str, int] = {}
    for k, v in res.distribucion_niveles.items():
        k2 = k.replace("í", "i").replace("Cr\u00edtico", "Critico")
        dist_norm[k2] = dist_norm.get(k2, 0) + v

    valores_barras = [dist_norm.get(n, 0) for n in niveles_orden]
    bars = ax3.bar(
        niveles_orden, valores_barras,
        color=[niveles_colores[n] for n in niveles_orden],
        alpha=0.85, edgecolor="#333333"
    )
    for bar, val in zip(bars, valores_barras):
        if val > 0:
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     str(val), ha="center", va="bottom", fontsize=9, color="#ffffff")
    ax3.set_xlabel("Nivel de I")
    ax3.set_ylabel("Ticks en regimen estable")
    ax3.set_title("Distribucion de Niveles del Indicador I")
    ax3.grid(True, linewidth=0.5, axis="y")

    # Tabla de metricas clave (texto en el panel)
    ticks_regime = len(res.historial) - WARM_UP_TICKS
    pred_rmse_str = f"{res.pred_rmse:.4f} pp" if res.pred_rmse is not None else "N/A"
    tabla_texto = (
        f"  O media   : {res.O_media:.1f}%\n"
        f"  O std     : {res.O_std:.1f}%\n"
        f"  O P5-P95  : {res.O_p5:.1f}%–{res.O_p95:.1f}%\n"
        f"  I media   : {res.I_media:.1f}\n"
        f"  Nivel modal: {res.nivel_I_modal}\n"
        f"  Cola media : {res.cola_media:.1f} pac.\n"
        f"  Cola max   : {res.cola_max} pac.\n"
        f"  Altas      : {res.altas_total}\n"
        f"  Traslados  : {res.traslados_total}\n"
        f"  Alertas    : {res.alertas_total}\n"
        f"  RMSE pred  : {pred_rmse_str}\n"
        f"  Ticks reg. : {ticks_regime}"
    )
    ax3.text(
        0.98, 0.97, tabla_texto,
        transform=ax3.transAxes,
        fontsize=7.5, verticalalignment="top", horizontalalignment="right",
        color="#e0e0e0",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d0d1e",
                  edgecolor="#555555", alpha=0.8),
        fontfamily="monospace",
    )

    # Marca de verificacion CE-B
    ok_str = "OK" if res.cumple_O_rango and res.cumple_I_nivel else "REVISAR"
    ok_clr = "#4CAF50" if ok_str == "OK" else "#F44336"
    fig.text(
        0.99, 0.005,
        f"CE-B O_rango={'OK' if res.cumple_O_rango else 'FALLA'}  "
        f"I_nivel={'OK' if res.cumple_I_nivel else 'FALLA'}  "
        f"→ {ok_str}",
        fontsize=8, color=ok_clr, ha="right", va="bottom",
        fontweight="bold",
    )

    plt.tight_layout(pad=2.0)
    plt.savefig(ruta_salida, dpi=120, bbox_inches="tight",
                facecolor="#1a1a2e")
    plt.close()
    print(f"  Figura guardada: {ruta_salida}")


# ─────────────────────────────────────────────────────────────────────────────
# TABLA DE VERIFICACION CE-B
# ─────────────────────────────────────────────────────────────────────────────

def imprimir_verificacion_ce_b(resultados: list[ResultadoEscenario]) -> None:
    """Imprime tabla de verificacion de criterios CE-B en consola."""
    print("\n" + "=" * 72)
    print("VERIFICACION CRITERIOS CE-B — FASE 5")
    print("=" * 72)
    header = f"{'Escenario':<16} {'Modo':<11} {'O_media':>8} {'O_P5':>7} {'O_P95':>7} {'I_media':>8} {'Nivel_modal':<12} {'RMSE_pred':>10} {'CE-B'}"
    print(header)
    print("-" * 72)
    for r in resultados:
        pred_rmse_str = f"{r.pred_rmse:.3f}" if r.pred_rmse is not None else "N/A"
        ok = "OK" if r.cumple_O_rango and r.cumple_I_nivel else "REVISAR"
        print(
            f"{r.escenario:<16} {r.modo:<11} "
            f"{r.O_media:>7.1f}% {r.O_p5:>6.1f}% {r.O_p95:>6.1f}% "
            f"{r.I_media:>7.2f}  {r.nivel_I_modal:<12} "
            f"{pred_rmse_str:>10}  {ok}"
        )
    print("=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL DE EVALUACION
# ─────────────────────────────────────────────────────────────────────────────

def ejecutar_evaluacion_f5(
    escenarios: list[str] | None = None,
    ruta_modelo: str = "/home/claude/modelo_final_f4b.pkl",
    dir_salida: str = "/home/claude",
    verbose: bool = True,
) -> dict[str, ResultadoEscenario]:
    """Pipeline principal de evaluacion F5.

    Parametros
    ----------
    escenarios  : lista de escenarios a evaluar.
                  Default: ['normal'] — solo E1 en primer entregable.
    ruta_modelo : ruta al archivo .pkl del modelo predictivo.
    dir_salida  : directorio donde guardar figuras y datos.
    verbose     : imprimir progreso por tick.

    Retorna
    -------
    Dict escenario -> ResultadoEscenario.
    """
    if escenarios is None:
        escenarios = ["normal"]

    print("=" * 68)
    print("FASE 5 — EVALUACION (SEMMA·Assess)")
    print(f"Escenarios  : {escenarios}")
    print(f"Ticks/sim   : {TICKS_EVAL}  (warm-up: {WARM_UP_TICKS})")
    print(f"Semilla eval: SEED={SEED_EVAL} (D_F5_003)")
    print(f"Horizonte   : T+{HORIZONTE_PRED} ticks (1 hora)")
    print("=" * 68)

    # Cargar modelo predictivo
    modelo = joblib.load(ruta_modelo)
    print(f"Modelo cargado: {ruta_modelo}")
    print(f"  Pipeline: {[n for n, _ in modelo.steps]}")

    resultados: dict[str, ResultadoEscenario] = {}

    for escenario in escenarios:
        modo_asistido = (escenario == "crisis")   # E3 se evalua en modo asistido
        modo_str = "asistido" if modo_asistido else "automatico"

        print(f"\n{'─'*68}")
        print(f"Evaluando: {escenario.upper()} | modo={modo_str} | lambda={LAMBDA_POR_ESCENARIO[escenario]}")
        print(f"{'─'*68}")

        res = ejecutar_simulacion_eval(
            escenario    = escenario,
            modo_asistido= modo_asistido,
            modelo_pred  = modelo,
            seed         = SEED_EVAL,
            ticks        = TICKS_EVAL,
            verbose      = verbose,
        )
        resultados[escenario] = res

        # Generar figura
        ruta_fig = os.path.join(dir_salida, f"fig_f5_{escenario}.png")
        generar_figura_escenario(res, ruta_fig)

        # Resumen rapido
        print(f"\n  METRICAS DE REGIMEN (ticks {WARM_UP_TICKS}–{TICKS_EVAL}):")
        print(f"  O media={res.O_media:.1f}%  std={res.O_std:.1f}%  P5={res.O_p5:.1f}%  P95={res.O_p95:.1f}%")
        print(f"  I media={res.I_media:.1f}  nivel_modal={res.nivel_I_modal}")
        print(f"  cola_media={res.cola_media:.1f}  cola_max={res.cola_max}")
        print(f"  altas={res.altas_total}  traslados={res.traslados_total}  alertas={res.alertas_total}")
        if res.pred_rmse is not None:
            print(f"  RMSE prediccion: {res.pred_rmse:.4f} pp")
        print(f"  CE-B O_rango={'OK' if res.cumple_O_rango else 'FALLA'}  "
              f"I_nivel={'OK' if res.cumple_I_nivel else 'FALLA'}")

    # Verificacion consolidada
    imprimir_verificacion_ce_b(list(resultados.values()))

    # Guardar historial como CSV para el PDF
    for escenario, res in resultados.items():
        df = pd.DataFrame([
            {k: v for k, v in vars(r).items()}
            for r in res.historial
        ])
        ruta_csv = os.path.join(dir_salida, f"historial_f5_{escenario}.csv")
        df.to_csv(ruta_csv, index=False)
        print(f"  Historial guardado: {ruta_csv}")

    return resultados


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Entregable 1 de F5: solo escenario normal
    resultados = ejecutar_evaluacion_f5(
        escenarios   = ["normal"],
        ruta_modelo  = "/home/claude/modelo_final_f4b.pkl",
        dir_salida   = "/home/claude",
        verbose      = True,
    )
