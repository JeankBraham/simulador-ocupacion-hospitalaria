"""
eda_pacientes.py
================
Simulador Inteligente de Ocupación Hospitalaria
Fase 3 — Preparación del dato (SEMMA·Explore+Modify)
Entregable 2: Análisis Exploratorio del Dato Sintético (EDA)

Autor  : Juan Camilo García Braham
Curso  : IA en Salud · Maestría IA y CD · UTP
Año    : 2026
Stack  : Python 3.12 · numpy 2.4 · scipy 1.17 · pandas 3.0
         matplotlib 3.10 · seaborn 0.13

Uso
---
    python eda_pacientes.py

Prerequisito: generador_pacientes.py en la misma carpeta.

Salida: 5 archivos PNG en la misma carpeta donde se ejecuta el script.
    eda_01_proporciones_prioridad.png
    eda_02_estancia_por_area.png
    eda_03_prioridad_vs_area.png
    eda_04_rangos_y_edad.png
    eda_05_llegadas_poisson.png

Análisis cubiertos
------------------
1. Proporciones P1–P4: observado vs esperado + test chi-cuadrado (R09)
2. Tiempos de estancia por área: boxplot + violín con referencias clínicas
3. Correspondencia prioridad × área: heatmap de frecuencia (D011)
4. Valores fuera de rango: histogramas edad y estancia con límites (D005)
5. Llegadas Poisson por escenario: histogramas vs λ teórico (D012)
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chisquare, poisson

# ── Importar módulo generador (debe estar en la misma carpeta) ────────────────
try:
    from generador_pacientes import (
        PROB_PRIORIDAD,
        PRIORIDADES,
        LAMBDA_POR_ESCENARIO,
        ESTANCIA_LOG_NORMAL,
        ESTANCIA_MIN_H,
        ESTANCIA_MAX_H,
        EDAD_MIN,
        EDAD_MAX,
        SEED,
        generar_lote_eda,
        generar_llegadas_tick,
    )
except ModuleNotFoundError:
    sys.exit(
        "ERROR: generador_pacientes.py no encontrado en la carpeta actual.\n"
        "Coloca ambos scripts en la misma carpeta y vuelve a ejecutar."
    )

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN VISUAL GLOBAL
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "P1": "#D62728",   # rojo  — crítico
    "P2": "#FF7F0E",   # naranja — urgente
    "P3": "#2CA02C",   # verde — menos urgente
    "P4": "#1F77B4",   # azul — no urgente
}
AREAS_ORDEN = ["UCI", "Urgencias", "Hospitalización", "Observación", "Sala_de_espera"]
COLOR_FONDO  = "#F8F9FA"
COLOR_LINEA  = "#333333"
DPI          = 150
FIGSIZE_STD  = (10, 6)

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "figure.facecolor": COLOR_FONDO,
    "axes.facecolor":   COLOR_FONDO,
    "font.family":      "DejaVu Sans",
})

N_LOTE = 500   # tamaño del lote EDA — criterio de salida F3

# ─────────────────────────────────────────────────────────────────────────────
# GENERACIÓN DEL LOTE BASE
# ─────────────────────────────────────────────────────────────────────────────

print(f"Generando lote EDA: n={N_LOTE}, seed={SEED} (D_F3_001)...")
lote = generar_lote_eda(n=N_LOTE, escenario="normal", seed=SEED)
df   = pd.DataFrame(lote)
print(f"Lote generado: {len(df)} filas × {len(df.columns)} columnas\n")

# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISIS 1 — PROPORCIONES DE PRIORIDAD CLÍNICA (R09)
# ─────────────────────────────────────────────────────────────────────────────

def analisis_1_proporciones() -> None:
    """Barras observado vs esperado + test χ² de bondad de ajuste."""

    conteos_obs = df["prioridad_clinica"].value_counts().reindex(PRIORIDADES, fill_value=0)
    frec_obs    = conteos_obs.values.astype(float)
    frec_esp    = np.array(PROB_PRIORIDAD) * N_LOTE

    chi2_stat, p_valor = chisquare(f_obs=frec_obs, f_exp=frec_esp)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                             gridspec_kw={"width_ratios": [2, 1]})
    fig.patch.set_facecolor(COLOR_FONDO)
    fig.suptitle("EDA — Análisis 1: Proporciones de Prioridad Clínica\n"
                 f"n = {N_LOTE} · Seed = {SEED}",
                 fontsize=13, fontweight="bold", y=1.01)

    # Panel izquierdo: barras agrupadas
    ax = axes[0]
    x  = np.arange(len(PRIORIDADES))
    w  = 0.35
    bars_obs = ax.bar(x - w/2, frec_obs / N_LOTE * 100, w,
                      label="Observado", color=[PALETTE[p] for p in PRIORIDADES],
                      edgecolor="white", linewidth=0.8)
    bars_esp = ax.bar(x + w/2, frec_esp / N_LOTE * 100, w,
                      label="Esperado (F2)", color="lightgray",
                      edgecolor="gray", linewidth=0.8, hatch="//")

    for bar in bars_obs:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{bar.get_height():.1f}%",
                ha="center", va="bottom", fontsize=9, color=COLOR_LINEA)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}\n({desc})" for p, desc in zip(
        PRIORIDADES, ["Crítico", "Urgente", "Menos urg.", "No urgente"])])
    ax.set_ylabel("Proporción (%)")
    ax.set_title("Proporción observada vs. esperada (F2 Tabla 9)")
    ax.legend()
    ax.set_ylim(0, max(frec_obs / N_LOTE * 100) * 1.25)

    # Panel derecho: resultado del test
    ax2 = axes[1]
    ax2.axis("off")
    interpretacion = ("No se rechaza H₀\n(distribución coherente\ncon F2)"
                      if p_valor > 0.05
                      else "⚠ Se rechaza H₀\n(distribución difiere\nde F2 — revisar)")
    color_resultado = "#2CA02C" if p_valor > 0.05 else "#D62728"

    resumen = (
        f"Test χ² de bondad de ajuste\n"
        f"H₀: obs ~ Multinomial(F2)\n\n"
        f"χ² = {chi2_stat:.4f}\n"
        f"gl  = {len(PRIORIDADES) - 1}\n"
        f"p   = {p_valor:.4f}\n\n"
        f"α   = 0.05\n\n"
    )
    ax2.text(0.05, 0.95, resumen, transform=ax2.transAxes,
             fontsize=10, va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    ax2.text(0.05, 0.28, interpretacion, transform=ax2.transAxes,
             fontsize=10, va="top", color=color_resultado, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.5",
                       facecolor="#E8F5E9" if p_valor > 0.05 else "#FFEBEE",
                       alpha=0.9))

    # Nota R09
    nota = ("⚠ R09 activo: proporciones P1–P4 determinan frecuencia de\n"
            "activación de reglas del sistema experto. Revisar en F5\n"
            "si C del indicador I no sube en escenario de crisis.")
    fig.text(0.01, -0.04, nota, fontsize=8, color="#7B4F00",
             bbox=dict(boxstyle="round", facecolor="#FFF8E1", alpha=0.8))

    plt.tight_layout()
    fig.savefig("eda_01_proporciones_prioridad.png", dpi=DPI,
                bbox_inches="tight", facecolor=COLOR_FONDO)
    plt.close(fig)

    print(f"[1] Proporciones prioridad | χ²={chi2_stat:.4f} p={p_valor:.4f} "
          f"| {'OK' if p_valor > 0.05 else 'REVISAR'}")


# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISIS 2 — TIEMPOS DE ESTANCIA POR ÁREA
# ─────────────────────────────────────────────────────────────────────────────

def analisis_2_estancia() -> None:
    """Boxplot + violín por área con referencias clínicas de F2."""

    # Referencias clínicas (F2 Tabla 10 — medias aritméticas aproximadas)
    referencias = {
        "UCI":             33.0,
        "Urgencias":        7.0,
        "Hospitalización": 20.0,
        "Observación":      5.0,
        "Sala_de_espera":   5.0,
    }

    df_area = df[df["area_requerida"].isin(AREAS_ORDEN)].copy()
    df_area["area_requerida"] = pd.Categorical(
        df_area["area_requerida"], categories=AREAS_ORDEN, ordered=True)
    df_area = df_area.sort_values("area_requerida")

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(COLOR_FONDO)

    colores_area = ["#D62728", "#FF7F0E", "#1F77B4", "#9467BD", "#8C564B"]
    palette_area = dict(zip(AREAS_ORDEN, colores_area))

    sns.violinplot(data=df_area, x="area_requerida", y="tiempo_estancia_esperado",
                   hue="area_requerida", palette=palette_area,
                   inner=None, alpha=0.35, ax=ax, order=AREAS_ORDEN, legend=False)
    sns.boxplot(data=df_area, x="area_requerida", y="tiempo_estancia_esperado",
                hue="area_requerida", palette=palette_area, width=0.25, fliersize=3,
                linewidth=1.2, ax=ax, order=AREAS_ORDEN, legend=False)

    # Líneas de referencia clínica
    for i, area in enumerate(AREAS_ORDEN):
        ref = referencias[area]
        ax.plot([i - 0.4, i + 0.4], [ref, ref],
                color="black", linewidth=1.5, linestyle="--", zorder=5)
        ax.text(i + 0.42, ref, f"ref={ref}h",
                va="center", fontsize=7.5, color="black")

    # Estadísticos por área
    stats = (df_area.groupby("area_requerida", observed=True)["tiempo_estancia_esperado"]
             .agg(["median", "mean", "count"])
             .reindex(AREAS_ORDEN))
    for i, area in enumerate(AREAS_ORDEN):
        if area in stats.index:
            med = stats.loc[area, "median"]
            mn  = stats.loc[area, "mean"]
            cnt = stats.loc[area, "count"]
            ax.text(i, ax.get_ylim()[0] - 2,
                    f"n={cnt}\nmed={med:.1f}h\nmean={mn:.1f}h",
                    ha="center", va="top", fontsize=7.5, color="#333333")

    ax.set_xlabel("Área requerida")
    ax.set_ylabel("Tiempo de estancia esperado (horas)")
    ax.set_title("EDA — Análisis 2: Distribución de tiempos de estancia por área\n"
                 f"n = {N_LOTE} · Log-normal truncada [0.25, 720] h (D005) · "
                 f"Líneas punteadas = media teórica F2",
                 fontsize=11)

    linea_ref = mpatches.Patch(color="black", label="Media teórica (F2 Tabla 10)")
    ax.legend(handles=[linea_ref], loc="upper right", fontsize=9)

    plt.tight_layout()
    fig.savefig("eda_02_estancia_por_area.png", dpi=DPI,
                bbox_inches="tight", facecolor=COLOR_FONDO)
    plt.close(fig)

    print("[2] Estancia por área | Generado OK")
    for area in AREAS_ORDEN:
        sub = df_area[df_area["area_requerida"] == area]["tiempo_estancia_esperado"]
        if len(sub) > 0:
            print(f"    {area:20s} n={len(sub):3d}  "
                  f"min={sub.min():.2f}h  med={sub.median():.2f}h  "
                  f"mean={sub.mean():.2f}h  max={sub.max():.2f}h")


# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISIS 3 — CORRESPONDENCIA PRIORIDAD × ÁREA (D011)
# ─────────────────────────────────────────────────────────────────────────────

def analisis_3_heatmap() -> None:
    """Heatmap de frecuencia prioridad × área para verificar D011."""

    tabla = (df.groupby(["prioridad_clinica", "area_requerida"])
               .size()
               .unstack(fill_value=0)
               .reindex(index=PRIORIDADES,
                        columns=AREAS_ORDEN,
                        fill_value=0))

    # Versión normalizada por fila (% dentro de cada prioridad)
    tabla_pct = tabla.div(tabla.sum(axis=1), axis=0) * 100

    # Probabilidades teóricas de D011 (F2 Tabla 11)
    teorico = pd.DataFrame(0.0, index=PRIORIDADES, columns=AREAS_ORDEN)
    from generador_pacientes import AREAS_POR_PRIORIDAD
    for p, dist in AREAS_POR_PRIORIDAD.items():
        for area, prob in dist.items():
            teorico.loc[p, area] = prob * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(COLOR_FONDO)
    fig.suptitle("EDA — Análisis 3: Correspondencia Prioridad × Área\n"
                 f"n = {N_LOTE} · Distribución categórica condicional D011",
                 fontsize=12, fontweight="bold")

    kw = dict(annot=True, fmt=".1f", linewidths=0.5, linecolor="white",
              cbar_kws={"label": "% dentro de prioridad"})

    sns.heatmap(tabla_pct, ax=axes[0], cmap="YlOrRd", vmin=0, vmax=100, **kw)
    axes[0].set_title("Observado (%)")
    axes[0].set_xlabel("Área requerida")
    axes[0].set_ylabel("Prioridad clínica")

    sns.heatmap(teorico, ax=axes[1], cmap="YlOrRd", vmin=0, vmax=100, **kw)
    axes[1].set_title("Esperado — F2 Tabla 11 (%)")
    axes[1].set_xlabel("Área requerida")
    axes[1].set_ylabel("")

    # Resaltar celdas con desviación > 10 pp
    dif = (tabla_pct - teorico).abs()
    for i, p in enumerate(PRIORIDADES):
        for j, a in enumerate(AREAS_ORDEN):
            if dif.loc[p, a] > 10:
                axes[0].add_patch(
                    plt.Rectangle((j, i), 1, 1, fill=False,
                                  edgecolor="#D62728", lw=2))

    nota = "Celdas con borde rojo: desviación > 10 pp respecto al teórico"
    fig.text(0.01, -0.03, nota, fontsize=8, color="#7B4F00",
             bbox=dict(boxstyle="round", facecolor="#FFF8E1", alpha=0.8))

    plt.tight_layout()
    fig.savefig("eda_03_prioridad_vs_area.png", dpi=DPI,
                bbox_inches="tight", facecolor=COLOR_FONDO)
    plt.close(fig)

    print("[3] Heatmap prioridad × área | Generado OK")
    print("    Desviaciones > 10 pp:")
    flag = False
    for p in PRIORIDADES:
        for a in AREAS_ORDEN:
            d = dif.loc[p, a]
            if d > 10:
                print(f"    ⚠  {p} × {a}: obs={tabla_pct.loc[p,a]:.1f}% "
                      f"esp={teorico.loc[p,a]:.1f}% Δ={d:.1f}pp")
                flag = True
    if not flag:
        print("    Ninguna — distribución coherente con D011")


# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISIS 4 — VALORES FUERA DE RANGO (D005, DAMA)
# ─────────────────────────────────────────────────────────────────────────────

def analisis_4_rangos() -> None:
    """Histogramas de edad y estancia con límites de rango marcados."""

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(COLOR_FONDO)
    fig.suptitle("EDA — Análisis 4: Validación de rangos\n"
                 f"n = {N_LOTE} · Líneas rojas = límites definidos en F2",
                 fontsize=12, fontweight="bold")

    # Panel 1: edad
    ax1 = axes[0]
    ax1.hist(df["edad"], bins=30, color="#4E9AF1", edgecolor="white",
             linewidth=0.6, alpha=0.85)
    ax1.axvline(EDAD_MIN, color="#D62728", linewidth=1.8, linestyle="--",
                label=f"Mín = {EDAD_MIN}")
    ax1.axvline(EDAD_MAX, color="#D62728", linewidth=1.8, linestyle="--",
                label=f"Máx = {EDAD_MAX}")
    ax1.axvline(df["edad"].mean(), color="#333333", linewidth=1.5,
                linestyle="-", label=f"Media obs = {df['edad'].mean():.1f}")
    fuera_edad = ((df["edad"] < EDAD_MIN) | (df["edad"] > EDAD_MAX)).sum()
    ax1.set_title(f"Distribución de edad\n"
                  f"[{EDAD_MIN}, {EDAD_MAX}] años · Fuera de rango: {fuera_edad}")
    ax1.set_xlabel("Edad (años)")
    ax1.set_ylabel("Frecuencia")
    ax1.legend(fontsize=8)

    # Panel 2: tiempo de estancia (escala log para visualizar cola)
    ax2 = axes[1]
    bins_log = np.logspace(np.log10(max(df["tiempo_estancia_esperado"].min(), 0.1)),
                           np.log10(df["tiempo_estancia_esperado"].max() + 1), 40)
    ax2.hist(df["tiempo_estancia_esperado"], bins=bins_log,
             color="#F4A460", edgecolor="white", linewidth=0.6, alpha=0.85)
    ax2.axvline(ESTANCIA_MIN_H, color="#D62728", linewidth=1.8, linestyle="--",
                label=f"Mín = {ESTANCIA_MIN_H}h (D005)")
    ax2.axvline(ESTANCIA_MAX_H, color="#D62728", linewidth=1.8, linestyle="--",
                label=f"Máx = {ESTANCIA_MAX_H}h")
    ax2.axvline(df["tiempo_estancia_esperado"].median(), color="#333333",
                linewidth=1.5, linestyle="-",
                label=f"Mediana obs = {df['tiempo_estancia_esperado'].median():.2f}h")
    fuera_est = ((df["tiempo_estancia_esperado"] < ESTANCIA_MIN_H) |
                 (df["tiempo_estancia_esperado"] > ESTANCIA_MAX_H)).sum()
    ax2.set_xscale("log")
    ax2.set_title(f"Distribución de tiempo de estancia (escala log)\n"
                  f"[{ESTANCIA_MIN_H}, {ESTANCIA_MAX_H}] h · Fuera de rango: {fuera_est}")
    ax2.set_xlabel("Tiempo de estancia esperado (horas, escala log)")
    ax2.set_ylabel("Frecuencia")
    ax2.legend(fontsize=8)

    # Tabla resumen debajo
    resumen_txt = (
        f"Resumen de validación de rangos (n={N_LOTE})\n"
        f"{'Campo':<30} {'Min obs':>10} {'Max obs':>10} {'Fuera rango':>12}\n"
        f"{'-'*65}\n"
        f"{'edad':<30} {df['edad'].min():>10}  {df['edad'].max():>9}  {fuera_edad:>12}\n"
        f"{'tiempo_estancia_esperado (h)':<30} "
        f"{df['tiempo_estancia_esperado'].min():>10.4f}  "
        f"{df['tiempo_estancia_esperado'].max():>9.2f}  {fuera_est:>12}"
    )
    fig.text(0.01, -0.08, resumen_txt, fontsize=8, fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    plt.tight_layout()
    fig.savefig("eda_04_rangos_y_edad.png", dpi=DPI,
                bbox_inches="tight", facecolor=COLOR_FONDO)
    plt.close(fig)

    print(f"[4] Rangos | edad fuera=[{fuera_edad}] estancia fuera=[{fuera_est}] "
          f"| {'OK' if fuera_edad == 0 and fuera_est == 0 else 'REVISAR'}")


# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISIS 5 — LLEGADAS POISSON POR ESCENARIO (D012)
# ─────────────────────────────────────────────────────────────────────────────

def analisis_5_poisson() -> None:
    """Histogramas de llegadas/tick vs distribución teórica Poisson(λ)."""

    N_TICKS = 200   # ticks simulados por escenario para estabilizar la media
    ESCENARIOS = list(LAMBDA_POR_ESCENARIO.keys())
    colores_esc = {"normal": "#2CA02C", "alta_demanda": "#FF7F0E", "crisis": "#D62728"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.patch.set_facecolor(COLOR_FONDO)
    fig.suptitle(f"EDA — Análisis 5: Distribución de llegadas por tick · "
                 f"{N_TICKS} ticks por escenario\n"
                 f"Barras = observado · Línea = Poisson(λ) teórico · D012",
                 fontsize=11, fontweight="bold")

    for ax, esc in zip(axes, ESCENARIOS):
        lam = LAMBDA_POR_ESCENARIO[esc]
        rng = np.random.default_rng(SEED)
        llegadas = [len(generar_llegadas_tick(t, esc, rng)) for t in range(N_TICKS)]

        media_obs = np.mean(llegadas)
        var_obs   = np.var(llegadas)
        max_k     = max(llegadas) + 1

        # Histograma observado
        bins = np.arange(-0.5, max_k + 0.5, 1)
        ax.hist(llegadas, bins=bins, density=True,
                color=colores_esc[esc], edgecolor="white",
                linewidth=0.7, alpha=0.75, label="Observado")

        # Distribución teórica Poisson(λ)
        k_vals  = np.arange(0, max_k + 1)
        pmf_teo = poisson.pmf(k_vals, mu=lam)
        ax.plot(k_vals, pmf_teo, "ko-", markersize=5, linewidth=1.5,
                label=f"Poisson(λ={lam})")

        ax.axvline(media_obs, color=colores_esc[esc], linewidth=1.8,
                   linestyle="--", alpha=0.9,
                   label=f"Media obs = {media_obs:.2f}")
        ax.axvline(lam, color="black", linewidth=1.2,
                   linestyle=":", label=f"λ teórico = {lam}")

        titulo = {"normal": "Escenario Normal\n(λ=1.5, ~6 pac/h)",
                  "alta_demanda": "Escenario Alta Demanda\n(λ=3.0, ~12 pac/h)",
                  "crisis": "Escenario Crisis\n(λ=5.0, ~20 pac/h)"}
        ax.set_title(titulo[esc], fontsize=10)
        ax.set_xlabel("Llegadas por tick (15 min)")
        ax.set_ylabel("Densidad de probabilidad")
        ax.legend(fontsize=7.5)

        resumen = (f"n_ticks={N_TICKS}\n"
                   f"media={media_obs:.3f} (λ={lam})\n"
                   f"var={var_obs:.3f}\n"
                   f"E[var/media]={var_obs/media_obs:.3f} (teo≈1.00)")
        ax.text(0.97, 0.97, resumen, transform=ax.transAxes,
                fontsize=7.5, va="top", ha="right", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

        print(f"[5] {esc:15s} | λ={lam} media_obs={media_obs:.3f} "
              f"var={var_obs:.3f} var/media={var_obs/media_obs:.3f}")

    plt.tight_layout()
    fig.savefig("eda_05_llegadas_poisson.png", dpi=DPI,
                bbox_inches="tight", facecolor=COLOR_FONDO)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# EJECUCIÓN SECUENCIAL
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("EDA — Simulador Inteligente de Ocupación Hospitalaria · F3")
    print(f"n_lote={N_LOTE} · seed={SEED} · stack: pandas/matplotlib/seaborn")
    print("=" * 65 + "\n")

    analisis_1_proporciones()
    analisis_2_estancia()
    analisis_3_heatmap()
    analisis_4_rangos()
    analisis_5_poisson()

    print("\n" + "=" * 65)
    print("Figuras guardadas:")
    for i, nombre in enumerate([
        "eda_01_proporciones_prioridad.png",
        "eda_02_estancia_por_area.png",
        "eda_03_prioridad_vs_area.png",
        "eda_04_rangos_y_edad.png",
        "eda_05_llegadas_poisson.png",
    ], 1):
        ruta = Path(nombre)
        estado = "✔" if ruta.exists() else "✘ NO GENERADO"
        print(f"  {estado}  {nombre}")
    print("=" * 65)
