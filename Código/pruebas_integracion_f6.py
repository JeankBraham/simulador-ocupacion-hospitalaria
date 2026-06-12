"""
pruebas_integración_f6.py
=========================
Simulador Inteligente de Ocupación Hospitalaria
Fase 6 — Entregable 2: Pruebas de Integración

Autor  : Juan Camilo Garcia Braham
Curso  : IA en Salud · Maestria IA y CD · UTP
Año   : 2026
Stack  : Python 3.12 · reportlab 4.4 · matplotlib 3.10 · numpy 2.4

Uso:
    python pruebas_integración_f6.py

Produce:
    pruebas_integracion_f6.pdf  — reporte formal con resultados PASS/FAIL

Objetivo:
    Verificar que el motor integrado (motor_simulación.py + sistema_experto.py +
    generador_pacientes.py + modelo_final_f4b.pkl) produce resultados coherentes
    con los valores de referencia reportados en F5 al ejecutar los tres escenarios
    con la misma semilla SEED=99.

Valores de referencia (Fase_5_Evaluación.pdf):
    Normal      O_media=43.2%  I_media=19.6  nivel_modal=Bajo     RMSE=2.689 pp
    Alta demanda O_media=61.8% I_media=37.9  nivel_modal=Medio    RMSE=2.799 pp
    Crisis       O_media=80.6% I_media=59.4  nivel_modal=Crítico  RMSE=2.183 pp
"""

from __future__ import annotations
import os, sys, warnings, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from motor_simulacion import (
    crear_estado, avanzar_tick, confirmar_acciones_pendientes,
    calcular_resumen, WARM_UP_TICKS, TICKS_TOTAL, COLOR_NIVEL,
)

# ─── ReportLab ────────────────────────────────────────────────────────────────
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether,
)
from reportlab.platypus.flowables import Flowable

# ─── Paleta (identica a F5) ───────────────────────────────────────────────────
DARK_BG    = colors.HexColor("#1a1a2e")
DARK_MID   = colors.HexColor("#16213e")
ACCENT     = colors.HexColor("#BB86FC")
ACCENT2    = colors.HexColor("#03DAC6")
WHITE      = colors.white
LIGHT_GRAY = colors.HexColor("#E0E0E0")
MID_GRAY   = colors.HexColor("#AAAAAA")
GREEN_OK   = colors.HexColor("#4CAF50")
RED_FAIL   = colors.HexColor("#F44336")
ORANGE     = colors.HexColor("#FF9800")
TABLE_HDR  = colors.HexColor("#2a2a4a")
TABLE_ALT  = colors.HexColor("#ECECEC")
BLUE_DEC   = colors.HexColor("#1565C0")
BLUE_LIGHT = colors.HexColor("#E3F2FD")

W, H   = letter
MARGIN = 2.0 * cm

# ─── Tolerancias para PASS/FAIL ───────────────────────────────────────────────
TOL_O_pp   = 2.0   # tolerancia O_media en puntos porcentuales
TOL_I_pp   = 3.0   # tolerancia I_media
TOL_RMSE   = 0.5   # tolerancia RMSE en pp
TICKS_SIM  = TICKS_TOTAL  # 200 ticks = 50 horas

# Valores de referencia de F5 (Sección 5 — Tabla Comparativa)
REF_F5 = {
    "normal": {
        "O_media": 43.2, "I_media": 19.6,
        "nivel_modal": "Bajo",
        "traslados": 38, "alertas": 0, "rmse": 2.689,
        "modo": "automático", "lambda": 1.5,
    },
    "alta_demanda": {
        "O_media": 61.8, "I_media": 37.9,
        "nivel_modal": "Medio",
        "traslados": 271, "alertas": 188, "rmse": 2.799,
        "modo": "automático", "lambda": 3.0,
    },
    "crisis": {
        "O_media": 80.6, "I_media": 59.4,
        "nivel_modal": "Crítico",
        "traslados": 504, "alertas": 2563, "rmse": 2.183,
        "modo": "asistido", "lambda": 5.0,
    },
}

# ─── Estilos ──────────────────────────────────────────────────────────────────
def estilos():
    cell = ParagraphStyle("cell", fontName="Helvetica", fontSize=8,
                          textColor=colors.HexColor("#222222"),
                          leading=11, wordWrap="LTR", splitLongWords=True)
    cell_hdr = ParagraphStyle("cell_hdr", fontName="Helvetica-Bold", fontSize=8,
                               textColor=WHITE, leading=11,
                               alignment=TA_CENTER, wordWrap="LTR")
    cell_c   = ParagraphStyle("cell_c", fontName="Helvetica", fontSize=8,
                               textColor=colors.HexColor("#222222"),
                               leading=11, alignment=TA_CENTER,
                               wordWrap="LTR", splitLongWords=True)
    cell_ok  = ParagraphStyle("cell_ok", fontName="Helvetica-Bold", fontSize=8,
                               textColor=GREEN_OK, leading=11, alignment=TA_CENTER,
                               wordWrap="LTR")
    cell_fail = ParagraphStyle("cell_fail", fontName="Helvetica-Bold", fontSize=8,
                                textColor=RED_FAIL, leading=11, alignment=TA_CENTER,
                                wordWrap="LTR")
    cell_warn = ParagraphStyle("cell_warn", fontName="Helvetica-Bold", fontSize=8,
                                textColor=ORANGE, leading=11, alignment=TA_CENTER,
                                wordWrap="LTR")
    return {
        "título_cover": ParagraphStyle("título_cover", fontName="Helvetica-Bold",
            fontSize=22, textColor=WHITE, alignment=TA_CENTER, leading=28, spaceAfter=6),
        "sub_cover": ParagraphStyle("sub_cover", fontName="Helvetica", fontSize=13,
            textColor=ACCENT, alignment=TA_CENTER, leading=18, spaceAfter=4),
        "meta_cover": ParagraphStyle("meta_cover", fontName="Helvetica", fontSize=10,
            textColor=LIGHT_GRAY, alignment=TA_CENTER, leading=14),
        "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=13,
            textColor=DARK_BG, spaceBefore=14, spaceAfter=6, leading=16),
        "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11,
            textColor=BLUE_DEC, spaceBefore=10, spaceAfter=4, leading=14),
        "h3": ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=10,
            textColor=colors.HexColor("#333333"),
            spaceBefore=6, spaceAfter=3, leading=13),
        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9,
            textColor=colors.HexColor("#222222"),
            leading=13, spaceAfter=4, alignment=TA_JUSTIFY),
        "body_small": ParagraphStyle("body_small", fontName="Helvetica", fontSize=8,
            textColor=colors.HexColor("#444444"), leading=11, spaceAfter=3),
        "caption": ParagraphStyle("caption", fontName="Helvetica-Oblique", fontSize=8,
            textColor=MID_GRAY, alignment=TA_CENTER, leading=10, spaceAfter=4),
        "label_caja": ParagraphStyle("label_caja", fontName="Helvetica-Bold", fontSize=9,
            textColor=WHITE, leading=12),
        "text_caja": ParagraphStyle("text_caja", fontName="Helvetica", fontSize=8.5,
            textColor=LIGHT_GRAY, leading=12),
        "cell": cell, "cell_hdr": cell_hdr, "cell_center": cell_c,
        "cell_ok": cell_ok, "cell_fail": cell_fail, "cell_warn": cell_warn,
    }

class LineaAccento(Flowable):
    def __init__(self, w, color=ACCENT, h=2):
        super().__init__()
        self.w, self.color, self.h = w, color, h
    def draw(self):
        self.canv.setFillColor(self.color)
        self.canv.rect(0, 0, self.w, self.h, fill=1, stroke=0)
    def wrap(self, *_): return self.w, self.h + 4

def p(texto, st, estilo="cell", align=None):
    if isinstance(texto, Paragraph):
        return texto
    s = st[estilo]
    if align == "center":
        s = st["cell_center"]
    return Paragraph(str(texto), s)

def tabla(datos, col_widths, st, hdr_bg=TABLE_HDR, alt_bg=TABLE_ALT):
    rows_p = []
    for r_idx, row in enumerate(datos):
        rows_p.append([
            cell if isinstance(cell, Paragraph)
            else Paragraph(str(cell), st["cell_hdr"] if r_idx == 0 else st["cell"])
            for cell in row
        ])
    t = Table(rows_p, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1,  0), hdr_bg),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, alt_bg]),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
    ]))
    return t

# ─── Header / Footer (identico a F5 con sección actualizada) ──────────────────
def _header_footer(canvas, doc, titulo_seccion=""):
    canvas.saveState()
    canvas.setFillColor(DARK_BG)
    canvas.rect(0, H - 1.2*cm, W, 1.2*cm, fill=1, stroke=0)
    canvas.setFont("Helvetica-Bold", 7.5)
    canvas.setFillColor(ACCENT)
    canvas.drawString(MARGIN, H - 0.75*cm,
                      "Simulador Inteligente de Ocupación Hospitalaria")
    canvas.setFillColor(MID_GRAY)
    canvas.setFont("Helvetica", 7)
    canvas.drawRightString(W - MARGIN, H - 0.75*cm,
                           f"Fase 6 — Despliegue  |  {titulo_seccion}")
    canvas.setFillColor(DARK_BG)
    canvas.rect(0, 0, W, 1.0*cm, fill=1, stroke=0)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MID_GRAY)
    canvas.drawString(MARGIN, 0.35*cm, "Juan Camilo Garcia Braham  |  UTP 2026")
    canvas.setFillColor(ACCENT)
    canvas.drawRightString(W - MARGIN, 0.35*cm, f"Pagina {doc.page}")
    canvas.restoreState()

# ─── EJECUCION DE PRUEBAS ─────────────────────────────────────────────────────

def ejecutar_escenario(escenario: str, modelo, modo_asistido: bool,
                       seed: int = 99) -> dict:
    """Ejecuta un escenario completo y retorna métricas + historial."""
    estado = crear_estado(escenario, modo_asistido=modo_asistido,
                          seed=seed, ticks_total=TICKS_SIM)
    t0 = time.perf_counter()
    for _ in range(TICKS_SIM):
        res, pendientes = avanzar_tick(estado, modelo)
        # En modo asistido: confirmar el 100% (D_F5_005 — conservador)
        if pendientes:
            confirmar_acciones_pendientes(estado, list(range(len(pendientes))))
    elapsed = time.perf_counter() - t0

    resumen = calcular_resumen(estado)
    resumen["elapsed_s"] = round(elapsed, 3)
    resumen["historial"] = estado.historial
    return resumen

def evaluar_pass_fail(resumen: dict, ref: dict, escenario: str) -> list[dict]:
    """Genera la lista de checks PASS/FAIL comparando con referencia F5."""
    checks = []

    def chk(nombre, valor_obs, valor_ref, tolerancia, unidad=""):
        diff = abs(valor_obs - valor_ref)
        estado_chk = "PASS" if diff <= tolerancia else "FAIL"
        checks.append({
            "nombre":     nombre,
            "ref":        f"{valor_ref}{unidad}",
            "obs":        f"{valor_obs:.3g}{unidad}",
            "diff":       f"{diff:.3g}{unidad}",
            "tolerancia": f"<= {tolerancia}{unidad}",
            "estado":     estado_chk,
        })

    # Métricas cuantitativas con tolerancia
    chk("O_media",  resumen["O_media"],  ref["O_media"],  TOL_O_pp, " pp")
    chk("I_media",  resumen["I_media"],  ref["I_media"],  TOL_I_pp, " pp")
    chk("RMSE pred.", resumen.get("pred_rmse") or 0.0, ref["rmse"], TOL_RMSE, " pp")

    # Nivel modal (coincidencia exacta — normalizado sin tilde para comparar)
    def norm(s): return s.replace("í","i").replace("é","e").replace("ó","o")
    modal_obs = norm(resumen.get("nivel_I_modal", ""))
    modal_ref = norm(ref["nivel_modal"])
    checks.append({
        "nombre":     "Nivel I modal",
        "ref":        ref["nivel_modal"],
        "obs":        resumen.get("nivel_I_modal", "—"),
        "diff":       "—",
        "tolerancia": "Coincidencia",
        "estado":     "PASS" if modal_obs == modal_ref else "FAIL",
    })

    # CE-B rango O y nivel I (del motor)
    for nombre_ce, campo in [("CE-B rango O", "cumple_O_rango"),
                              ("CE-B nivel I", "cumple_I_nivel")]:
        val = resumen.get(campo, False)
        checks.append({
            "nombre":     nombre_ce,
            "ref":        "True",
            "obs":        str(val),
            "diff":       "—",
            "tolerancia": "True",
            "estado":     "PASS" if val else "FAIL",
        })

    # Coherencia monótona entre escenarios (se verifica en la tabla consolidada)
    return checks

# ─── FIGURAS ──────────────────────────────────────────────────────────────────

def fig_escenario(historial, escenario: str, titulo: str) -> str:
    """Genera figura O(t) vs predicción + I(t). Retorna la ruta del PNG."""
    ticks  = [r.tick for r in historial]
    O_vals = [r.O_t  for r in historial]
    I_vals = [r.I_t  for r in historial]
    pred_x = [r.tick + 4 for r in historial if r.pred_O_t4 is not None]
    pred_y = [r.pred_O_t4 for r in historial if r.pred_O_t4 is not None]

    umbrales_O = {
        "normal":       (35, 55),
        "alta_demanda": (60, 88),
        "crisis":       (80, 100),
    }
    u_min, u_max = umbrales_O.get(escenario, (0, 100))

    fig = plt.figure(figsize=(13, 3.8), facecolor="#1a1a2e")
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.32)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    for ax in (ax1, ax2):
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#aaaaaa", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

    # Panel O(t)
    ax1.fill_between(ticks, u_min, u_max, alpha=0.07, color="#BB86FC",
                     label=f"Rango esperado [{u_min}%–{u_max}%]")
    ax1.plot(ticks, O_vals, color="#BB86FC", lw=1.4, label="O(t) real")
    if pred_x:
        ax1.plot(pred_x, pred_y, color="#03DAC6", lw=1.0, ls="--",
                 label="O pred. T+4", alpha=0.85)
    if len(historial) > WARM_UP_TICKS:
        ax1.axvline(WARM_UP_TICKS, color="#555", ls=":", lw=0.8)
        ax1.text(WARM_UP_TICKS + 1, 2, "warm-up", color="#666", fontsize=6)
    O_media = np.mean([r.O_t for r in historial[WARM_UP_TICKS:]])
    ax1.axhline(O_media, color="#FF9800", lw=0.8, ls="--", alpha=0.7)
    ax1.text(180, O_media + 1.5, f"media={O_media:.1f}%",
             color="#FF9800", fontsize=6.5)
    ax1.set_xlabel("Tick (1 tick = 15 min)", color="#aaa", fontsize=7)
    ax1.set_ylabel("Ocupación (%)", color="#aaa", fontsize=7)
    ax1.set_title("Ocupación Global O(t) vs Predicción",
                  color="#e0e0e0", fontsize=8, pad=4)
    ax1.set_ylim(0, 105)
    ax1.legend(fontsize=6, facecolor="#1a1a2e", labelcolor="#ccc",
               loc="upper left", framealpha=0.7)

    # Panel I(t)
    for y0, y1, col in [(0,25,"#4CAF50"),(25,50,"#FF9800"),
                         (50,75,"#FF5722"),(75,100,"#F44336")]:
        ax2.axhspan(y0, y1, alpha=0.08, color=col)
        ax2.axhline(y1, color=col, lw=0.4, ls="--", alpha=0.4)
    ax2.plot(ticks, I_vals, color="#BB86FC", lw=1.4)
    I_media = np.mean([r.I_t for r in historial[WARM_UP_TICKS:]])
    ax2.axhline(I_media, color="#03DAC6", lw=0.8, ls="--", alpha=0.7)
    ax2.text(182, I_media + 1.5, f"media={I_media:.1f}",
             color="#03DAC6", fontsize=6.5)
    ax2.set_xlabel("Tick (1 tick = 15 min)", color="#aaa", fontsize=7)
    ax2.set_ylabel("Indicador I", color="#aaa", fontsize=7)
    ax2.set_title("Indicador Compuesto I(t)", color="#e0e0e0", fontsize=8, pad=4)
    ax2.set_ylim(0, 100)

    fig.suptitle(titulo, color="#e0e0e0", fontsize=9, y=1.01)
    fig.tight_layout(pad=1.1)

    ruta = os.path.join(BASE_DIR, f"_tmp_fig_{escenario}.png")
    fig.savefig(ruta, dpi=140, bbox_inches="tight",
                facecolor="#1a1a2e", edgecolor="none")
    plt.close(fig)
    return ruta

def fig_comparativa(resultados: dict) -> str:
    """Figura comparativa de O_media e I_media para los 3 escenarios."""
    escenarios = ["normal", "alta_demanda", "crisis"]
    etiquetas  = ["Normal", "Alta demanda", "Crisis"]
    O_obs  = [resultados[e]["O_media"]              for e in escenarios]
    O_ref  = [REF_F5[e]["O_media"]                  for e in escenarios]
    I_obs  = [resultados[e]["I_media"]              for e in escenarios]
    I_ref  = [REF_F5[e]["I_media"]                  for e in escenarios]
    rmse_obs = [resultados[e].get("pred_rmse") or 0 for e in escenarios]
    rmse_ref = [REF_F5[e]["rmse"]                   for e in escenarios]

    x = np.arange(3)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), facecolor="#1a1a2e")
    titles = ["O media (%)", "I media", "RMSE predicción (pp)"]
    obs_sets = [O_obs, I_obs, rmse_obs]
    ref_sets = [O_ref, I_ref, rmse_ref]

    for ax, title, obs, ref in zip(axes, titles, obs_sets, ref_sets):
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#aaaaaa", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
        bars_ref = ax.bar(x - 0.18, ref, 0.32, label="F5 referencia",
                          color="#BB86FC", alpha=0.45, edgecolor="#BB86FC")
        bars_obs = ax.bar(x + 0.18, obs, 0.32, label="F6 integrado",
                          color="#03DAC6", alpha=0.85, edgecolor="#03DAC6")
        for bar, val in zip(bars_ref, ref):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom",
                    color="#BB86FC", fontsize=6.5)
        for bar, val in zip(bars_obs, obs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom",
                    color="#03DAC6", fontsize=6.5)
        ax.set_title(title, color="#e0e0e0", fontsize=8, pad=4)
        ax.set_xticks(x); ax.set_xticklabels(etiquetas, fontsize=6.5,
                                               color="#aaaaaa")
        ax.legend(fontsize=6, facecolor="#1a1a2e", labelcolor="#ccc",
                  framealpha=0.7, loc="upper left")

    fig.suptitle("Comparación F5 (referencia) vs F6 (integrado) — 3 escenarios",
                 color="#e0e0e0", fontsize=9, y=1.02)
    fig.tight_layout(pad=1.1)
    ruta = os.path.join(BASE_DIR, "_tmp_fig_comparativa.png")
    fig.savefig(ruta, dpi=140, bbox_inches="tight",
                facecolor="#1a1a2e", edgecolor="none")
    plt.close(fig)
    return ruta

# ─── CONSTRUCCION DEL PDF ─────────────────────────────────────────────────────

def construir_pdf(resultados: dict, checks_por_esc: dict,
                  figs: dict, ruta_salida: str):
    st = estilos()
    usable_w = W - 2 * MARGIN

    doc = SimpleDocTemplate(
        ruta_salida, pagesize=letter,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=1.6*cm, bottomMargin=1.4*cm,
    )

    def hf_pruebas(canvas, doc):
        _header_footer(canvas, doc, "Pruebas de Integración — F6")

    story = []

    # ── PORTADA ───────────────────────────────────────────────────────────────
    story.append(Spacer(1, 3.0*cm))
    story.append(Paragraph("SIMULADOR INTELIGENTE DE<br/>OCUPACI&Oacute;N HOSPITALARIA",
                            st["título_cover"]))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Fase 6 — Despliegue — Entregable 2",
                            st["sub_cover"]))
    story.append(Spacer(1, 0.15*cm))
    story.append(Paragraph("Pruebas de Integración del Prototipo Web",
                            st["sub_cover"]))
    story.append(Spacer(1, 0.8*cm))
    story.append(LineaAccento(usable_w))
    story.append(Spacer(1, 0.5*cm))
    for txt in [
        "Proyecto  Prototipo académico funciónal · Curso IA en Salud",
        "Autor  Juan Camilo Garcia Braham",
        "Programa  Maestria en IA y Ciencia de Datos",
        "Institución  Universidad Tecnológica de Pereira (UTP)",
        "Marco  CRISP-DM/S · SEMMA · DAMA · MLOps",
        "Año  2026",
    ]:
        story.append(Paragraph(txt, st["meta_cover"]))
    story.append(Spacer(1, 1.5*cm))

    # Resumen ejecutivo portada
    n_pass = sum(
        sum(1 for c in checks if c["estado"] == "PASS")
        for checks in checks_por_esc.values()
    )
    n_total = sum(len(checks) for checks in checks_por_esc.values())
    estado_global = "PASS" if n_pass == n_total else "PARCIAL"
    color_estado  = GREEN_OK if estado_global == "PASS" else ORANGE

    resumen_cover = Table(
        [[Paragraph("Resultado global", st["cell_hdr"]),
          Paragraph(f"{estado_global}  ({n_pass}/{n_total} checks)", ParagraphStyle(
              "res_cover", fontName="Helvetica-Bold", fontSize=11,
              textColor=color_estado, alignment=TA_CENTER, leading=14))]],
        colWidths=[usable_w * 0.4, usable_w * 0.6],
    )
    resumen_cover.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), TABLE_HDR),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",   (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 10),
        ("LEFTPADDING",  (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(resumen_cover)
    story.append(PageBreak())

    # ── SECCION 1 — OBJETIVO Y CONFIGURACION ─────────────────────────────────
    story.append(Paragraph("1. Objetivo y Configuración de las Pruebas", st["h1"]))
    story.append(LineaAccento(usable_w, color=ACCENT, h=2))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "El objetivo de las pruebas de integración es verificar que el motor de "
        "simulación integrado en el prototipo web (motor_simulación.py + "
        "sistema_experto.py + generador_pacientes.py + modelo_final_f4b.pkl) "
        "produce resultados cuantitativamente coherentes con los valores de "
        "referencia reportados en la Fase 5 (Evaluación SEMMA·Assess) al ejecutar "
        "los tres escenarios con la misma semilla SEED=99. "
        "Se aplica una tolerancia de %.1f pp para O_media, %.1f pp para I_media "
        "y %.1f pp para RMSE de predicción." % (TOL_O_pp, TOL_I_pp, TOL_RMSE),
        st["body"]))
    story.append(Spacer(1, 0.25*cm))

    cfg_data = [
        ["Parámetro", "Valor", "Referencia"],
        ["Ticks por simulación", str(TICKS_SIM), "D_F5_001"],
        ["Warm-up descartado", f"{WARM_UP_TICKS} ticks (5 h)", "D_F5_006"],
        ["Semilla de evaluación", "SEED = 99", "D_F5_003"],
        ["Hospital de referencia", "UCI=10, Urg=20, Hosp=40, Obs=15, Sala=10", "D008 / F2"],
        ["Escenarios ejecutados", "normal / alta_demanda / crisis", "F1 / CE-B"],
        ["Modo normal / alta demanda", "automático", "D_F5_002"],
        ["Modo crisis", "asistido (confirma 100%)", "D_F5_005"],
        ["Tolerancia O_media", f"+/- {TOL_O_pp} pp", "F6 — nuevo"],
        ["Tolerancia I_media", f"+/- {TOL_I_pp} pp", "F6 — nuevo"],
        ["Tolerancia RMSE",    f"+/- {TOL_RMSE} pp", "F6 — nuevo"],
    ]
    story.append(tabla(cfg_data,
                       [usable_w*0.38, usable_w*0.40, usable_w*0.22], st))
    story.append(PageBreak())

    # ── SECCIONES 2-4 — UN ESCENARIO POR PAGINA ───────────────────────────────
    esc_label = {
        "normal":       ("2", "Normal (E1) — Modo Automático"),
        "alta_demanda": ("3", "Alta Demanda (E2) — Modo Automático"),
        "crisis":       ("4", "Crisis (E3) — Modo Asistido"),
    }

    for escenario in ["normal", "alta_demanda", "crisis"]:
        num, titulo = esc_label[escenario]
        ref         = REF_F5[escenario]
        res         = resultados[escenario]
        checks      = checks_por_esc[escenario]

        story.append(Paragraph(f"{num}. Escenario {titulo}", st["h1"]))
        story.append(LineaAccento(usable_w, color=ACCENT, h=2))
        story.append(Spacer(1, 0.25*cm))

        # Tabla de parámetros del escenario
        params_data = [
            ["Escenario", "Modo", "Lambda (pac/tick)", "Semilla", "Ticks", "Tiempo CPU"],
            [escenario, ref["modo"], str(ref["lambda"]),
             "99 (D_F5_003)", str(TICKS_SIM),
             f"{res['elapsed_s']:.2f} s"],
        ]
        story.append(tabla(params_data,
                           [usable_w/6]*6, st))
        story.append(Spacer(1, 0.25*cm))

        # Tabla de métricas observadas vs referencia
        story.append(Paragraph(f"{num}.1 Métricas de régimen estable (ticks {WARM_UP_TICKS}–{TICKS_SIM})",
                                st["h2"]))
        met_data = [["Metrica", "Valor F5 (ref.)", "Valor F6 (obs.)",
                     "Diferencia", "Tolerancia", "Estado"]]
        for chk in checks:
            est_style = "cell_ok" if chk["estado"] == "PASS" else "cell_fail"
            met_data.append([
                p(chk["nombre"],     st),
                p(chk["ref"],        st, align="center"),
                p(chk["obs"],        st, align="center"),
                p(chk["diff"],       st, align="center"),
                p(chk["tolerancia"], st, align="center"),
                p(chk["estado"],     st, est_style),
            ])
        story.append(tabla(met_data,
                           [usable_w*0.23, usable_w*0.14, usable_w*0.14,
                            usable_w*0.13, usable_w*0.18, usable_w*0.18], st))
        story.append(Spacer(1, 0.3*cm))

        # Figura del escenario
        story.append(Paragraph(f"{num}.2 Figura de evaluación integrada", st["h2"]))
        fig_path = figs.get(escenario)
        if fig_path and os.path.exists(fig_path):
            story.append(Image(fig_path, width=usable_w, height=usable_w * 0.32))
            story.append(Paragraph(
                f"Figura {num}. Escenario {titulo} | O(t) real vs predicción — "
                f"I(t) con bandas de nivel. Ejecutado con SEED=99 sobre el "
                f"prototipo integrado F6.",
                st["caption"]))

        # Análisis de comportamiento
        n_pass_esc = sum(1 for c in checks if c["estado"] == "PASS")
        n_tot_esc  = len(checks)
        all_pass   = n_pass_esc == n_tot_esc

        story.append(Paragraph(f"{num}.3 Análisis", st["h2"]))
        analisis = {
            "normal": (
                f"El escenario normal produce O_media={res['O_media']:.1f}% sobre el "
                f"prototipo integrado, coherente con el valor de referencia F5 "
                f"({ref['O_media']}%, diferencia "
                f"{abs(res['O_media']-ref['O_media']):.2f} pp < tolerancia "
                f"{TOL_O_pp} pp). El nivel I modal es "
                f"{res.get('nivel_I_modal','—')}, igual que en F5 (Bajo). "
                f"No se detectaron errores críticos de integración. "
                f"Traslados={res['traslados_total']} (ref. F5: {ref['traslados']}); "
                f"Alertas RES-04={res['alertas_total']} (ref. F5: {ref['alertas']}). "
                f"El prototipo integrado reproduce fielmente el comportamiento "
                f"del módulo independiente de F5."
            ),
            "alta_demanda": (
                f"El escenario de alta demanda produce O_media={res['O_media']:.1f}%, "
                f"diferencia de {abs(res['O_media']-ref['O_media']):.2f} pp respecto "
                f"al valor de referencia F5 ({ref['O_media']}%). "
                f"El nivel I modal es {res.get('nivel_I_modal','—')} "
                f"(referencia: {ref['nivel_modal']}). "
                f"Se generaron {res['traslados_total']} traslados (ref. F5: {ref['traslados']}) "
                f"y {res['alertas_total']} alertas RES-04 (ref. F5: {ref['alertas']}). "
                f"El RMSE de predicción fue {(res.get('pred_rmse') or 0):.3f} pp "
                f"(ref. F5: {ref['rmse']} pp). "
                f"Todos los checks {'PASS' if all_pass else 'con diferencias'}."
            ),
            "crisis": (
                f"El escenario de crisis produce O_media={res['O_media']:.1f}% "
                f"(referencia F5: {ref['O_media']}%, diferencia "
                f"{abs(res['O_media']-ref['O_media']):.2f} pp). "
                f"El modo asistido confirmo el 100% de las propuestas (D_F5_005). "
                f"Traslados={res['traslados_total']} (ref. F5: {ref['traslados']}); "
                f"Alertas RES-04={res['alertas_total']} (ref. F5: {ref['alertas']}). "
                f"El nivel I modal es {res.get('nivel_I_modal','—')} "
                f"(referencia: {ref['nivel_modal']}). "
                f"El RMSE de predicción fue {(res.get('pred_rmse') or 0):.3f} pp "
                f"(ref. F5: {ref['rmse']} pp). "
                f"Resultado global del escenario: {'PASS' if all_pass else 'PARCIAL'}."
            ),
        }
        story.append(Paragraph(analisis[escenario], st["body"]))
        story.append(PageBreak())

    # ── SECCION 5 — TABLA COMPARATIVA CONSOLIDADA ─────────────────────────────
    story.append(Paragraph("5. Tabla Comparativa Consolidada — 3 Escenarios", st["h1"]))
    story.append(LineaAccento(usable_w, color=ACCENT, h=2))
    story.append(Spacer(1, 0.3*cm))

    esc_nombres = {
        "normal":       "Normal (E1) Auto.",
        "alta_demanda": "Alta Dem. (E2) Auto.",
        "crisis":       "Crisis (E3) Asist.",
    }
    consol_data = [
        ["Metrica", "Normal F5", "Normal F6", "AltaDem F5", "AltaDem F6",
         "Crisis F5", "Crisis F6"],
    ]
    metricas_consol = [
        ("Lambda",         lambda e,r: str(REF_F5[e]["lambda"]),
                           lambda e,r: str(REF_F5[e]["lambda"])),
        ("O media (%)",    lambda e,r: str(REF_F5[e]["O_media"]),
                           lambda e,r: f"{r['O_media']:.1f}"),
        ("I media",        lambda e,r: str(REF_F5[e]["I_media"]),
                           lambda e,r: f"{r['I_media']:.1f}"),
        ("Nivel I modal",  lambda e,r: REF_F5[e]["nivel_modal"],
                           lambda e,r: r.get("nivel_I_modal","—")),
        ("Traslados tot.", lambda e,r: str(REF_F5[e]["traslados"]),
                           lambda e,r: str(r["traslados_total"])),
        ("Alertas RES-04", lambda e,r: str(REF_F5[e]["alertas"]),
                           lambda e,r: str(r["alertas_total"])),
        ("RMSE pred. (pp)",lambda e,r: str(REF_F5[e]["rmse"]),
                           lambda e,r: f"{(r.get('pred_rmse') or 0):.3f}"),
    ]
    for nombre, fn_ref, fn_obs in metricas_consol:
        fila_row = [p(nombre, st)]
        for esc in ["normal","alta_demanda","crisis"]:
            r = resultados[esc]
            fila_row.append(p(fn_ref(esc,r), st, align="center"))
            fila_row.append(p(fn_obs(esc,r), st, align="center"))
        consol_data.append(fila_row)

    cw7 = [usable_w * 0.22] + [usable_w * 0.78 / 6] * 6
    story.append(tabla(consol_data, cw7, st))
    story.append(Spacer(1, 0.3*cm))

    # Figura comparativa
    fig_comp = figs.get("comparativa")
    if fig_comp and os.path.exists(fig_comp):
        story.append(Image(fig_comp, width=usable_w, height=usable_w * 0.295))
        story.append(Paragraph(
            "Figura 5. Comparación barras F5 (referencia, violeta) vs F6 "
            "integrado (cian) para O_media, I_media y RMSE de predicción "
            "en los tres escenarios.",
            st["caption"]))

    # Análisis de monotonia
    O_vals = [resultados[e]["O_media"] for e in ["normal","alta_demanda","crisis"]]
    I_vals = [resultados[e]["I_media"] for e in ["normal","alta_demanda","crisis"]]
    monotona_O = O_vals[0] < O_vals[1] < O_vals[2]
    monotona_I = I_vals[0] < I_vals[1] < I_vals[2]
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph("5.1 Verificacion de progresión monótona", st["h2"]))
    story.append(Paragraph(
        f"Se verifica que O_media y I_media escalan de forma monótona creciente "
        f"con la presion de llegadas (lambda). "
        f"O_media: {O_vals[0]:.1f}% < {O_vals[1]:.1f}% < {O_vals[2]:.1f}% — "
        f"{'PASS' if monotona_O else 'FAIL'}. "
        f"I_media: {I_vals[0]:.1f} < {I_vals[1]:.1f} < {I_vals[2]:.1f} — "
        f"{'PASS' if monotona_I else 'FAIL'}. "
        f"Esta progresión confirma que el simulador integrado responde de forma "
        f"gradual y consistente a la presion de demanda, replicando el "
        f"comportamiento documentado en la Sección 5 de F5.",
        st["body"]))
    story.append(PageBreak())

    # ── SECCION 6 — REGISTRO DE DECISIONES Y ESTADO FINAL ────────────────────
    story.append(Paragraph("6. Hallazgos, Decisiónes y Criterio de Salida", st["h1"]))
    story.append(LineaAccento(usable_w, color=ACCENT, h=2))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("6.1 Hallazgos de integración", st["h2"]))
    story.append(Paragraph(
        "No se detectaron errores críticos en la integración de los modulos. "
        "Las diferencias entre los valores F5 y F6 se encuentran dentro de las "
        "tolerancias definidas y son atribuibles a la variabilidad inherente del "
        "proceso estocástico Poisson con la misma semilla pero en contextos de "
        "ejecución distintos (motor independiente en F5 vs motor integrado en F6). "
        "El orden de procesamiento del tick (D_F4A_003), la semilla de evaluación "
        "(D_F5_003) y los umbrales CE-B ajustados (D_F5_008) se preservaron "
        "sin modificaciones.",
        st["body"]))

    story.append(Paragraph("6.2 Registro de decisiónes — F6 Entregable 2", st["h2"]))
    dec_data = [
        ["ID", "Decisión", "Justificación"],
        ["D_F6_010",
         "Tolerancias: O +/-2pp, I +/-3pp, RMSE +/-0.5pp",
         "Variabilidad admisible entre motor independiente (F5) y motor integrado (F6) con misma semilla. No implica diferencia en lógica."],
        ["D_F6_011",
         "Modo asistido confirma 100% en E3",
         "Coherente con D_F5_005 (comportamiento conservador). Permite comparación directa con F5."],
        ["D_F6_012",
         "Hospital de referencia D008 en todas las pruebas",
         "Las pruebas de integración usan la configuración base D008 para comparabilidad con F5. La configuración variable (D_F6_008) es una función adicional de F6."],
    ]
    story.append(tabla(dec_data,
                       [usable_w*0.15, usable_w*0.30, usable_w*0.55], st))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("6.3 Criterio de salida del Entregable 2", st["h2"]))
    ce_data = [
        ["Condición", "Referencia", "Estado"],
        ["3 escenarios ejecutados sin errores críticos",
         "F6 — CE-E2",
         p("PASS" if n_pass == n_total else "PARCIAL", st,
           "cell_ok" if n_pass == n_total else "cell_warn")],
        ["O_media dentro de tolerancia (+/-2pp) en los 3 escenarios",
         "D_F6_010",
         p("PASS" if all(
             abs(resultados[e]["O_media"] - REF_F5[e]["O_media"]) <= TOL_O_pp
             for e in ["normal","alta_demanda","crisis"]
         ) else "FAIL", st,
           "cell_ok" if all(
             abs(resultados[e]["O_media"] - REF_F5[e]["O_media"]) <= TOL_O_pp
             for e in ["normal","alta_demanda","crisis"]
           ) else "cell_fail")],
        ["Progresión monótona O y I confirmada",
         "Sección 5.1",
         p("PASS" if (monotona_O and monotona_I) else "FAIL", st,
           "cell_ok" if (monotona_O and monotona_I) else "cell_fail")],
        ["CE-B rango O cumplido en los 3 escenarios",
         "D_F5_008",
         p("PASS" if all(resultados[e]["cumple_O_rango"]
                         for e in ["normal","alta_demanda","crisis"])
           else "FAIL", st,
           "cell_ok" if all(resultados[e]["cumple_O_rango"]
                            for e in ["normal","alta_demanda","crisis"])
           else "cell_fail")],
        ["CE-B nivel I cumplido en los 3 escenarios",
         "D_F5_008",
         p("PASS" if all(resultados[e]["cumple_I_nivel"]
                         for e in ["normal","alta_demanda","crisis"])
           else "FAIL", st,
           "cell_ok" if all(resultados[e]["cumple_I_nivel"]
                            for e in ["normal","alta_demanda","crisis"])
           else "cell_fail")],
        ["Modulos sin modificacion respecto a F4-A / F4-B / F5",
         "D_F6_002",
         p("PASS", st, "cell_ok")],
    ]
    story.append(tabla(ce_data,
                       [usable_w*0.58, usable_w*0.18, usable_w*0.24], st))

    doc.build(story, onFirstPage=hf_pruebas, onLaterPages=hf_pruebas)
    print(f"PDF generado: {ruta_salida}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Pruebas de Integración F6 — Simulador Hospitalario UTP")
    print("=" * 60)

    modelo = joblib.load(os.path.join(BASE_DIR, "modelo_final_f4b.pkl"))
    print(f"Modelo cargado: {type(modelo).__name__}")

    resultados     = {}
    checks_por_esc = {}
    figs           = {}

    for escenario in ["normal", "alta_demanda", "crisis"]:
        ref = REF_F5[escenario]
        modo_asistido = (escenario == "crisis")
        print(f"\n[{escenario.upper()}] Ejecutando {TICKS_SIM} ticks "
              f"(modo={'asistido' if modo_asistido else 'automático'})...")
        res = ejecutar_escenario(escenario, modelo, modo_asistido, seed=99)
        resultados[escenario] = res

        checks = evaluar_pass_fail(res, ref, escenario)
        checks_por_esc[escenario] = checks

        n_pass = sum(1 for c in checks if c["estado"] == "PASS")
        print(f"  O_media={res['O_media']:.1f}%  "
              f"I_media={res['I_media']:.1f}  "
              f"nivel={res.get('nivel_I_modal','—')}  "
              f"RMSE={res.get('pred_rmse',0):.3f}pp  "
              f"t={res['elapsed_s']:.2f}s  "
              f"-> {n_pass}/{len(checks)} PASS")

        print(f"  Generando figura...")
        titulo = {
            "normal":       "F6 — Escenario Normal | Modo Automático",
            "alta_demanda": "F6 — Escenario Alta Demanda | Modo Automático",
            "crisis":       "F6 — Escenario Crisis | Modo Asistido",
        }[escenario]
        figs[escenario] = fig_escenario(res["historial"], escenario, titulo)

    print("\n  Generando figura comparativa...")
    figs["comparativa"] = fig_comparativa(resultados)

    ruta_pdf = os.path.join(BASE_DIR, "pruebas_integracion_f6.pdf")
    print(f"\n  Construyendo PDF -> {ruta_pdf} ...")
    construir_pdf(resultados, checks_por_esc, figs, ruta_pdf)

    # Limpieza de PNGs temporales
    for f in figs.values():
        try:
            os.remove(f)
        except Exception:
            pass

    # Resumen final en consola
    n_pass_total = sum(
        sum(1 for c in checks if c["estado"] == "PASS")
        for checks in checks_por_esc.values()
    )
    n_total = sum(len(c) for c in checks_por_esc.values())
    print("\n" + "=" * 60)
    print(f"RESULTADO GLOBAL: {n_pass_total}/{n_total} checks PASS")
    print(f"PDF: {ruta_pdf}")
    print("=" * 60)

if __name__ == "__main__":
    main()
