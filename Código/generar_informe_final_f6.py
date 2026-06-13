"""
generar_informe_final_f6.py
============================
Simulador Inteligente de Ocupacion Hospitalaria
Fase 6 - Entregable 5: Informe Final del Proyecto

Autor  : Juan Camilo Garcia Braham
Curso  : IA en Salud - Maestria IA y CD - UTP
Anno   : 2026
Stack  : Python 3.12 - reportlab 4.4 - matplotlib 3.10

Uso:
    python generar_informe_final_f6.py

Produce:
    informe_final_f6.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether,
)
from reportlab.platypus.flowables import Flowable

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Paleta ------------------------------------------------------------------
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

# --- Flowables ---------------------------------------------------------------
class LineaAccento(Flowable):
    def __init__(self, w, color=ACCENT, h=2):
        super().__init__()
        self.w, self.color, self.h = w, color, h
    def draw(self):
        self.canv.setFillColor(self.color)
        self.canv.rect(0, 0, self.w, self.h, fill=1, stroke=0)
    def wrap(self, *_): return self.w, self.h + 4

def CajaDestacada(texto, ancho=None):
    aw = ancho or (W - 2 * MARGIN)
    sty = ParagraphStyle("caja_dest", fontName="Helvetica", fontSize=8.5,
                         textColor=colors.HexColor("#0d2a4a"),
                         leading=13, wordWrap="LTR", splitLongWords=True)
    t = Table([[Paragraph(texto, sty)]], colWidths=[aw - 0.6*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), BLUE_LIGHT),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ("RIGHTPADDING",  (0,0), (-1,-1), 10),
        ("BOX",           (0,0), (-1,-1), 1.2, BLUE_DEC),
        ("ROUNDEDCORNERS",[4]),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
    ]))
    return t

# --- Estilos -----------------------------------------------------------------
def estilos():
    return {
        "titulo_cover": ParagraphStyle("titulo_cover",
            fontName="Helvetica-Bold", fontSize=22,
            textColor=WHITE, alignment=TA_CENTER, leading=28, spaceAfter=6),
        "sub_cover": ParagraphStyle("sub_cover",
            fontName="Helvetica", fontSize=13,
            textColor=ACCENT, alignment=TA_CENTER, leading=18, spaceAfter=4),
        "meta_cover": ParagraphStyle("meta_cover",
            fontName="Helvetica", fontSize=10,
            textColor=LIGHT_GRAY, alignment=TA_CENTER, leading=14),
        "resumen_cover": ParagraphStyle("resumen_cover",
            fontName="Helvetica", fontSize=9,
            textColor=LIGHT_GRAY, alignment=TA_JUSTIFY, leading=13),
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
        "body_bullet": ParagraphStyle("body_bullet",
            fontName="Helvetica", fontSize=9,
            textColor=colors.HexColor("#222222"),
            leading=13, spaceAfter=2, leftIndent=14),
        "caption": ParagraphStyle("caption",
            fontName="Helvetica-Oblique", fontSize=8,
            textColor=colors.HexColor("#444444"), alignment=TA_CENTER, leading=11, spaceAfter=6),
        "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=8,
            textColor=colors.HexColor("#222222"), leading=11,
            wordWrap="LTR", splitLongWords=True),
        "cell_hdr": ParagraphStyle("cell_hdr",
            fontName="Helvetica-Bold", fontSize=8,
            textColor=WHITE, leading=11, alignment=TA_CENTER, wordWrap="LTR"),
        "cell_ok": ParagraphStyle("cell_ok",
            fontName="Helvetica-Bold", fontSize=8,
            textColor=GREEN_OK, leading=11, alignment=TA_CENTER, wordWrap="LTR"),
        "cell_c": ParagraphStyle("cell_c", fontName="Helvetica", fontSize=8,
            textColor=colors.HexColor("#222222"), leading=11,
            alignment=TA_CENTER, wordWrap="LTR"),
    }

def p(texto, st, estilo="body"):
    if isinstance(texto, Paragraph): return texto
    return Paragraph(str(texto), st[estilo])

def tabla(datos, col_widths, st, hdr_bg=TABLE_HDR):
    rows_p = []
    for r_idx, row in enumerate(datos):
        rows_p.append([
            cell if isinstance(cell, Paragraph)
            else Paragraph(str(cell),
                st["cell_hdr"] if r_idx == 0 else st["cell"])
            for cell in row
        ])
    t = Table(rows_p, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1,  0), hdr_bg),
        ("VALIGN",         (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, TABLE_ALT]),
        ("GRID",           (0, 0), (-1, -1), 0.3,
                           colors.HexColor("#CCCCCC")),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
        ("LEFTPADDING",    (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 6),
    ]))
    return t

# --- Header / Footer ---------------------------------------------------------
def _hf(canvas, doc, seccion=""):
    canvas.saveState()
    canvas.setFillColor(DARK_BG)
    canvas.rect(0, H - 1.2*cm, W, 1.2*cm, fill=1, stroke=0)
    canvas.setFont("Helvetica-Bold", 7.5)
    canvas.setFillColor(ACCENT)
    canvas.drawString(MARGIN, H - 0.75*cm,
                      "Simulador Inteligente de Ocupacion Hospitalaria")
    canvas.setFillColor(MID_GRAY)
    canvas.setFont("Helvetica", 7)
    canvas.drawRightString(W - MARGIN, H - 0.75*cm,
                           f"Informe Final  |  {seccion}")
    canvas.setFillColor(DARK_BG)
    canvas.rect(0, 0, W, 1.0*cm, fill=1, stroke=0)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MID_GRAY)
    canvas.drawString(MARGIN, 0.35*cm,
                      "Juan Camilo Garcia Braham  |  UTP 2026")
    canvas.setFillColor(ACCENT)
    canvas.drawRightString(W - MARGIN, 0.35*cm, f"Pagina {doc.page}")
    canvas.restoreState()

# --- Figuras -----------------------------------------------------------------
def fig_ciclo_crisp() -> str:
    """Diagrama del ciclo CRISP-DM/S con las 6 fases del proyecto."""
    fig, ax = plt.subplots(figsize=(9, 3.8), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.axis("off")

    fases = [
        ("F1\nComprension\ndel problema",  0.5, 1.8, "#BB86FC"),
        ("F2\nComprension\ndel dato",       2.0, 1.8, "#BB86FC"),
        ("F3\nPreparacion\ndel dato",       3.5, 1.8, "#03DAC6"),
        ("F4-A\nSistema\nexperto",          5.0, 1.8, "#03DAC6"),
        ("F4-B\nModelo\npredictivo",        6.5, 1.8, "#03DAC6"),
        ("F5\nEvaluacion",                  8.0, 1.8, "#FF9800"),
        ("F6\nDespliegue",                  9.5, 1.8, "#4CAF50"),
    ]
    import matplotlib.patches as mpatches
    for txt, x, y, col in fases:
        rect = mpatches.FancyBboxPatch(
            (x - 0.62, y - 0.55), 1.24, 1.1,
            boxstyle="round,pad=0.08",
            facecolor=colors.HexColor(col).hexval() if False else col,
            edgecolor=col, linewidth=1.5, alpha=0.18,
        )
        ax.add_patch(rect)
        ax.text(x, y, txt, ha="center", va="center",
                color=col, fontsize=7.5, fontweight="bold",
                fontfamily="DejaVu Sans")

    # Flechas entre fases
    for i in range(len(fases) - 1):
        x1 = fases[i][1] + 0.62
        x2 = fases[i+1][1] - 0.62
        y0 = fases[i][2]
        ax.annotate("", xy=(x2, y0), xytext=(x1, y0),
                    arrowprops=dict(arrowstyle="->",
                                   color="#555", lw=1.2))

    # Etiqueta SEMMA
    ax.text(5.0, 0.6, "SEMMA (subciclo F2-F5):  Sample - Explore - Modify - Model - Assess",
            ha="center", va="center", color="#aaa", fontsize=7.5,
            style="italic")

    ax.set_xlim(-0.2, 10.2)
    ax.set_ylim(0.2, 2.7)
    ruta = os.path.join(BASE_DIR, "_tmp_crisp.png")
    fig.tight_layout(pad=0.4)
    fig.savefig(ruta, dpi=130, bbox_inches="tight",
                facecolor="#1a1a2e", edgecolor="none")
    plt.close(fig)
    return ruta

def fig_resultados_escenarios() -> str:
    """Figura comparativa de metricas clave para los 3 escenarios."""
    escenarios  = ["Normal\n(E1)", "Alta demanda\n(E2)", "Crisis\n(E3)"]
    O_vals      = [43.2, 61.8, 80.6]
    I_vals      = [19.6, 37.9, 59.4]
    traslados   = [38,   271,  504]
    alertas     = [0,    188,  2563]

    fig = plt.figure(figsize=(12, 3.6), facecolor="#1a1a2e")
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.42)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

    datasets = [
        ("O media (%)",         O_vals,    "#BB86FC"),
        ("I media",             I_vals,    "#03DAC6"),
        ("Traslados totales",   traslados, "#FF9800"),
        ("Alertas RES-04",      alertas,   "#F44336"),
    ]
    x = np.arange(3)
    for ax, (titulo, vals, col) in zip(axes, datasets):
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#aaaaaa", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
        bars = ax.bar(x, vals, color=col, alpha=0.82, edgecolor=col,
                      width=0.55)
        for bar, val in zip(bars, vals):
            label = f"{val:,.0f}" if val >= 100 else f"{val:.1f}"
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.02,
                    label, ha="center", va="bottom",
                    color=col, fontsize=7, fontweight="bold")
        ax.set_title(titulo, color="#e0e0e0", fontsize=8, pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels(escenarios, fontsize=6.5, color="#aaa")
        ax.set_ylim(0, max(vals) * 1.22)

    fig.suptitle("Resultados de evaluacion — 3 escenarios (SEED=99, 200 ticks)",
                 color="#e0e0e0", fontsize=9, y=1.03)
    fig.tight_layout(pad=1.0)
    ruta = os.path.join(BASE_DIR, "_tmp_resultados.png")
    fig.savefig(ruta, dpi=130, bbox_inches="tight",
                facecolor="#1a1a2e", edgecolor="none")
    plt.close(fig)
    return ruta

def fig_arquitectura_capas() -> str:
    """Diagrama de capas del sistema."""
    fig, ax = plt.subplots(figsize=(9, 3.2), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.axis("off")

    import matplotlib.patches as mpatches
    capas = [
        (0.5, "Capa de presentacion\napp.py  (Streamlit F6)",
         "#4CAF50", 0.9),
        (1.5, "Capa de motor\nmotor_simulacion.py  (F6)",
         "#03DAC6", 0.9),
        (2.5, "Capa de logica del dominio\nsistema_experto.py  (F4-A)    "
              "generador_pacientes.py  (F3)",
         "#BB86FC", 0.9),
        (3.5, "Capa de prediccion\nmodelo_final_f4b.pkl  (F4-B, sklearn Pipeline)",
         "#FF9800", 0.9),
    ]
    for y, txt, col, h in capas:
        rect = mpatches.FancyBboxPatch(
            (0.3, y - h/2), 8.4, h,
            boxstyle="round,pad=0.05",
            facecolor="none", edgecolor=col, linewidth=1.6, alpha=0.85,
        )
        ax.add_patch(rect)
        ax.text(4.5, y, txt, ha="center", va="center",
                color=col, fontsize=8.5, fontweight="bold",
                fontfamily="DejaVu Sans")
        # Flecha hacia abajo entre capas
        if y < 3.5:
            ax.annotate("", xy=(4.5, y + h/2 + 0.02),
                        xytext=(4.5, y + h/2 + 0.38),
                        arrowprops=dict(arrowstyle="<-",
                                        color="#555", lw=1.2))

    ax.set_xlim(0, 9); ax.set_ylim(0.2, 4.3)
    ruta = os.path.join(BASE_DIR, "_tmp_arq.png")
    fig.tight_layout(pad=0.3)
    fig.savefig(ruta, dpi=130, bbox_inches="tight",
                facecolor="#1a1a2e", edgecolor="none")
    plt.close(fig)
    return ruta

# --- Portada completa (funcion de canvas pura — onFirstPage) ----------------
def dibujar_portada(canvas, doc):
    """Dibuja la portada completa con fondo oscuro sobre el canvas directamente.

    Se registra como onFirstPage en SimpleDocTemplate. Esto garantiza
    control total sobre coordenadas absolutas de la pagina sin depender
    del sistema de layout de Platypus.
    """
    canvas.saveState()

    # Coordenadas absolutas de la pagina (origen = esquina inf-izq)
    # Fondo oscuro completo
    canvas.setFillColor(DARK_BG)
    canvas.rect(0, 0, W, H, fill=1, stroke=0)

    # Centro horizontal de la pagina
    cx = W / 2

    # ── Titulo principal ──────────────────────────────────────────────────
    y = H - 3.8*cm
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 24)
    for linea in ["SIMULADOR INTELIGENTE DE", "OCUPACION HOSPITALARIA"]:
        canvas.drawCentredString(cx, y, linea)
        y -= 0.9*cm

    # ── Subtitulo ─────────────────────────────────────────────────────────
    y -= 0.3*cm
    canvas.setFillColor(ACCENT)
    canvas.setFont("Helvetica", 14)
    canvas.drawCentredString(cx, y, "Informe Final del Proyecto")
    y -= 0.6*cm
    canvas.setFont("Helvetica", 12)
    canvas.drawCentredString(cx, y, "Fase 6 - Despliegue - Entregable 5")

    # ── Linea de acento ───────────────────────────────────────────────────
    y -= 0.6*cm
    canvas.setFillColor(ACCENT)
    canvas.rect(MARGIN, y, W - 2*MARGIN, 2, fill=1, stroke=0)
    y -= 0.55*cm

    # ── Tabla de metadatos ────────────────────────────────────────────────
    metadatos = [
        ("Proyecto",    "Prototipo academico funcional - Curso IA en Salud"),
        ("Autor",       "Juan Camilo Garcia Braham"),
        ("Programa",    "Maestria en IA y Ciencia de Datos"),
        ("Institucion", "Universidad Tecnologica de Pereira (UTP)"),
        ("Marco",       "CRISP-DM/S - SEMMA - DAMA - MLOps"),
        ("Anno",        "2026"),
    ]
    x_etiq = MARGIN
    x_val  = MARGIN + (W - 2*MARGIN) * 0.22
    row_h  = 0.52*cm

    for etiqueta, valor in metadatos:
        canvas.setFont("Helvetica-Bold", 10)
        canvas.setFillColor(ACCENT)
        canvas.drawString(x_etiq, y, etiqueta)
        canvas.setFont("Helvetica", 10)
        canvas.setFillColor(LIGHT_GRAY)
        canvas.drawString(x_val, y, valor)
        y -= row_h

    # ── Linea separadora cian ─────────────────────────────────────────────
    y -= 0.35*cm
    canvas.setFillColor(ACCENT2)
    canvas.rect(MARGIN, y, W - 2*MARGIN, 1, fill=1, stroke=0)
    y -= 0.45*cm

    # ── Caja de resumen ejecutivo ─────────────────────────────────────────
    resumen_titulo = "Resumen ejecutivo"
    resumen_texto = (
        "Este documento es el informe final del Simulador Inteligente de "
        "Ocupacion Hospitalaria, prototipo academico desarrollado "
        "individualmente bajo el framework CRISP-DM/S para el curso de "
        "IA en Salud de la Maestria en IA y Ciencia de Datos de la UTP "
        "(2026). El sistema integra un generador de pacientes sinteticos, "
        "un sistema experto de asignacion de camas con 19 reglas en 6 "
        "capas, un modelo predictivo de ocupacion (Regresion Lineal, "
        "RMSE=2.64 pp) y una interfaz web en Streamlit. Los tres "
        "escenarios de evaluacion (normal, alta demanda, crisis) producen "
        "indicadores I coherentes con sus umbrales ajustados (D_F5_008) "
        "y una progresion monotona que confirma la validez del simulador. "
        "Todos los riesgos identificados en F1 fueron mitigados. "
        "El criterio de salida de F6 se declara cumplido."
    )

    pad_x    = 0.4*cm
    inner_w  = W - 2*MARGIN - 2*pad_x
    # Wrap con medida real de fuente — evita lineas cortas por estimacion incorrecta
    from reportlab.pdfbase import pdfmetrics as _pm
    words    = resumen_texto.split()
    lines_r, cur = [], ""
    for word in words:
        test = (cur + " " + word).strip()
        if _pm.stringWidth(test, "Helvetica", 8.5) <= inner_w:
            cur = test
        else:
            lines_r.append(cur)
            cur = word
    if cur:
        lines_r.append(cur)

    line_h = 0.40*cm
    caja_h = (len(lines_r) + 1.8) * line_h + 0.2*cm

    canvas.setStrokeColor(ACCENT2)
    canvas.setFillColor(DARK_MID)
    canvas.setLineWidth(1.0)
    canvas.roundRect(MARGIN, y - caja_h, W - 2*MARGIN, caja_h,
                     4, fill=1, stroke=1)

    y_t = y - 0.38*cm
    canvas.setFont("Helvetica-Bold", 9)
    canvas.setFillColor(ACCENT2)
    canvas.drawString(MARGIN + pad_x, y_t, resumen_titulo)
    y_t -= line_h

    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(LIGHT_GRAY)
    for linea in lines_r:
        canvas.drawString(MARGIN + pad_x, y_t, linea)
        y_t -= line_h

    canvas.restoreState()


# --- PDF ---------------------------------------------------------------------
def construir_pdf(ruta_salida: str):
    st = estilos()
    uw = W - 2 * MARGIN

    doc = SimpleDocTemplate(
        ruta_salida, pagesize=letter,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=1.6*cm, bottomMargin=1.4*cm,
    )
    def hf(canvas, doc): _hf(canvas, doc, "Descripcion, resultados y trabajo futuro")

    story = []
    # Pagina 1 reservada para la portada (dibujada en onFirstPage)
    story.append(PageBreak())

    # =========================================================================
    # =========================================================================
    # PORTADA — fondo oscuro completo, patron identico a F2/F5
    # =========================================================================

    # =========================================================================
    # TABLA DE CONTENIDO
    # =========================================================================
    story.append(Paragraph("Tabla de Contenido", st["h1"]))
    story.append(LineaAccento(uw))
    story.append(Spacer(1, 0.3*cm))
    toc = [
        ["Seccion", "Contenido"],
        ["1", "Descripcion del sistema"],
        ["2", "Metodologia CRISP-DM/S aplicada"],
        ["3", "Resultados de evaluacion por escenario"],
        ["4", "Analisis del modelo predictivo"],
        ["5", "Limitaciones del prototipo"],
        ["6", "Trabajo futuro y normograma colombiano"],
        ["7", "Criterio de salida de F6 - verificacion final"],
        ["8", "Referencias normativas"],
    ]
    story.append(tabla(toc, [uw*0.08, uw*0.92], st))
    story.append(PageBreak())

    # =========================================================================
    # SECCION 1 — DESCRIPCION DEL SISTEMA
    # =========================================================================
    story.append(Paragraph("1. Descripcion del Sistema", st["h1"]))
    story.append(LineaAccento(uw))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("1.1 Vision general", st["h2"]))
    story.append(p(
        "El Simulador Inteligente de Ocupacion Hospitalaria es un prototipo "
        "academico funcional que modela el comportamiento dinamico de un "
        "hospital colombiano durante hasta 50 horas simuladas (200 ticks de "
        "15 minutos). El sistema opera exclusivamente con datos sinteticos "
        "generados por distribuciones probabilisticas calibradas con referencia "
        "a la normativa colombiana vigente (Resolucion 5596/2015 y "
        "Resolucion 3100/2019). No procesa datos reales de pacientes en "
        "ninguna fase del ciclo de vida.", st))
    story.append(Spacer(1, 0.2*cm))
    story.append(p(
        "El sistema aborda el problema de gestion de la ocupacion hospitalaria "
        "en tiempo real: dado un flujo estocastico de llegada de pacientes con "
        "distintas prioridades clinicas (P1-critico a P4-no urgente), "
        "el simulador decide en cada tick que paciente ocupa que cama, "
        "genera alertas cuando el sistema se aproxima al colapso, y predice "
        "la ocupacion con 1 hora de antelacion para permitir intervenciones "
        "preventivas.", st))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("1.2 Componentes del sistema", st["h2"]))
    comp_data = [
        ["Componente", "Tecnologia", "Funcion principal"],
        ["Generador de pacientes sinteticos",
         "Python 3.12, numpy 2.4",
         "Genera el flujo de llegadas siguiendo distribuciones Poisson "
         "para la tasa de llegadas y distribuciones empiricas para "
         "edad, prioridad clinica, area requerida y tiempo de estancia."],
        ["Sistema experto de asignacion",
         "Python 3.12 (dataclasses)",
         "19 reglas en 6 capas (elegibilidad, priorizacion, asignacion, "
         "sobreocupacion, escalamiento, alta/limpieza). Dos modos: "
         "automatico y asistido. Alineado con Res. 5596/2015."],
        ["Indicador compuesto I",
         "Python 3.12",
         "I = 0.4*O + 0.2*E + 0.2*P + 0.2*C. Cuatro niveles: "
         "Bajo (0-25), Medio (26-50), Alto (51-75), Critico (76-100). "
         "Forzado a Critico por RES-04 si P1/P2 espera > 30 min."],
        ["Modelo predictivo",
         "scikit-learn 1.8, joblib 1.5",
         "Pipeline StandardScaler + LinearRegression. Predice O en T+4 "
         "(1 hora) con 15 features del tick actual. RMSE=2.64 pp. "
         "Seleccionado por parsimonia sobre Random Forest."],
        ["Motor de simulacion",
         "Python 3.12, numpy 2.4",
         "Orquesta el loop tick a tick. Estado reanudable "
         "(EstadoSimulacion). Snapshot del hospital para la UI. "
         "Reproduce resultados de F5 con diferencia < 2 pp."],
        ["Interfaz web",
         "Streamlit 1.45.1, matplotlib 3.10",
         "Tres paginas: configuracion de infraestructura, simulador en "
         "tiempo real (grid interactivo, graficas, modo asistido) "
         "y resultados con verificacion CE-B."],
    ]
    story.append(tabla(comp_data, [uw*0.28, uw*0.22, uw*0.50], st))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("1.3 Arquitectura de capas", st["h2"]))
    fig_arq = fig_arquitectura_capas()
    story.append(Image(fig_arq, width=uw, height=uw*0.37))
    story.append(Paragraph(
        "Figura 1. Arquitectura de cuatro capas del simulador. "
        "Las dependencias son estrictamente descendentes: la capa de "
        "presentacion no contiene logica de dominio.",
        st["caption"]))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("1.4 Modelo de datos — entidades principales", st["h2"]))
    ent_data = [
        ["Entidad", "Atributos clave", "Relaciones"],
        ["Paciente",
         "id_paciente, prioridad_clinica (P1-P4), area_requerida, "
         "tiempo_espera, tiempo_en_sistema, estado, tick_ingreso",
         "Asignado a una Cama (1:0..1)"],
        ["Cama",
         "id_cama, tipo, area_id, estado (libre/ocupada/en_limpieza), "
         "paciente_id, es_temporal, tiempo_limpieza_restante",
         "Pertenece a un Area (N:1).\nContiene 0 o 1 Paciente."],
        ["Area",
         "id_area, nombre, piso_id, capacidad_total, "
         "capacidad_disponible, prioridades_aceptadas, acepta_desborde",
         "Contiene N Camas.\nPertenece a un Piso (N:1)."],
        ["Piso",
         "id_piso, nombre, areas[]",
         "Contiene N Areas."],
        ["Hospital",
         "pisos{}, areas{}, camas{}, pacientes{} (indices O(1) por id)",
         "Contiene todos los objetos del dominio."],
    ]
    story.append(tabla(ent_data, [uw*0.14, uw*0.48, uw*0.38], st))
    story.append(PageBreak())

    # =========================================================================
    # SECCION 2 — METODOLOGIA CRISP-DM/S
    # =========================================================================
    story.append(Paragraph("2. Metodologia CRISP-DM/S Aplicada", st["h1"]))
    story.append(LineaAccento(uw))
    story.append(Spacer(1, 0.3*cm))
    story.append(p(
        "El proyecto sigue el framework CRISP-DM/S: CRISP-DM como estructura "
        "principal del ciclo de vida, SEMMA como subciclo interno en las fases "
        "de comprension del dato, preparacion y modelado, principios DAMA para "
        "el diseno del modelo de datos, y MLOps como referencia teorica para "
        "la documentacion de decisiones y trazabilidad.", st))
    story.append(Spacer(1, 0.25*cm))

    fig_crisp = fig_ciclo_crisp()
    story.append(Image(fig_crisp, width=uw, height=uw*0.41))
    story.append(Paragraph(
        "Figura 2. Las seis fases del ciclo CRISP-DM/S aplicado al proyecto "
        "con el subciclo SEMMA interno en F2-F5.",
        st["caption"]))
    story.append(Spacer(1, 0.2*cm))

    fases_data = [
        ["Fase", "SEMMA", "Entregables principales", "Criterio de salida"],
        ["F1 — Comprension del problema",
         "—",
         "Marco conceptual hospitalario colombiano. "
         "Definicion formal del problema. "
         "Criterios de exito CE-A/B/C/D.",
         "Criterios CE documentados y aprobados."],
        ["F2 — Comprension del dato simulado",
         "Sample",
         "Diccionario de datos (entidades, atributos, "
         "transiciones). Distribuciones probabilisticas. "
         "Modelo conceptual de entidades.",
         "Modelo de datos aprobado (D010)."],
        ["F3 — Preparacion del dato",
         "Explore + Modify",
         "generador_pacientes.py. EDA (eda_pacientes.py). "
         "Calibracion de distribuciones. "
         "Semilla global SEED=42 (D_F3_001).",
         "Generador produce distribuciones coherentes "
         "con el dominio colombiano."],
        ["F4-A — Modelado — Sistema experto",
         "Model",
         "sistema_experto.py (19 reglas, 6 capas). "
         "Modo automatico y asistido. "
         "Suite de 33 pruebas unitarias.",
         "33/33 pruebas pasando. "
         "Ambos modos funcionales."],
        ["F4-B — Modelado — Modelo predictivo",
         "Model",
         "dataset_f4b.parquet (2304 filas). "
         "modelo_final_f4b.pkl (RL, RMSE=2.64 pp). "
         "Comparacion RL vs RF.",
         "RMSE < 5 pp. "
         "Modelo serializado listo para F6."],
        ["F5 — Evaluacion",
         "Assess",
         "evaluar_escenarios_f5.py. "
         "3 figuras de evaluacion. "
         "PDF Fase_5_Evaluacion. "
         "Riesgo R03 mitigado.",
         "CE-B cumplido en 3 escenarios. "
         "RMSE coherente con F4-B."],
        ["F6 — Despliegue",
         "—",
         "app.py + motor_simulacion.py. "
         "Pruebas de integracion (18/18 PASS). "
         "Manual de usuario. "
         "Documentacion tecnica. "
         "Informe final.",
         "Sistema unificado operativo. "
         "Codigo documentado y reproducible."],
    ]
    story.append(tabla(fases_data,
                       [uw*0.16, uw*0.10, uw*0.44, uw*0.30], st))
    story.append(Spacer(1, 0.25*cm))
    story.append(CajaDestacada(
        "<b>Principio de fase-gate:</b> cada fase produce entregables "
        "formales en PDF con el mismo estilo visual y registro de decisiones "
        "D_Fx_NNN. No se avanza a la siguiente fase sin verificar "
        "explicitamente el criterio de salida de la fase anterior. "
        "Este principio garantiza trazabilidad completa desde F1 hasta F6."))
    story.append(PageBreak())

    # =========================================================================
    # SECCION 3 — RESULTADOS DE EVALUACION
    # =========================================================================
    story.append(Paragraph("3. Resultados de Evaluacion por Escenario", st["h1"]))
    story.append(LineaAccento(uw))
    story.append(Spacer(1, 0.3*cm))
    story.append(p(
        "La evaluacion se realizo en F5 con SEED=99 (D_F5_003), distinta "
        "a las semillas de entrenamiento del modelo predictivo "
        "(42, 123, 456, 789) para garantizar separacion entre "
        "entrenamiento y evaluacion. Se ejecutaron 200 ticks (50 horas "
        "simuladas) por escenario con warm-up de 20 ticks descartados "
        "del analisis (D_F5_006). "
        "Las pruebas de integracion de F6 reprodujeron los tres "
        "escenarios con diferencias < 2 pp respecto a F5 (18/18 PASS, "
        "D_F6_010).", st))
    story.append(Spacer(1, 0.25*cm))

    # Tabla comparativa consolidada
    story.append(Paragraph("3.1 Tabla comparativa consolidada", st["h2"]))
    consol = [
        ["Metrica",
         Paragraph("Normal (E1)<br/>Automatico", st["cell_hdr"]),
         Paragraph("Alta demanda (E2)<br/>Automatico", st["cell_hdr"]),
         Paragraph("Crisis (E3)<br/>Asistido",   st["cell_hdr"])],
        ["Lambda (pac/tick)",          "1.5",    "3.0",     "5.0"],
        ["O media",                    "43.2%",  "61.8%",   "80.6%"],
        ["O std",                      "5.3%",   "6.8%",    "9.4%"],
        ["O P5-P95",                   "37-52%", "50-70%",  "59-91%"],
        ["I media",                    "19.6",   "37.9",    "59.4"],
        ["Nivel I modal",              "Bajo",   "Medio",   "Critico"],
        ["Cola maxima (pac.)",         "0",      "4",       "27"],
        ["Traslados totales",          "38",     "271",     "504"],
        ["Alertas RES-04",             "0",      "188",     "2 563"],
        ["RMSE prediccion",            "2.689 pp","2.799 pp","2.183 pp"],
        [Paragraph("CE-B rango O", st["cell"]),
         Paragraph("OK", st["cell_ok"]),
         Paragraph("OK", st["cell_ok"]),
         Paragraph("OK", st["cell_ok"])],
        [Paragraph("CE-B nivel I", st["cell"]),
         Paragraph("OK", st["cell_ok"]),
         Paragraph("OK", st["cell_ok"]),
         Paragraph("OK", st["cell_ok"])],
    ]
    t_consol = Table(
        [[Paragraph(str(cell) if not isinstance(cell, Paragraph) else "",
                    st["cell_hdr"] if r_idx == 0 else st["cell"])
          if not isinstance(cell, Paragraph) else cell
          for cell in row]
         for r_idx, row in enumerate(consol)],
        colWidths=[uw*0.32, uw*0.22, uw*0.24, uw*0.22],
        repeatRows=1,
    )
    t_consol.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1,  0), TABLE_HDR),
        ("VALIGN",         (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, TABLE_ALT]),
        ("GRID",           (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
        ("LEFTPADDING",    (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 6),
    ]))
    story.append(t_consol)
    story.append(Spacer(1, 0.25*cm))

    story.append(Paragraph(
        "Fuente: Fase_5_Evaluacion.pdf, Seccion 5 — "
        "Tabla Comparativa Consolidada (SEED=99).",
        st["caption"]))

    # Figura de resultados
    fig_res = fig_resultados_escenarios()
    story.append(Image(fig_res, width=uw, height=uw*0.315))
    story.append(Paragraph(
        "Figura 3. Progresion monotona de O media, I media, traslados "
        "totales y alertas RES-04 en los tres escenarios. "
        "La escala de alertas en E3 confirma el colapso simulado.",
        st["caption"]))
    story.append(Spacer(1, 0.25*cm))

    story.append(Paragraph("3.2 Analisis por escenario", st["h2"]))

    analisis_esc = [
        ("Normal (E1) — Modo automatico",
         "Con lambda=1.5 pac/tick el sistema opera en regimen estable "
         "con O media=43.2%, coherente con el equilibrio matematico "
         "predicho por la Ley de Little (L=lambda*W, W~7.9h, O_eq~50%). "
         "El indicador I se mantuvo en nivel Bajo el 96% del tiempo "
         "(I media=19.6). La cola de espera fue cero en todos los ticks "
         "del regimen estable y no se activo ninguna alerta RES-04, "
         "confirmando que el sistema experto gestiona correctamente "
         "la carga normal sin intervencion."),
        ("Alta demanda (E2) — Modo automatico",
         "Con lambda=3.0 pac/tick la ocupacion media sube a 61.8% "
         "(dentro del rango ajustado [60%, 88%], D_F5_008). "
         "El nivel I alterna entre Medio (63% de los ticks) y Critico "
         "(37%), siendo el nivel Critico activado no por el valor "
         "numerico de I sino por la regla RES-04: en 188 ticks hubo "
         "pacientes P1/P2 que superaron el umbral de 2 ticks en espera. "
         "El sistema activo 271 traslados en 180 ticks de regimen, "
         "con pico de 13 traslados por tick, demostrando que las reglas "
         "RSO-01/02 funcionan como se especifico en F4-A."),
        ("Crisis (E3) — Modo asistido",
         "Con lambda=5.0 pac/tick la ocupacion media alcanza 80.6% "
         "(rango [80%, 100%], D_F5_008) con alta variabilidad "
         "(std=9.4%). La cola maxima llego a 27 pacientes y el "
         "componente P (proporcion en desborde) alcanzo media=46.2%, "
         "confirmando la activacion sistematica de RSO-03 "
         "(camas temporales). Se generaron 2.563 alertas RES-04 "
         "indicando saturacion critica sostenida. El modo asistido "
         "genero 504 propuestas de accion confirmadas, validando el "
         "criterio CE-D1 (al menos 2 opciones de redistribucion "
         "cuando I >= 76)."),
    ]
    for titulo_esc, texto_esc in analisis_esc:
        story.append(KeepTogether([
            p(f"<b>{titulo_esc}.</b> {texto_esc}", st),
            Spacer(1, 0.15*cm),
        ]))
    story.append(PageBreak())

    # =========================================================================
    # SECCION 4 — MODELO PREDICTIVO
    # =========================================================================
    story.append(Paragraph("4. Analisis del Modelo Predictivo", st["h1"]))
    story.append(LineaAccento(uw))
    story.append(Spacer(1, 0.3*cm))

    story.append(p(
        "El modelo predictivo es un Pipeline de scikit-learn compuesto por "
        "StandardScaler y LinearRegression. Se selecciono por parsimonia "
        "sobre Random Forest: la diferencia de RMSE entre ambos fue de "
        "0.14 pp, inferior al umbral de 1.0 pp definido en D_F4B_010. "
        "El modelo fue entrenado con 2.304 filas de 15 features "
        "(3 escenarios x 4 semillas x 196 ventanas por simulacion) "
        "y validado con split 80/20 estratificado por escenario.", st))
    story.append(Spacer(1, 0.2*cm))

    modelo_data = [
        ["Parametro", "Valor"],
        ["Algoritmo",            "Regresion Lineal Multiple (sklearn.LinearRegression)"],
        ["Pipeline completo",    "StandardScaler + LinearRegression"],
        ["Variable objetivo",    "O_{t+4}: ocupacion global en T+4 (horizonte 1 hora)"],
        ["Features",             "15: componentes O/E/P/C/I, contadores del tick, "
                                 "4 lags de O, 2 one-hot de escenario"],
        ["RMSE (validacion F4-B)","2.643 pp"],
        ["RMSE (evaluacion F5) — Normal",       "2.689 pp (+0.046 pp vs F4-B)"],
        ["RMSE (evaluacion F5) — Alta demanda", "2.799 pp (+0.156 pp vs F4-B)"],
        ["RMSE (evaluacion F5) — Crisis",       "2.183 pp (-0.460 pp vs F4-B)"],
        ["Interpretacion crisis",
         "El rango saturado (80-93%) reduce la varianza del target, "
         "facilitando la prediccion lineal en ese escenario."],
        ["Semilla",              "SEED=42 (D_F3_001)"],
        ["Feature mas importante (RF)",
         "O_t: 91.3% de importancia Gini. Confirma alta inercia del proceso."],
        ["Umbral clinico aceptable", "RMSE < 5 pp (definido en F4-B)"],
    ]
    story.append(tabla(modelo_data, [uw*0.32, uw*0.68], st))
    story.append(Spacer(1, 0.2*cm))
    story.append(p(
        "Los tres valores de RMSE en evaluacion (F5) son coherentes con "
        "el valor de validacion (F4-B, RMSE=2.643 pp), confirmando "
        "ausencia de sobreajuste y buena generalizacion a semillas y "
        "escenarios no vistos durante el entrenamiento. "
        "Esta coherencia fue el criterio que permitio declarar "
        "el riesgo R03 (distribuciones no representativas) como "
        "MITIGADO al cierre de F5 (D_F5_009).", st))
    story.append(PageBreak())

    # =========================================================================
    # SECCION 5 — LIMITACIONES
    # =========================================================================
    story.append(Paragraph("5. Limitaciones del Prototipo", st["h1"]))
    story.append(LineaAccento(uw))
    story.append(Spacer(1, 0.3*cm))
    story.append(p(
        "Las limitaciones documentadas a continuacion son inherentes "
        "al alcance de un prototipo academico individual (PMV). "
        "No representan fallas de diseno sino decisiones explicitas "
        "de alcance registradas como riesgos o decisiones en las "
        "fases correspondientes.", st))
    story.append(Spacer(1, 0.2*cm))

    lim_data = [
        ["Limitacion", "Impacto operativo", "Decision / Riesgo"],
        ["Datos 100% sinteticos. "
         "Las distribuciones de edad, prioridad y estancia son "
         "aproximaciones empiricas sin calibracion con datos "
         "historicos reales de hospitales colombianos.",
         "Los resultados cuantitativos (O media, RMSE) no son "
         "extrapolables directamente a hospitales reales sin "
         "recalibracion.",
         "R03 mitigado en F5 dentro del contexto sintetico. "
         "Pendiente calibracion real (TF-03)."],
        ["Lambda fija por escenario. "
         "La tasa de llegadas es constante durante toda la simulacion.",
         "Un hospital real tiene variaciones diurnas, semanales "
         "y estacionales que el modelo actual no captura.",
         "D012 (F1): simplificacion justificada para el PMV. "
         "Generalizacion planificada en TF-02."],
        ["Topologia de pisos y areas fija (D008). "
         "El usuario puede configurar el numero de camas por area "
         "pero no agregar nuevas areas o pisos.",
         "Limita la aplicabilidad a hospitales con una "
         "distribucion de areas distinta al modelo de referencia.",
         "D_F6_009: declarado trabajo futuro TF-01. "
         "Requiere refactorizacion de reglas RSO-02."],
        ["Modelo predictivo lineal. "
         "La Regresion Lineal captura bien la inercia del proceso "
         "pero no modela el efecto techo al 100% de ocupacion "
         "ni los umbrales del indicador I.",
         "En crisis el modelo subestima la ocupacion cuando "
         "esta se aproxima al techo de capacidad.",
         "D_F4B_010: parsimonia sobre RF (diff=0.14 pp < 1 pp). "
         "Modelo no lineal planificado en TF-05."],
        ["Un solo usuario concurrente. "
         "Streamlit en modo local no esta disenado para "
         "multiples sesiones simultaneas.",
         "Uso limitado a entorno academico individual. "
         "No apto para despliegue institucional sin modificaciones.",
         "Alcance PMV. Despliegue multi-usuario planificado en TF-06."],
        ["Sin persistencia entre sesiones. "
         "Al cerrar el navegador el estado de la simulacion se pierde.",
         "El usuario no puede retomar una simulacion interrumpida.",
         "Alcance PMV. Guardado de sesiones planificado en TF-04."],
    ]
    story.append(tabla(lim_data, [uw*0.38, uw*0.36, uw*0.26], st))
    story.append(PageBreak())

    # =========================================================================
    # SECCION 6 — TRABAJO FUTURO Y NORMOGRAMA
    # =========================================================================
    story.append(Paragraph(
        "6. Trabajo Futuro y Normograma Colombiano", st["h1"]))
    story.append(LineaAccento(uw))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph(
        "6.1 Requisitos previos para despliegue real", st["h2"]))
    story.append(p(
        "Un despliegue productivo de este simulador en un hospital "
        "colombiano real requiere cumplir un conjunto de requisitos "
        "normativos, tecnicos y de datos que trascienden el alcance "
        "del prototipo academico. Se describen a continuacion "
        "organizados por el marco normativo colombiano aplicable.", st))
    story.append(Spacer(1, 0.25*cm))

    norma_data = [
        ["Norma / Marco", "Requisito para despliegue real", "Estado en prototipo"],
        ["Resolucion 5596/2015\n(triage hospitalario)",
         "Las reglas de priorizacion P1-P4 deben estar alineadas con "
         "el modelo de triage de 5 niveles de la resolucion. "
         "La regla RSO-04 (P1 sin UCI permanece en espera) debe "
         "revisarse contra los tiempos maximos de atencion por nivel.",
         "Parcialmente alineado. "
         "El modelo usa 4 niveles (P1-P4) como aproximacion al "
         "sistema de 5 niveles. Requiere mapeo formal."],
        ["Resolucion 3100/2019\n(habilitacion de servicios)",
         "La configuracion de areas y camas debe reflejar la "
         "dotacion real habilitada del establecimiento "
         "(numero de camas UCI, urgencias, hospitalizacion, etc.).",
         "Configurable desde la UI (D_F6_008). "
         "Requiere integracion con el REPS "
         "(Registro Especial de Prestadores)."],
        ["Resolucion 866/2021\n(historia clinica digital)",
         "Si el sistema consume o produce datos de pacientes reales, "
         "debe articularse con el sistema de historia clinica digital "
         "siguiendo los estandares de interoperabilidad HL7 FHIR.",
         "No aplica al prototipo (datos sinteticos). "
         "Requisito critico para TF-07."],
        ["Ley 1581/2012\n(proteccion de datos personales)",
         "Cualquier uso de datos reales de pacientes requiere "
         "tratamiento conforme a la politica de datos personales, "
         "incluyendo anonimizacion, consentimiento informado y "
         "registro ante la SIC.",
         "No aplica al prototipo. "
         "Prerequisito para TF-03 (calibracion con datos reales)."],
        ["Circular 047/2007 MinSalud\n(indicadores hospitalarios)",
         "Los indicadores de gestion hospitalaria usados "
         "(ocupacion, estancia promedio, rotacion de camas) deben "
         "alinearse con las definiciones oficiales del ministerio.",
         "Parcialmente alineado. "
         "O y el indicador I usan definiciones propias. "
         "Requiere mapeo formal en TF-03."],
        ["CONPES 3975/2019\n(politica nacional de IA)",
         "El uso de sistemas de IA en salud debe seguir los principios "
         "de transparencia, explicabilidad, equidad y auditabilidad "
         "del marco de IA del gobierno colombiano.",
         "Prototipo academico: "
         "decisiones documentadas (D_Fx), reglas auditables, "
         "sin sesgos demograficos (R09 mitigado)."],
    ]
    story.append(tabla(norma_data, [uw*0.22, uw*0.46, uw*0.32], st))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("6.2 Hoja de ruta de trabajo futuro", st["h2"]))
    tf_data = [
        ["ID", "Mejora", "Prerequisito", "Prioridad"],
        ["TF-01",
         "Configuracion dinamica de pisos y areas",
         "Refactorizar reglas RSO-02 para topologias arbitrarias.",
         "Media"],
        ["TF-02",
         "Lambda variable con patron diurno/nocturno",
         "Extender generador_pacientes con funcion lambda(t).",
         "Alta"],
        ["TF-03",
         "Calibracion con datos historicos reales anonimizados",
         "Acceso a datos bajo Res. 3100/2019 y Ley 1581/2012. "
         "Ajuste de distribuciones en F3.",
         "Alta"],
        ["TF-04",
         "Guardado y carga de sesiones de simulacion",
         "Serializar EstadoSimulacion. "
         "UI de gestion de sesiones en Streamlit.",
         "Media"],
        ["TF-05",
         "Modelo predictivo no lineal (Gradient Boosting)",
         "Dataset mayor (>10.000 filas). "
         "Ajuste de hiperparametros con CV.",
         "Baja"],
        ["TF-06",
         "Despliegue productivo multi-usuario",
         "Estado en BD (Redis/PostgreSQL). "
         "Autenticacion OAuth. Docker.",
         "Alta (v2.0)"],
        ["TF-07",
         "Integracion con HIS/EHR reales (HL7 FHIR)",
         "API FHIR. Cumplimiento Res. 866/2021.",
         "Alta (v2.0)"],
    ]
    story.append(tabla(tf_data, [uw*0.10, uw*0.27, uw*0.43, uw*0.20], st))
    story.append(PageBreak())

    # =========================================================================
    # SECCION 7 — CRITERIO DE SALIDA F6
    # =========================================================================
    story.append(Paragraph(
        "7. Criterio de Salida de F6 — Verificacion Final", st["h1"]))
    story.append(LineaAccento(uw))
    story.append(Spacer(1, 0.3*cm))

    ce_data = [
        ["Condicion", "Referencia", "Evidencia", "Estado"],
        ["Todos los modulos operan correctamente "
         "como sistema unificado",
         "CE-F6-01",
         "18/18 checks PASS en pruebas de integracion. "
         "O_media F6 vs F5: diff < 2 pp en los 3 escenarios.",
         Paragraph("CUMPLIDO", st["cell_ok"])],
        ["El sistema puede ser operado por un usuario "
         "sin conocimiento tecnico previo",
         "CE-F6-02",
         "Manual de usuario (Entregable 3): 15 paginas, "
         "10 secciones, sin jerga tecnica. "
         "Pantalla de configuracion con vista previa interactiva.",
         Paragraph("CUMPLIDO", st["cell_ok"])],
        ["El codigo esta documentado y es reproducible",
         "CE-F6-03",
         "Documentacion tecnica (Entregable 4): API publica, "
         "diagrama de dependencias, 22 decisiones de diseno, "
         "instrucciones de reproduccion. "
         "Verificacion de reproducibilidad en pruebas F6.",
         Paragraph("CUMPLIDO", st["cell_ok"])],
        ["El proyecto esta documentado segun los "
         "estandares del curso",
         "CE-F6-04",
         "Informe final (este documento): descripcion del sistema, "
         "metodologia CRISP-DM/S, resultados por escenario, "
         "limitaciones y normograma.",
         Paragraph("CUMPLIDO", st["cell_ok"])],
        ["Entregables F6 producidos: "
         "1) Integracion web, "
         "2) Pruebas de integracion, "
         "3) Manual de usuario, "
         "4) Documentacion tecnica, "
         "5) Informe final",
         "CE-F6-05",
         "app.py + motor_simulacion.py, "
         "pruebas_integracion_f6.pdf (18/18 PASS), "
         "manual_usuario_f6.pdf, "
         "documentacion_tecnica_f6.pdf, "
         "informe_final_f6.pdf (este documento).",
         Paragraph("CUMPLIDO", st["cell_ok"])],
        ["Riesgos R02, R03, R09 mitigados",
         "F4-A, F5",
         "R02: funciones atomicas + 33 pruebas unitarias. "
         "R03: RMSE eval coherente con F4-B, CE-B OK. "
         "R09: ningun atributo demografico en reglas del SE.",
         Paragraph("CUMPLIDO", st["cell_ok"])],
    ]
    t_ce = Table(
        [[Paragraph(str(c), st["cell_hdr"] if r == 0 else st["cell"])
          if not isinstance(c, Paragraph) else c
          for c in row]
         for r, row in enumerate(ce_data)],
        colWidths=[uw*0.35, uw*0.10, uw*0.40, uw*0.15],
        repeatRows=1,
    )
    t_ce.setStyle(TableStyle([
        ("BACKGROUND",     (0,0), (-1, 0), TABLE_HDR),
        ("VALIGN",         (0,0), (-1,-1), "TOP"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, TABLE_ALT]),
        ("GRID",           (0,0), (-1,-1), 0.3, colors.HexColor("#CCCCCC")),
        ("TOPPADDING",     (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 5),
        ("LEFTPADDING",    (0,0), (-1,-1), 6),
        ("RIGHTPADDING",   (0,0), (-1,-1), 6),
    ]))
    story.append(t_ce)
    story.append(Spacer(1, 0.3*cm))
    story.append(CajaDestacada(
        "<b>Fase 6 completada. Criterio de salida verificado.</b> "
        "El simulador integrado produce resultados coherentes con F5 "
        "(18/18 PASS), opera correctamente con los tres escenarios "
        "de evaluacion, puede ser configurado y ejecutado sin "
        "conocimiento tecnico previo, y cuenta con documentacion "
        "completa reproducible. El ciclo CRISP-DM/S se declara "
        "completado a nivel de prototipo academico funcional (PMV)."))
    story.append(PageBreak())

    # =========================================================================
    # SECCION 8 — REFERENCIAS NORMATIVAS
    # =========================================================================
    story.append(Paragraph("8. Referencias Normativas", st["h1"]))
    story.append(LineaAccento(uw))
    story.append(Spacer(1, 0.3*cm))

    refs = [
        ("[1]",
         "Ministerio de Salud y Proteccion Social de Colombia (2015). "
         "Resolucion 5596 de 2015: Criterios para la clasificacion de "
         "usuarios en los servicios de urgencias. Bogota, Colombia."),
        ("[2]",
         "Ministerio de Salud y Proteccion Social de Colombia (2019). "
         "Resolucion 3100 de 2019: Procedimientos y condiciones de "
         "inscripcion de los prestadores de servicios de salud. "
         "Bogota, Colombia."),
        ("[3]",
         "Ministerio de Salud y Proteccion Social de Colombia (2021). "
         "Resolucion 866 de 2021: Caracteristicas del sistema de "
         "historia clinica electronica. Bogota, Colombia."),
        ("[4]",
         "Congreso de la Republica de Colombia (2012). "
         "Ley 1581 de 2012: Disposiciones generales para la proteccion "
         "de datos personales. Bogota, Colombia."),
        ("[5]",
         "Consejo Nacional de Politica Economica y Social (2019). "
         "CONPES 3975: Politica nacional para la transformacion digital "
         "e inteligencia artificial. Bogota, Colombia."),
        ("[6]",
         "Chapman, P., Clinton, J., Kerber, R., Khabaza, T., Reinartz, T., "
         "Shearer, C., & Wirth, R. (2000). "
         "CRISP-DM 1.0: Step-by-step data mining guide. "
         "SPSS Inc."),
        ("[7]",
         "SAS Institute (1998). "
         "SEMMA: Sample, Explore, Modify, Model, Assess. "
         "SAS Institute White Paper."),
        ("[8]",
         "DAMA International (2017). "
         "DAMA-DMBOK: Data Management Body of Knowledge (2nd ed.). "
         "Technics Publications."),
        ("[9]",
         "Pedregosa, F. et al. (2011). "
         "Scikit-learn: Machine Learning in Python. "
         "Journal of Machine Learning Research, 12, 2825-2830."),
        ("[10]",
         "Little, J. D. C. (1961). "
         "A proof for the queuing formula: L = lambda W. "
         "Operations Research, 9(3), 383-387."),
    ]
    ref_data = [["#", "Referencia"]] + [[num, desc] for num, desc in refs]
    story.append(tabla(ref_data, [uw*0.06, uw*0.94], st))

    def primera_pagina(canvas, doc):
        dibujar_portada(canvas, doc)
    doc.build(story, onFirstPage=primera_pagina, onLaterPages=hf)
    print(f"PDF generado: {ruta_salida}")


def main():
    print("Generando Informe Final F6...")
    ruta = os.path.join(BASE_DIR, "informe_final_f6.pdf")
    construir_pdf(ruta)
    for tmp in ["_tmp_crisp.png", "_tmp_resultados.png", "_tmp_arq.png"]:
        ruta_tmp = os.path.join(BASE_DIR, tmp)
        if os.path.exists(ruta_tmp):
            os.remove(ruta_tmp)
    print("Listo.")

if __name__ == "__main__":
    main()
