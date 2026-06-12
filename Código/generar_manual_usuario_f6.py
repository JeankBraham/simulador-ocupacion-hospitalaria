"""
generar_manual_usuario_f6.py
============================
Simulador Inteligente de Ocupación Hospitalaria
Fase 6 — Entregable 3: Manual de Usuario

Autor  : Juan Camilo García Braham
Curso  : IA en Salud · Maestría IA y CD · UTP
Año    : 2026
Stack  : Python 3.12 · reportlab 4.4

Uso:
    python generar_manual_usuario_f6.py

Produce:
    manual_usuario_f6.pdf
"""

import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

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

# ─── Paleta ───────────────────────────────────────────────────────────────────
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
TIP_BG     = colors.HexColor("#E8F5E9")
TIP_BOR    = colors.HexColor("#4CAF50")
WARN_BG    = colors.HexColor("#FFF3E0")
WARN_BOR   = colors.HexColor("#FF9800")
INFO_BG    = colors.HexColor("#E3F2FD")
INFO_BOR   = colors.HexColor("#1565C0")

W, H   = letter
MARGIN = 2.0 * cm

# ─── Flowables personalizados ─────────────────────────────────────────────────
class LineaAccento(Flowable):
    def __init__(self, w, color=ACCENT, h=2):
        super().__init__()
        self.w, self.color, self.h = w, color, h
    def draw(self):
        self.canv.setFillColor(self.color)
        self.canv.rect(0, 0, self.w, self.h, fill=1, stroke=0)
    def wrap(self, *_): return self.w, self.h + 4

def CajaInfo(texto, tipo="info", ancho=None):
    """Caja coloreada para notas, consejos y advertencias.

    Implementada como funcion que retorna una Table con Paragraph interno.
    El wrap lo gestiona ReportLab — sin estimacion manual de altura.
    """
    aw = ancho or (W - 2 * MARGIN)
    cfg = {
        "info": (INFO_BG,  INFO_BOR,  "[i]  "),
        "tip":  (TIP_BG,   TIP_BOR,   "[OK] "),
        "warn": (WARN_BG,  WARN_BOR,  "[!]  "),
    }
    bg, border, prefijo = cfg.get(tipo, cfg["info"])
    txt_color = colors.HexColor("#1a3a1a") if tipo == "tip" else \
                colors.HexColor("#3a2800") if tipo == "warn" else \
                colors.HexColor("#0d2a4a")

    estilo_caja = ParagraphStyle(
        f"caja_{tipo}",
        fontName="Helvetica",
        fontSize=8.5,
        textColor=txt_color,
        leading=13,
        wordWrap="LTR",
        splitLongWords=True,
    )
    contenido = f"<b>{prefijo}</b>{texto}"
    parrafo = Paragraph(contenido, estilo_caja)

    t = Table([[parrafo]], colWidths=[aw - 0.6*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), bg),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("BOX",           (0, 0), (-1, -1), 1.2, border),
        ("ROUNDEDCORNERS", [5]),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    return t

# ─── Estilos ──────────────────────────────────────────────────────────────────
def estilos():
    return {
        "titulo_cover": ParagraphStyle("titulo_cover", fontName="Helvetica-Bold",
            fontSize=22, textColor=WHITE, alignment=TA_CENTER, leading=28, spaceAfter=6),
        "sub_cover": ParagraphStyle("sub_cover", fontName="Helvetica",
            fontSize=13, textColor=ACCENT, alignment=TA_CENTER, leading=18, spaceAfter=4),
        "meta_cover": ParagraphStyle("meta_cover", fontName="Helvetica",
            fontSize=10, textColor=LIGHT_GRAY, alignment=TA_CENTER, leading=14),
        "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=13,
            textColor=DARK_BG, spaceBefore=14, spaceAfter=6, leading=16),
        "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11,
            textColor=BLUE_DEC, spaceBefore=10, spaceAfter=4, leading=14),
        "h3": ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=10,
            textColor=colors.HexColor("#333333"),
            spaceBefore=6, spaceAfter=3, leading=13),
        "paso": ParagraphStyle("paso", fontName="Helvetica-Bold", fontSize=10,
            textColor=colors.HexColor("#1a1a2e"), leading=14,
            spaceBefore=8, spaceAfter=2,
            borderPad=4, backColor=colors.HexColor("#EDE7F6"),
            leftIndent=6),
        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9,
            textColor=colors.HexColor("#222222"),
            leading=13, spaceAfter=4, alignment=TA_JUSTIFY),
        "body_bullet": ParagraphStyle("body_bullet", fontName="Helvetica", fontSize=9,
            textColor=colors.HexColor("#222222"),
            leading=13, spaceAfter=2, leftIndent=14, bulletIndent=4),
        "caption": ParagraphStyle("caption", fontName="Helvetica-Oblique",
            fontSize=8, textColor=MID_GRAY, alignment=TA_CENTER,
            leading=10, spaceAfter=4),
        "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=8,
            textColor=colors.HexColor("#222222"), leading=11,
            wordWrap="LTR", splitLongWords=True),
        "cell_hdr": ParagraphStyle("cell_hdr", fontName="Helvetica-Bold", fontSize=8,
            textColor=WHITE, leading=11, alignment=TA_CENTER,
            wordWrap="LTR"),
        "cell_c": ParagraphStyle("cell_c", fontName="Helvetica", fontSize=8,
            textColor=colors.HexColor("#222222"), leading=11,
            alignment=TA_CENTER, wordWrap="LTR"),
        "num_paso": ParagraphStyle("num_paso", fontName="Helvetica-Bold",
            fontSize=18, textColor=ACCENT, leading=22, alignment=TA_CENTER),
    }

def p(texto, st, estilo="body"):
    if isinstance(texto, Paragraph): return texto
    return Paragraph(str(texto), st[estilo])

def tabla(datos, col_widths, st, hdr_bg=TABLE_HDR):
    rows_p = []
    for r_idx, row in enumerate(datos):
        rows_p.append([
            cell if isinstance(cell, Paragraph)
            else Paragraph(str(cell), st["cell_hdr"] if r_idx == 0 else st["cell"])
            for cell in row
        ])
    t = Table(rows_p, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1,  0), hdr_bg),
        ("VALIGN",         (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, TABLE_ALT]),
        ("GRID",           (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
        ("LEFTPADDING",    (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 6),
    ]))
    return t

def paso_numerado(num, titulo, st):
    """Retorna una fila de tabla que actúa como encabezado numerado de paso."""
    fila = Table(
        [[Paragraph(str(num), st["num_paso"]),
          Paragraph(titulo, ParagraphStyle("tit_paso",
              fontName="Helvetica-Bold", fontSize=11,
              textColor=DARK_BG, leading=14, leftIndent=4))]],
        colWidths=[1.2*cm, W - 2*MARGIN - 1.2*cm],
    )
    fila.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), colors.HexColor("#EDE7F6")),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ("LEFTPADDING",  (0, 0), (0,  0),  4),
        ("ROUNDEDCORNERS", [4]),
    ]))
    return fila

# ─── Header / Footer ──────────────────────────────────────────────────────────
def _hf(canvas, doc, seccion=""):
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
                           f"Manual de Usuario  |  {seccion}")
    canvas.setFillColor(DARK_BG)
    canvas.rect(0, 0, W, 1.0*cm, fill=1, stroke=0)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MID_GRAY)
    canvas.drawString(MARGIN, 0.35*cm, "Juan Camilo García Braham  |  UTP 2026")
    canvas.setFillColor(ACCENT)
    canvas.drawRightString(W - MARGIN, 0.35*cm, f"Página {doc.page}")
    canvas.restoreState()

# ─── Figuras ilustrativas ─────────────────────────────────────────────────────
def fig_flujo_paginas() -> str:
    """Diagrama de flujo de las 3 páginas de la app."""
    fig, ax = plt.subplots(figsize=(10, 2.4), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.axis("off")

    cajas = [
        (1.2, "(1) Configurar\nhospital", "#2a2a4a", "#BB86FC"),
        (4.5, "(2) Simulador\nen curso",  "#2a2a4a", "#03DAC6"),
        (7.8, "(3) Resultados\nfinales",  "#2a2a4a", "#4CAF50"),
    ]
    for x, txt, bg, bord in cajas:
        rect = mpatches.FancyBboxPatch((x - 1.0, 0.3), 2.0, 1.4,
            boxstyle="round,pad=0.1", facecolor=bg,
            edgecolor=bord, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, 1.0, txt, ha="center", va="center",
                color="white", fontsize=9, fontweight="bold",
                fontfamily="monospace")

    for x_start, x_end in [(2.2, 3.3), (5.5, 6.6)]:
        ax.annotate("", xy=(x_end, 1.0), xytext=(x_start, 1.0),
                    arrowprops=dict(arrowstyle="->", color="#BB86FC",
                                   lw=2.0))
    ax.annotate("[ Iniciar simulacion ]", xy=(3.85, 1.1),
                color="#aaa", fontsize=7, ha="center")
    ax.annotate("Ver resultados", xy=(6.05, 1.1),
                color="#aaa", fontsize=7, ha="center")

    ax.set_xlim(0, 9); ax.set_ylim(0, 1.8)
    ruta = os.path.join(BASE_DIR, "_tmp_flujo.png")
    fig.tight_layout(pad=0.3)
    fig.savefig(ruta, dpi=130, bbox_inches="tight",
                facecolor="#1a1a2e", edgecolor="none")
    plt.close(fig)
    return ruta

def fig_grid_hospital() -> str:
    """Ilustración esquemática del grid de camas."""
    fig, ax = plt.subplots(figsize=(10, 3.2), facecolor="#16213e")
    ax.set_facecolor("#16213e")
    ax.axis("off")

    areas = [
        ("UCI",             10, 0.0,  "#F44336", "#4CAF50"),
        ("Urgencias",       20, 1.0,  "#F44336", "#4CAF50"),
        ("Hospitalización", 40, 2.0,  "#4CAF50", "#F44336"),
        ("Observación",     15, 3.2,  "#FF9800", "#4CAF50"),
    ]
    colores_estado = {"libre": "#4CAF50", "ocupada": "#F44336",
                      "limpieza": "#FF9800", "temporal": "#BB86FC"}

    np.random.seed(42)
    for nombre, n_camas, y_off, c1, c2 in areas:
        ax.text(0.15, 2.55 - y_off, nombre, color="#BB86FC",
                fontsize=8, fontweight="bold", va="center")
        estados = np.random.choice(
            ["ocupada", "libre", "limpieza", "temporal"],
            size=min(n_camas, 30),
            p=[0.55, 0.30, 0.10, 0.05]
        )
        for i, est in enumerate(estados):
            col = colores_estado[est]
            x = 1.2 + i * 0.28
            rect = mpatches.FancyBboxPatch(
                (x, 2.3 - y_off), 0.22, 0.22,
                boxstyle="round,pad=0.02",
                facecolor=col, edgecolor="#333", linewidth=0.5,
                alpha=0.9
            )
            ax.add_patch(rect)
        if n_camas > 30:
            ax.text(1.2 + 30 * 0.28 + 0.1, 2.41 - y_off,
                    f"+{n_camas-30}", color="#666", fontsize=7)

    # Leyenda
    leyenda_items = [
        ("Libre", "#4CAF50"), ("Ocupada", "#F44336"),
        ("En limpieza", "#FF9800"), ("Temporal", "#BB86FC"),
    ]
    for i, (lbl, col) in enumerate(leyenda_items):
        x_l = 0.15 + i * 2.3
        rect = mpatches.FancyBboxPatch((x_l, -0.25), 0.22, 0.18,
            boxstyle="round,pad=0.02", facecolor=col, edgecolor="#333", linewidth=0.5)
        ax.add_patch(rect)
        ax.text(x_l + 0.28, -0.16, lbl, color="#ccc", fontsize=7.5, va="center")

    ax.set_xlim(0, 10); ax.set_ylim(-0.5, 3.0)
    ruta = os.path.join(BASE_DIR, "_tmp_grid.png")
    fig.tight_layout(pad=0.4)
    fig.savefig(ruta, dpi=130, bbox_inches="tight",
                facecolor="#16213e", edgecolor="none")
    plt.close(fig)
    return ruta

def fig_indicador_I() -> str:
    """Ilustración de los 4 niveles del indicador I."""
    fig, ax = plt.subplots(figsize=(9, 1.8), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.axis("off")

    bandas = [
        (0,  25,  "#4CAF50", "BAJO\n0–25"),
        (25, 50,  "#FF9800", "MEDIO\n26–50"),
        (50, 75,  "#FF5722", "ALTO\n51–75"),
        (75, 100, "#F44336", "CRÍTICO\n76–100"),
    ]
    for x0, x1, col, lbl in bandas:
        ax.barh(0, x1 - x0, left=x0, height=0.7,
                color=col, alpha=0.8, edgecolor="none")
        ax.text((x0 + x1) / 2, 0, lbl, ha="center", va="center",
                color="white", fontsize=8.5, fontweight="bold")

    # Flecha indicativa
    ax.annotate("", xy=(43, 0.55), xytext=(43, 0.9),
                arrowprops=dict(arrowstyle="->", color="white", lw=1.5))
    ax.text(43, 1.0, "Ejemplo: I=43 -> Medio", color="#ccc",
            fontsize=7.5, ha="center")

    ax.set_xlim(-2, 105); ax.set_ylim(-0.5, 1.5)
    ruta = os.path.join(BASE_DIR, "_tmp_indicador.png")
    fig.tight_layout(pad=0.3)
    fig.savefig(ruta, dpi=130, bbox_inches="tight",
                facecolor="#1a1a2e", edgecolor="none")
    plt.close(fig)
    return ruta

# ─── CONSTRUCCIÓN DEL PDF ─────────────────────────────────────────────────────
def construir_manual(ruta_salida: str):
    st = estilos()
    uw = W - 2 * MARGIN

    doc = SimpleDocTemplate(
        ruta_salida, pagesize=letter,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=1.6*cm, bottomMargin=1.4*cm,
    )

    def hf(canvas, doc): _hf(canvas, doc, "Guía de uso")

    story = []

    # ── PORTADA ───────────────────────────────────────────────────────────────
    story.append(Spacer(1, 2.8*cm))
    story.append(Paragraph(
        "SIMULADOR INTELIGENTE DE<br/>OCUPACI&Oacute;N HOSPITALARIA",
        st["titulo_cover"]))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Manual de Usuario", st["sub_cover"]))
    story.append(Spacer(1, 0.15*cm))
    story.append(Paragraph("Fase 6 — Despliegue · Entregable 3", st["sub_cover"]))
    story.append(Spacer(1, 0.8*cm))
    story.append(LineaAccento(uw))
    story.append(Spacer(1, 0.5*cm))
    for txt in [
        "Proyecto  Prototipo académico funcional · Curso IA en Salud",
        "Autor  Juan Camilo García Braham",
        "Programa  Maestría en IA y Ciencia de Datos",
        "Institución  Universidad Tecnológica de Pereira (UTP)",
        "Marco  CRISP-DM/S · SEMMA · DAMA · MLOps",
        "Año  2026",
    ]:
        story.append(Paragraph(txt, st["meta_cover"]))
    story.append(Spacer(1, 1.2*cm))
    story.append(CajaInfo(
        "Este manual está dirigido a usuarios sin conocimiento técnico previo. "
        "No se requiere saber programación para operar el simulador.",
        tipo="tip"))
    story.append(PageBreak())

    # ── TABLA DE CONTENIDO ────────────────────────────────────────────────────
    story.append(Paragraph("Tabla de Contenido", st["h1"]))
    story.append(LineaAccento(uw))
    story.append(Spacer(1, 0.3*cm))
    toc_data = [
        ["Sección", "Contenido", "Pág."],
        ["1", "Introducción — ¿Qué es el simulador?", "3"],
        ["2", "Requisitos e instalación", "3"],
        ["3", "Inicio rápido — primera simulación en 3 pasos", "4"],
        ["4", "Pantalla 1 — Configurar el hospital", "5"],
        ["5", "Pantalla 2 — Simulador en curso", "7"],
        ["6", "Pantalla 3 — Resultados finales", "11"],
        ["7", "Modo asistido — guía paso a paso", "12"],
        ["8", "Interpretación del indicador I", "13"],
        ["9", "Referencia rápida de controles", "14"],
        ["10", "Preguntas frecuentes", "15"],
    ]
    story.append(tabla(toc_data, [uw*0.08, uw*0.74, uw*0.18], st))
    story.append(PageBreak())

    # ── SECCIÓN 1 — INTRODUCCIÓN ──────────────────────────────────────────────
    story.append(Paragraph("1. Introducción — ¿Qué es el simulador?", st["h1"]))
    story.append(LineaAccento(uw))
    story.append(Spacer(1, 0.3*cm))
    story.append(p(
        "El Simulador Inteligente de Ocupación Hospitalaria es una herramienta "
        "académica que reproduce el funcionamiento de un hospital durante un "
        "período de hasta 50 horas simuladas. Permite observar cómo varían la "
        "ocupación de camas, la cola de pacientes en espera y el estado general "
        "del hospital bajo tres condiciones de demanda distintas: normal, alta "
        "demanda y crisis.", st))
    story.append(Spacer(1, 0.2*cm))
    story.append(p(
        "El sistema incluye un motor de asignación inteligente (sistema experto) "
        "que decide en cada momento qué paciente ocupa qué cama según su "
        "prioridad clínica (P1 crítico -> P4 no urgente), y un modelo predictivo "
        "que anticipa la ocupación del hospital con una hora de antelación.",
        st))
    story.append(Spacer(1, 0.2*cm))
    story.append(CajaInfo(
        "Todos los pacientes son sintéticos (generados artificialmente). "
        "El simulador no procesa datos reales de pacientes en ningún momento.",
        tipo="info"))
    story.append(Spacer(1, 0.3*cm))

    # Diagrama de flujo de páginas
    fig_flujo = fig_flujo_paginas()
    story.append(Paragraph("Flujo de uso de la aplicación:", st["h3"]))
    story.append(Image(fig_flujo, width=uw, height=uw * 0.26))
    story.append(Paragraph(
        "Figura 1. Las tres pantallas del simulador se recorren en orden: "
        "Configurar -> Simular -> Resultados.",
        st["caption"]))
    story.append(PageBreak())

    # ── SECCIÓN 2 — REQUISITOS E INSTALACIÓN ─────────────────────────────────
    story.append(Paragraph("2. Requisitos e Instalación", st["h1"]))
    story.append(LineaAccento(uw))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("2.1 Requisitos del sistema", st["h2"]))
    req_data = [
        ["Componente", "Versión mínima", "Notas"],
        ["Python", "3.10 o superior", "Disponible en python.org"],
        ["Sistema operativo", "Windows 10 / macOS 12 / Linux",
         "Cualquier SO con Python instalado"],
        ["Memoria RAM", "2 GB disponibles", "Recomendado 4 GB"],
        ["Espacio en disco", "50 MB libres", "Para los archivos del simulador"],
        ["Navegador web", "Chrome, Firefox o Edge (reciente)",
         "Streamlit abre la app automáticamente"],
    ]
    story.append(tabla(req_data, [uw*0.22, uw*0.28, uw*0.50], st))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("2.2 Archivos necesarios", st["h2"]))
    story.append(p(
        "Descarga o copia los siguientes archivos en una misma carpeta "
        "(por ejemplo, <b>simulador_f6/</b>):", st))
    story.append(Spacer(1, 0.15*cm))
    arch_data = [
        ["Archivo", "Descripción"],
        ["app.py", "Aplicación principal — punto de entrada del simulador"],
        ["motor_simulacion.py", "Motor de simulación tick a tick"],
        ["sistema_experto.py", "Sistema experto de asignación de camas"],
        ["generador_pacientes.py", "Generador de pacientes sintéticos"],
        ["modelo_final_f4b.pkl", "Modelo predictivo serializado (Regresión Lineal)"],
    ]
    story.append(tabla(arch_data, [uw*0.38, uw*0.62], st))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("2.3 Instalación de dependencias", st["h2"]))
    story.append(p(
        "Abre una terminal (símbolo del sistema en Windows, Terminal en macOS/Linux), "
        "navega a la carpeta del simulador y ejecuta el siguiente comando:", st))
    story.append(Spacer(1, 0.15*cm))

    cmd_table = Table(
        [[Paragraph(
            "pip install streamlit==1.45.1 numpy pandas matplotlib "
            "scikit-learn joblib reportlab",
            ParagraphStyle("cmd", fontName="Courier", fontSize=8.5,
                           textColor=ACCENT2,
                           backColor=DARK_MID, leading=12))]],
        colWidths=[uw],
    )
    cmd_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), DARK_MID),
        ("TOPPADDING", (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING", (0,0), (-1,-1), 10),
        ("ROUNDEDCORNERS", [4]),
    ]))
    story.append(cmd_table)
    story.append(Spacer(1, 0.15*cm))
    story.append(CajaInfo(
        "Si ya tienes instaladas versiones anteriores de estas librerías "
        "desde fases previas del proyecto, puedes omitir este paso — "
        "el simulador es compatible con el stack aprobado en F1–F5.",
        tipo="tip"))
    story.append(PageBreak())

    # ── SECCIÓN 3 — INICIO RÁPIDO ─────────────────────────────────────────────
    story.append(Paragraph("3. Inicio Rápido — Primera Simulación en 3 Pasos",
                            st["h1"]))
    story.append(LineaAccento(uw))
    story.append(Spacer(1, 0.3*cm))
    story.append(CajaInfo(
        "Si es tu primera vez, sigue estos 3 pasos para ver el simulador "
        "funcionando en menos de 2 minutos.",
        tipo="info"))
    story.append(Spacer(1, 0.3*cm))

    pasos_rapidos = [
        ("1", "Abre la terminal y lanza la aplicación",
         'Navega a la carpeta del simulador y ejecuta:  '
         'streamlit run app.py\n'
         'El navegador se abrirá automáticamente en http://localhost:8501'),
        ("2", "Pulsa \"[ Iniciar simulacion ]\" con la configuración por defecto",
         "No es necesario cambiar nada en la primera ejecución. "
         "La configuración por defecto (escenario Normal, modo automático, "
         "velocidad 3x, semilla 99) es válida para explorar el simulador."),
        ("3", "Observa la simulación y espera a que termine",
         "Verás las camas del hospital llenarse en tiempo real, "
         "la cola de espera actualizarse y las gráficas crecer tick a tick. "
         "Al completarse los 200 ticks (50 horas simuladas) aparecerá "
         "el resumen de resultados con las métricas del sistema."),
    ]
    for num, tit, desc in pasos_rapidos:
        story.append(paso_numerado(num, tit, st))
        story.append(Spacer(1, 0.1*cm))
        # Separar descripción de comando si aplica
        if '\n' in desc:
            partes = desc.split('\n')
            story.append(p(partes[0], st))
            cmd = Table(
                [[Paragraph(partes[1], ParagraphStyle("cmd2",
                    fontName="Courier", fontSize=9, textColor=ACCENT2,
                    backColor=DARK_MID, leading=13))]],
                colWidths=[uw],
            )
            cmd.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,-1), DARK_MID),
                ("TOPPADDING",    (0,0), (-1,-1), 6),
                ("BOTTOMPADDING", (0,0), (-1,-1), 6),
                ("LEFTPADDING",   (0,0), (-1,-1), 10),
            ]))
            story.append(cmd)
        else:
            story.append(p(desc, st))
        story.append(Spacer(1, 0.25*cm))

    story.append(PageBreak())

    # ── SECCIÓN 4 — PANTALLA 1: CONFIGURAR ───────────────────────────────────
    story.append(Paragraph("4. Pantalla 1 — Configurar el Hospital", st["h1"]))
    story.append(LineaAccento(uw))
    story.append(Spacer(1, 0.3*cm))
    story.append(p(
        "La primera pantalla que aparece al abrir el simulador es la de "
        "configuración. Aquí defines la infraestructura del hospital y los "
        "parámetros de la simulación antes de comenzar.", st))
    story.append(Spacer(1, 0.25*cm))

    story.append(Paragraph("4.1 Infraestructura del hospital (columna izquierda)",
                            st["h2"]))
    infra_data = [
        ["Campo", "Descripción", "Valor por defecto", "Rango"],
        ["Camas UCI",
         "Unidad de Cuidados Intensivos. Solo acepta pacientes críticos (P1).",
         "10", "2 – 30"],
        ["Camas Urgencias",
         "Acepta pacientes P1, P2 y P3. Primera línea de atención.",
         "20", "5 – 60"],
        ["Camas Sala de espera",
         "Área de desborde para P3 y P4 cuando las demás están llenas.",
         "10", "2 – 30"],
        ["Camas Hospitalización",
         "Área principal para P2 y P3. Mayor capacidad del hospital.",
         "40", "10 – 100"],
        ["Camas Observación",
         "Área secundaria para P2, P3 y P4. Se usa como zona de traslado.",
         "15", "5 – 40"],
    ]
    story.append(tabla(infra_data,
                       [uw*0.22, uw*0.44, uw*0.16, uw*0.18], st))
    story.append(Spacer(1, 0.2*cm))
    story.append(CajaInfo(
        "La vista previa a la derecha de los controles muestra en tiempo real "
        "cuántas camas tiene cada área con cuadrados de color verde. "
        "Úsala para verificar la configuración antes de iniciar.",
        tipo="tip"))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("4.2 Parámetros de simulación (columna derecha)",
                            st["h2"]))
    params_data = [
        ["Parámetro", "Descripción", "Opciones / Rango"],
        ["Escenario de demanda",
         "Define la presión de llegada de pacientes al hospital.",
         "Normal / Alta demanda / Crisis"],
        ["Modo asistido",
         "Activa o desactiva la intervención manual en traslados. "
         "Ver Sección 7 para detalles.",
         "Activado / Desactivado"],
        ["Velocidad de simulación",
         "Controla qué tan rápido avanza el reloj de la simulación "
         "en pantalla. No afecta los resultados.",
         "1x (lento) – 10x (rápido)"],
        ["Semilla aleatoria",
         "Número que garantiza reproducibilidad. Con la misma semilla "
         "y configuración, la simulación siempre produce los mismos resultados.",
         "0 – 9999 (por defecto: 99)"],
    ]
    story.append(tabla(params_data, [uw*0.22, uw*0.50, uw*0.28], st))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("4.3 Los tres escenarios de demanda", st["h2"]))
    esc_data = [
        ["Escenario", "Llegadas", "Ocupación esperada",
         "Cuándo usarlo"],
        ["Normal",
         "~6 pac/hora",
         "35%–55%",
         "Condiciones habituales de un hospital con carga baja. "
         "El sistema experto gestiona todo sin alertas."],
        ["Alta demanda",
         "~12 pac/hora",
         "60%–88%",
         "Situación de presión moderada-alta. El sistema activa "
         "traslados internos y genera alertas RES-04."],
        ["Crisis",
         "~20 pac/hora",
         "80%–100%",
         "Colapso hospitalario. Cola de espera activa, camas "
         "temporales, múltiples alertas. Recomendado con modo asistido."],
    ]
    story.append(tabla(esc_data,
                       [uw*0.16, uw*0.14, uw*0.20, uw*0.50], st))
    story.append(PageBreak())

    # ── SECCIÓN 5 — PANTALLA 2: SIMULADOR EN CURSO ───────────────────────────
    story.append(Paragraph("5. Pantalla 2 — Simulador en Curso", st["h1"]))
    story.append(LineaAccento(uw))
    story.append(Spacer(1, 0.3*cm))
    story.append(p(
        "Una vez pulsado «[ Iniciar simulacion ]», la aplicación pasa a la "
        "pantalla de simulación en curso. Esta pantalla se actualiza "
        "automáticamente en cada tick (cada 15 minutos simulados).", st))
    story.append(Spacer(1, 0.25*cm))

    # Grid ilustrativo
    fig_grid = fig_grid_hospital()
    story.append(Paragraph("5.1 Distribución de camas en tiempo real", st["h2"]))
    story.append(Image(fig_grid, width=uw, height=uw * 0.33))
    story.append(Paragraph(
        "Figura 2. Cada cuadrado representa una cama. El color indica "
        "su estado: verde=libre, rojo=ocupada, naranja=en limpieza, "
        "violeta=temporal. Haz clic en una cama roja para ver la "
        "ficha del paciente.",
        st["caption"]))
    story.append(Spacer(1, 0.2*cm))

    camas_data = [
        ["Color", "Estado", "Significado"],
        [Paragraph("-", ParagraphStyle("v", fontName="Helvetica-Bold",
            fontSize=14, textColor=GREEN_OK, alignment=TA_CENTER)),
         "Libre", "La cama está disponible para asignar a un nuevo paciente."],
        [Paragraph("-", ParagraphStyle("v", fontName="Helvetica-Bold",
            fontSize=14, textColor=RED_FAIL, alignment=TA_CENTER)),
         "Ocupada", "Hay un paciente hospitalizado. Haz clic para ver su ficha."],
        [Paragraph("-", ParagraphStyle("v", fontName="Helvetica-Bold",
            fontSize=14, textColor=ORANGE, alignment=TA_CENTER)),
         "En limpieza",
         "El paciente fue dado de alta y la cama está siendo desinfectada "
         "antes de poder asignarse de nuevo."],
        [Paragraph("-", ParagraphStyle("v", fontName="Helvetica-Bold",
            fontSize=14, textColor=ACCENT, alignment=TA_CENTER)),
         "Temporal",
         "Cama extra creada por el sistema cuando el área está llena. "
         "Se destruye automáticamente cuando el paciente es dado de alta."],
    ]
    story.append(tabla(camas_data, [uw*0.10, uw*0.18, uw*0.72], st))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("5.2 Ficha del paciente (clic en cama ocupada)",
                            st["h2"]))
    story.append(p(
        "Al hacer clic en cualquier cama roja o violeta, aparece un panel "
        "lateral dentro del grid con la información del paciente:", st))
    story.append(Spacer(1, 0.1*cm))
    ficha_data = [
        ["Campo", "Descripción"],
        ["ID", "Identificador único del paciente (primeros 8 caracteres)."],
        ["Prioridad",
         "P1 (crítico) · P2 (urgente) · P3 (menos urgente) · P4 (no urgente). "
         "El color del badge indica la gravedad."],
        ["Área requerida",
         "El área del hospital donde el paciente debería estar según su diagnóstico."],
        ["Cama",
         "Nombre del área y código de la cama actualmente asignada."],
        ["Estancia",
         "Horas que lleva el paciente hospitalizado en esta cama."],
        ["Espera acumulada",
         "Horas que el paciente esperó en cola antes de ser asignado."],
    ]
    story.append(tabla(ficha_data, [uw*0.22, uw*0.78], st))
    story.append(Spacer(1, 0.2*cm))
    story.append(CajaInfo(
        "Haz clic en «[ cerrar ]» o en cualquier cama libre para ocultar el panel.",
        tipo="tip"))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("5.3 Métricas en tiempo real", st["h2"]))
    story.append(p(
        "En la parte superior del grid se muestran cuatro métricas que se "
        "actualizan en cada tick:", st))
    story.append(Spacer(1, 0.1*cm))
    met_data = [
        ["Métrica", "Significado", "Valor normal"],
        ["Ocupación O (%)",
         "Porcentaje de camas ocupadas sobre el total de camas válidas "
         "(excluye las que están en limpieza).",
         "35%–55%"],
        ["Indicador I",
         "Índice compuesto de estado del hospital (0–100). Combina ocupación, "
         "tiempos de espera, pacientes en desborde y críticos sin cama.",
         "0–25 (Bajo)"],
        ["Nivel",
         "Clasificación cualitativa: Bajo · Medio · Alto · Crítico.",
         "Bajo"],
        ["En cola",
         "Número de pacientes esperando una cama en este momento.",
         "0"],
    ]
    story.append(tabla(met_data, [uw*0.22, uw*0.56, uw*0.22], st))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("5.4 Predicción de ocupación (T+4)", st["h2"]))
    story.append(p(
        "Debajo de las métricas principales aparece la predicción del modelo: "
        "«Predicción O (T+4 · 1h)». Este valor indica cuál será la ocupación "
        "estimada dentro de 1 hora (4 ticks de 15 min). "
        "La flecha verde indica que se espera que la ocupación baje; "
        "la roja que suba. Esta predicción está disponible a partir del "
        "tick 5 (cuando el modelo tiene suficientes datos históricos).", st))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("5.5 Controles durante la simulación", st["h2"]))
    ctrl_data = [
        ["Control", "Ubicación", "Función"],
        ["[ Pausar ] / [ Reanudar ]",
         "Cabecera superior derecha",
         "Detiene o reanuda el avance tick a tick. El estado del hospital "
         "se preserva exactamente."],
        ["[ Config. ]",
         "Cabecera superior derecha",
         "Vuelve a la pantalla de configuración e inicia una nueva "
         "simulación. La simulación actual se descarta."],
        ["Velocidad (slider en config.)",
         "Pantalla de configuración",
         "1x = ~0.3 s/tick · 10x = ~0.03 s/tick. "
         "Ajusta antes de iniciar la simulación."],
    ]
    story.append(tabla(ctrl_data, [uw*0.28, uw*0.26, uw*0.46], st))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("5.6 Gráficas de evolución temporal", st["h2"]))
    story.append(p(
        "A la derecha del grid aparecen dos gráficas actualizadas en tiempo real:", st))
    story.append(Spacer(1, 0.1*cm))
    for bul in [
        "O(t) real vs Predicción T+4: muestra cómo ha variado la ocupación "
        "en los últimos 60 ticks y la predicción del modelo (línea discontinua cian). "
        "La línea vertical punteada marca el fin del período de calentamiento (tick 20).",
        "Indicador Compuesto I(t): muestra la evolución del indicador I. "
        "Las bandas de color representan los cuatro niveles (verde=Bajo, "
        "naranja=Medio, rojo oscuro=Alto, rojo=Crítico).",
    ]:
        story.append(p(f"• {bul}", st, "body_bullet"))
    story.append(Spacer(1, 0.15*cm))

    story.append(Paragraph("5.7 Panel de último tick", st["h2"]))
    story.append(p(
        "Debajo de las gráficas aparece un panel con los eventos del tick "
        "más reciente: llegadas nuevas , altas , traslados  y alertas . "
        "Úsalo para entender qué está ocurriendo en el hospital en cada momento.",
        st))
    story.append(PageBreak())

    # ── SECCIÓN 6 — PANTALLA 3: RESULTADOS ───────────────────────────────────
    story.append(Paragraph("6. Pantalla 3 — Resultados Finales", st["h1"]))
    story.append(LineaAccento(uw))
    story.append(Spacer(1, 0.3*cm))
    story.append(p(
        "Cuando la simulación llega al tick 200 (50 horas simuladas), "
        "aparece automáticamente el mensaje «[OK] Simulación completada» y "
        "se despliega el resumen de resultados al final de la página. "
        "También puedes navegar a la pantalla completa de resultados "
        "con el botón «[ Ver resultados completos ]».", st))
    story.append(Spacer(1, 0.25*cm))

    story.append(Paragraph("6.1 Métricas del resumen", st["h2"]))
    res_data = [
        ["Métrica", "Descripción"],
        ["O media",
         "Ocupación promedio durante el régimen estable (ticks 20–200). "
         "Excluye el período de calentamiento inicial."],
        ["O P5–P95",
         "Rango del percentil 5 al 95 de la ocupación. Indica la variabilidad."],
        ["I media",
         "Valor promedio del indicador compuesto I en régimen estable."],
        ["Nivel modal",
         "El nivel del indicador I más frecuente durante la simulación "
         "(Bajo, Medio, Alto o Crítico)."],
        ["Traslados totales",
         "Número total de traslados internos y desbordes realizados "
         "por el sistema experto."],
        ["Alertas RES-04",
         "Número de ticks en que algún paciente P1 o P2 llevaba más de "
         "30 minutos (2 ticks) en cola sin cama asignada."],
        ["RMSE predicción",
         "Error cuadrático medio del modelo predictivo en este escenario. "
         "Valores por debajo de 5 pp se consideran clínicamente aceptables."],
        ["Altas totales",
         "Número total de altas médicas producidas durante la simulación."],
    ]
    story.append(tabla(res_data, [uw*0.22, uw*0.78], st))
    story.append(Spacer(1, 0.25*cm))

    story.append(Paragraph("6.2 Verificación CE-B", st["h2"]))
    story.append(p(
        "Al final del resumen aparecen dos indicadores de validación "
        "automática (criterios CE-B de la Fase 5):", st))
    story.append(Spacer(1, 0.1*cm))
    for bul in [
        "[OK] CE-B rango O: la ocupación media está dentro del rango esperado "
        "para el escenario seleccionado.",
        "[OK] CE-B nivel I: el nivel modal del indicador I corresponde al "
        "nivel esperado para el escenario.",
    ]:
        story.append(p(f"• {bul}", st, "body_bullet"))
    story.append(Spacer(1, 0.15*cm))
    story.append(CajaInfo(
        "Si aparece [FALLO] en lugar de [OK], no indica un error del sistema — "
        "puede ocurrir en configuraciones de hospital muy distintas a la "
        "referencia (D008) o con semillas que generan variabilidad extrema.",
        tipo="warn"))
    story.append(PageBreak())

    # ── SECCIÓN 7 — MODO ASISTIDO ─────────────────────────────────────────────
    story.append(Paragraph("7. Modo Asistido — Guía Paso a Paso", st["h1"]))
    story.append(LineaAccento(uw))
    story.append(Spacer(1, 0.3*cm))
    story.append(p(
        "En el modo asistido, el sistema experto propone las acciones de "
        "traslado y desborde pero no las ejecuta automáticamente: "
        "eres tú quien decide cuáles aprobar.", st))
    story.append(Spacer(1, 0.2*cm))
    story.append(CajaInfo(
        "El modo asistido es especialmente útil con el escenario «Crisis» "
        "para analizar qué redistribuciones propone el sistema y decidir "
        "cuáles tienen sentido clínico.",
        tipo="info"))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Flujo de uso del modo asistido:", st["h2"]))
    asistido_pasos = [
        ("1", "Activa el modo asistido en la pantalla de configuración",
         "Desliza el interruptor «Modo asistido» a la posición activada "
         "antes de pulsar «[ Iniciar simulacion ]»."),
        ("2", "La simulación avanza normalmente hasta encontrar una acción propuesta",
         "Cuando el sistema experto detecta que hay pacientes que deberían "
         "trasladarse o que se necesita crear una cama temporal, "
         "la simulación se pausa automáticamente."),
        ("3", "Revisa las acciones propuestas en el panel violeta",
         "Aparece un panel con el título «Modo asistido — acciones propuestas» "
         "listando cada traslado o desborde recomendado. "
         "Cada acción tiene una casilla de verificación activada por defecto."),
        ("4", "Selecciona cuáles acciones ejecutar",
         "Desmarca las acciones que no deseas realizar. "
         "Las acciones desmarcadas se descartan: los pacientes afectados "
         "permanecen en su estado actual."),
        ("5", "Pulsa «[ Ejecutar ]» o «[ Omitir todas ]»",
         "«Ejecutar»: realiza las acciones marcadas y reanuda la simulación. "
         "«Omitir todas»: descarta todas las propuestas y reanuda la "
         "simulación sin realizar ningún traslado."),
    ]
    for num, tit, desc in asistido_pasos:
        story.append(paso_numerado(num, tit, st))
        story.append(Spacer(1, 0.08*cm))
        story.append(p(desc, st))
        story.append(Spacer(1, 0.2*cm))

    story.append(CajaInfo(
        "Diferencia clave entre modos: en modo automático el sistema toma "
        "todas las decisiones solo y la simulación no se interrumpe. "
        "En modo asistido, cada propuesta de traslado o desborde requiere "
        "tu aprobación antes de ejecutarse.",
        tipo="info"))
    story.append(PageBreak())

    # ── SECCIÓN 8 — INDICADOR I ───────────────────────────────────────────────
    story.append(Paragraph("8. Interpretación del Indicador I", st["h1"]))
    story.append(LineaAccento(uw))
    story.append(Spacer(1, 0.3*cm))
    story.append(p(
        "El indicador I es el índice compuesto que resume el estado global "
        "del hospital en un único número entre 0 y 100. "
        "Se calcula con la siguiente fórmula:", st))
    story.append(Spacer(1, 0.15*cm))

    formula = Table(
        [[Paragraph(
            "I = 0.4 × O + 0.2 × E + 0.2 × P + 0.2 × C",
            ParagraphStyle("formula", fontName="Courier-Bold", fontSize=11,
                           textColor=ACCENT, backColor=DARK_MID,
                           alignment=TA_CENTER, leading=16))]],
        colWidths=[uw],
    )
    formula.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), DARK_MID),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
    ]))
    story.append(formula)
    story.append(Spacer(1, 0.2*cm))

    comp_data = [
        ["Componente", "Peso", "Descripción"],
        ["O — Ocupación",
         "40%",
         "Porcentaje de camas ocupadas. Es el factor más importante del índice."],
        ["E — Espera normalizada",
         "20%",
         "Tiempo promedio de espera de los pacientes en cola, "
         "normalizado al máximo de referencia (4 horas)."],
        ["P — Desborde",
         "20%",
         "Proporción de pacientes en camas temporales (pasillos). "
         "Indica si el hospital está usando capacidad extra."],
        ["C — Críticos sin cama",
         "20%",
         "Proporción de pacientes P1 o P2 que llevan más de 30 minutos "
         "sin cama asignada. El indicador más grave."],
    ]
    story.append(tabla(comp_data, [uw*0.24, uw*0.10, uw*0.66], st))
    story.append(Spacer(1, 0.3*cm))

    # Figura del indicador
    fig_ind = fig_indicador_I()
    story.append(Image(fig_ind, width=uw, height=uw * 0.21))
    story.append(Paragraph(
        "Figura 3. Los cuatro niveles del indicador I con sus rangos numéricos.",
        st["caption"]))
    story.append(Spacer(1, 0.2*cm))

    niveles_data = [
        ["Nivel", "Rango I", "Color", "Interpretación y acciones recomendadas"],
        ["BAJO",
         "0 – 25",
         Paragraph("-", ParagraphStyle("v", fontName="Helvetica-Bold",
             fontSize=14, textColor=GREEN_OK, alignment=TA_CENTER)),
         "El hospital opera con normalidad. No se requiere intervención. "
         "El sistema experto gestiona las asignaciones sin alertas."],
        ["MEDIO",
         "26 – 50",
         Paragraph("-", ParagraphStyle("v", fontName="Helvetica-Bold",
             fontSize=14, textColor=ORANGE, alignment=TA_CENTER)),
         "Presión moderada. El sistema empieza a activar traslados internos. "
         "En modo asistido, conviene revisar las propuestas."],
        ["ALTO",
         "51 – 75",
         Paragraph("-", ParagraphStyle("v", fontName="Helvetica-Bold",
             fontSize=14, textColor=colors.HexColor("#FF5722"),
             alignment=TA_CENTER)),
         "Presión alta. Traslados activos, posible cola de espera. "
         "Se recomienda revisar la distribución de camas."],
        ["CRÍTICO",
         "76 – 100\no RES-04",
         Paragraph("-", ParagraphStyle("v", fontName="Helvetica-Bold",
             fontSize=14, textColor=RED_FAIL, alignment=TA_CENTER)),
         "Colapso. Pacientes P1/P2 sin cama más de 30 min, camas temporales "
         "saturadas. En modo asistido, el sistema presenta opciones de "
         "redistribución urgente."],
    ]
    story.append(tabla(niveles_data, [uw*0.12, uw*0.12, uw*0.08, uw*0.68], st))
    story.append(Spacer(1, 0.15*cm))
    story.append(CajaInfo(
        "El nivel CRÍTICO puede activarse aunque I < 76 si algún paciente P1 "
        "o P2 lleva más de 2 ticks (30 minutos) en cola. Esto se indica en "
        "la métrica «Alertas RES-04» del resumen final.",
        tipo="warn"))
    story.append(PageBreak())

    # ── SECCIÓN 9 — REFERENCIA RÁPIDA ────────────────────────────────────────
    story.append(Paragraph("9. Referencia Rápida de Controles", st["h1"]))
    story.append(LineaAccento(uw))
    story.append(Spacer(1, 0.3*cm))

    ref_data = [
        ["Elemento", "Pantalla", "Acción"],
        ["[ Iniciar simulacion ]",
         "Configurar hospital",
         "Guarda la configuración e inicia la simulación con los "
         "parámetros actuales."],
        ["[ Pausar ] / [ Reanudar ]",
         "Simulador en curso",
         "Congela o reanuda la simulación. El estado del hospital se preserva."],
        ["[ Config. ]",
         "Simulador en curso",
         "Vuelve a la pantalla de configuración. "
         "La simulación actual se descarta."],
        ["Cama roja (clic)",
         "Grid de camas",
         "Abre la ficha del paciente hospitalizado en esa cama."],
        ["[ cerrar ] (ficha)",
         "Panel ficha paciente",
         "Cierra el panel de ficha del paciente."],
        ["[ Ejecutar ] (modo asistido)",
         "Panel asistido",
         "Confirma las acciones marcadas y reanuda la simulación."],
        ["[ Omitir todas ] (modo asistido)",
         "Panel asistido",
         "Descarta todas las propuestas y reanuda sin ejecutarlas."],
        ["[ Ver resultados completos ]",
         "Simulador en curso (al finalizar)",
         "Navega a la pantalla de resultados con el resumen completo."],
        ["[ Nueva simulacion ]",
         "Resultados",
         "Vuelve a la pantalla de configuración para iniciar otra simulación."],
    ]
    story.append(tabla(ref_data, [uw*0.28, uw*0.24, uw*0.48], st))
    story.append(PageBreak())

    # ── SECCIÓN 10 — PREGUNTAS FRECUENTES ────────────────────────────────────
    story.append(Paragraph("10. Preguntas Frecuentes", st["h1"]))
    story.append(LineaAccento(uw))
    story.append(Spacer(1, 0.3*cm))

    faqs = [
        ("¿Por qué la simulación «salta» al inicio sin mostrar mucho movimiento?",
         "Los primeros 20 ticks (5 horas) son el período de calentamiento: "
         "el hospital parte de cero camas ocupadas y la ocupación crece "
         "gradualmente. Es comportamiento normal — el régimen estable "
         "comienza en el tick 20."),
        ("¿Por qué a veces el nivel I aparece como CRÍTICO aunque la ocupación "
         "no sea muy alta?",
         "El nivel CRÍTICO puede activarse por la regla RES-04: si algún "
         "paciente P1 o P2 lleva más de 2 ticks (30 minutos) en cola sin "
         "cama asignada, el sistema fuerza el nivel a CRÍTICO "
         "independientemente del valor numérico de I."),
        ("¿Puedo cambiar la velocidad durante la simulación?",
         "La velocidad se configura en la pantalla inicial antes de iniciar. "
         "Para cambiarla, pulsa «[ Config. ]» para volver a la pantalla de "
         "configuración — esto descartará la simulación actual."),
        ("¿Los resultados son siempre iguales con la misma semilla?",
         "Sí. Con la misma semilla, el mismo escenario y la misma "
         "configuración de hospital, la simulación produce exactamente "
         "los mismos resultados. Esto permite reproducibilidad académica."),
        ("¿Qué ocurre si omito todas las acciones en modo asistido?",
         "Los pacientes propuestos para traslado permanecen en su área "
         "actual. Si el área sigue llena, en el siguiente tick el sistema "
         "volverá a proponer acciones. Omitir acciones repetidamente en "
         "crisis puede aumentar la cola y el nivel CRÍTICO."),
        ("¿Por qué hay camas de color violeta (temporales)?",
         "Cuando un área está completamente llena y llega un paciente que "
         "no puede esperar, el sistema crea una cama temporal (\"pasillo\"). "
         "Estas camas no cuentan para la capacidad oficial del hospital "
         "y se destruyen cuando el paciente es dado de alta."),
        ("¿Qué significa RMSE en los resultados?",
         "RMSE (Root Mean Square Error) es el error del modelo predictivo: "
         "indica en promedio cuántos puntos porcentuales se equivoca la "
         "predicción de ocupación a 1 hora. Un RMSE < 5 pp se considera "
         "clínicamente aceptable para este prototipo."),
        ("¿Puedo usar el simulador con datos reales de pacientes?",
         "No. El simulador está diseñado exclusivamente para datos sintéticos "
         "generados por el módulo generador_pacientes.py. No procesa ni "
         "importa datos reales de pacientes en ninguna fase."),
    ]
    for i, (pregunta, respuesta) in enumerate(faqs):
        story.append(KeepTogether([
            p(f"P{i+1}. {pregunta}", st, "h3"),
            p(respuesta, st),
            Spacer(1, 0.2*cm),
        ]))

    doc.build(story, onFirstPage=hf, onLaterPages=hf)
    print(f"Manual generado: {ruta_salida}")


def main():
    print("Generando Manual de Usuario F6...")
    ruta = os.path.join(BASE_DIR, "manual_usuario_f6.pdf")
    construir_manual(ruta)
    # Limpiar figuras temporales
    for tmp in ["_tmp_flujo.png", "_tmp_grid.png", "_tmp_indicador.png"]:
        ruta_tmp = os.path.join(BASE_DIR, tmp)
        if os.path.exists(ruta_tmp):
            os.remove(ruta_tmp)
    print("Listo.")

if __name__ == "__main__":
    main()
