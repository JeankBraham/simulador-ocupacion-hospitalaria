"""
app.py
======
Simulador Inteligente de Ocupación Hospitalaria
Fase 6 — Despliegue · Interfaz Web

Autor  : Juan Camilo García Braham
Curso  : IA en Salud · Maestría IA y CD · UTP
Año    : 2026
Stack  : Python 3.12 · streamlit 1.45.1 · numpy 2.4 · matplotlib 3.10

Ejecución:
    streamlit run app.py

Decisiones de diseño activas — F6
───────────────────────────────────
D_F6_001  Stack: streamlit 1.45.1 (ver motor_simulacion.py)
D_F6_002  Estado de simulación en st.session_state["sim"] (EstadoSimulacion)
D_F6_003  Grid HTML del hospital via st.components.v1.html()
D_F6_004  Paleta del proyecto: fondo #1a1a2e · acento #BB86FC (coherencia con F4/F5)
D_F6_005  Velocidad 1x–10x: implementada con time.sleep(delay/velocidad) en loop
          de autoavance. Pausa preserva el estado sin reiniciar.
D_F6_006  Modo asistido: cuando hay acciones_pendientes el autoavance se detiene
          y se muestra panel de confirmación antes de continuar.
"""

import time
import os
import sys

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib

# ── Rutas de módulos ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from motor_simulacion import (
    crear_estado,
    avanzar_tick,
    confirmar_acciones_pendientes,
    calcular_resumen,
    snapshot_hospital,
    COLOR_NIVEL,
    COLOR_CAMA,
    TICKS_TOTAL,
    WARM_UP_TICKS,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN STREAMLIT
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Simulador Hospitalario · UTP",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS GLOBAL — D_F6_004 · paleta coherente con F4-B y F5
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #1a1a2e;
    color: #e0e0e0;
    font-family: 'Liberation Sans', 'DejaVu Sans', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #16213e;
    border-right: 1px solid #2a2a4a;
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

/* ── Métricas ── */
[data-testid="stMetric"] {
    background: #16213e;
    border: 1px solid #2a2a4a;
    border-radius: 8px;
    padding: 10px 14px;
}
[data-testid="stMetricLabel"] { color: #aaaaaa !important; font-size: 0.75rem; }
[data-testid="stMetricValue"] { color: #BB86FC !important; font-size: 1.4rem; font-weight: 700; }
[data-testid="stMetricDelta"] { font-size: 0.7rem; }

/* ── Botones ── */
.stButton > button {
    background: #BB86FC;
    color: #1a1a2e;
    border: none;
    border-radius: 6px;
    font-weight: 700;
    padding: 6px 18px;
    transition: opacity 0.15s;
}
.stButton > button:hover { opacity: 0.85; }

/* ── Selectbox / slider ── */
.stSelectbox label, .stSlider label { color: #aaaaaa !important; font-size: 0.8rem; }
.stSlider [data-baseweb="slider"] div[role="slider"] { background: #BB86FC; }

/* ── Separadores ── */
hr { border-color: #2a2a4a; }

/* ── Expander ── */
.streamlit-expanderHeader { color: #BB86FC !important; font-weight: 600; }

/* ── Alertas ── */
.nivel-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 12px;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.04em;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DEL MODELO (cacheado)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def cargar_modelo():
    ruta = os.path.join(BASE_DIR, "modelo_final_f4b.pkl")
    return joblib.load(ruta)


# ─────────────────────────────────────────────────────────────────────────────
# INICIALIZACIÓN DE SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

def _init_session():
    if "sim" not in st.session_state:
        st.session_state["sim"] = None
    if "autoavance" not in st.session_state:
        st.session_state["autoavance"] = False
    if "ultimo_snap" not in st.session_state:
        st.session_state["ultimo_snap"] = None
    if "ultimo_resultado" not in st.session_state:
        st.session_state["ultimo_resultado"] = None


# ─────────────────────────────────────────────────────────────────────────────
# GRID HTML DEL HOSPITAL — D_F6_003
# ─────────────────────────────────────────────────────────────────────────────

def _html_grid_hospital(snap: dict) -> str:
    """Genera el HTML del grid de camas del hospital.

    Cada celda muestra el estado de la cama con color semántico.
    El tooltip (title) contiene datos del paciente si la cama está ocupada.
    """
    if not snap:
        return "<p style='color:#aaa'>Sin datos de hospital.</p>"

    # Leyenda
    leyenda_items = "".join(
        f"""<span style="
                display:inline-flex;align-items:center;gap:5px;
                margin-right:14px;font-size:0.72rem;color:#ccc;">
            <span style="
                width:12px;height:12px;border-radius:3px;
                background:{color};display:inline-block;">
            </span>{label}
        </span>"""
        for label, color in [
            ("Libre",       COLOR_CAMA["libre"]),
            ("Ocupada",     COLOR_CAMA["ocupada"]),
            ("En limpieza", COLOR_CAMA["en_limpieza"]),
            ("Temporal",    COLOR_CAMA["temporal"]),
        ]
    )

    secciones_html = ""
    for area in snap["areas"]:
        nombre  = area["nombre"]
        cap     = area["capacidad_total"]
        cap_dis = area["cap_disponible"]
        camas   = area["camas"]

        ocupadas = sum(1 for c in camas if c["estado"] == "ocupada")
        temporales = sum(1 for c in camas if c["estado"] == "temporal")

        celdas_html = ""
        for cama in camas:
            estado = cama["estado"]
            color  = COLOR_CAMA.get(estado, "#555")
            pac    = cama["paciente"]

            if pac:
                tooltip = (
                    f"ID: {pac['id']} | {pac['prioridad']} | "
                    f"Área req: {pac['area_req']} | "
                    f"Estancia: {pac['estancia_h']:.1f}h"
                )
            elif estado == "en_limpieza":
                tooltip = "En limpieza — no disponible"
            else:
                tooltip = "Libre"

            celdas_html += f"""
            <div title="{tooltip}" style="
                width:18px;height:18px;border-radius:3px;
                background:{color};cursor:default;
                transition:transform 0.1s;
                border: 1px solid rgba(255,255,255,0.08);"
                onmouseover="this.style.transform='scale(1.4)'"
                onmouseout="this.style.transform='scale(1)'">
            </div>"""

        estado_area_color = (
            "#F44336" if cap_dis <= 0
            else "#FF9800" if cap_dis <= cap * 0.2
            else "#4CAF50"
        )

        secciones_html += f"""
        <div style="margin-bottom:16px;">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
                <span style="font-size:0.8rem;font-weight:700;color:#BB86FC;">
                    {nombre}
                </span>
                <span style="
                    font-size:0.7rem;color:#aaa;
                    background:#2a2a4a;border-radius:4px;padding:1px 7px;">
                    {ocupadas + temporales}/{cap} ocupadas
                </span>
                <span style="
                    font-size:0.7rem;font-weight:600;
                    color:{estado_area_color};">
                    {"⚠ DESBORDE" if cap_dis <= 0 else f"Disp: {cap_dis}"}
                </span>
            </div>
            <div style="display:flex;flex-wrap:wrap;gap:4px;">
                {celdas_html}
            </div>
        </div>"""

    html = f"""
    <div style="
        background:#16213e;border-radius:10px;
        padding:16px 18px;border:1px solid #2a2a4a;">
        <div style="margin-bottom:12px;display:flex;flex-wrap:wrap;gap:4px;">
            {leyenda_items}
        </div>
        {secciones_html}
    </div>"""
    return html


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA MATPLOTLIB — gráfico de serie temporal (O, I, predicción)
# ─────────────────────────────────────────────────────────────────────────────

def _fig_tiempo_real(historial, escenario: str):
    """Genera la figura de O(t), I(t) y predicción, estilo F5."""
    if not historial or len(historial) < 2:
        return None

    ticks   = [r.tick for r in historial]
    O_vals  = [r.O_t for r in historial]
    I_vals  = [r.I_t for r in historial]
    pred_x  = [r.tick for r in historial if r.pred_O_t4 is not None]
    pred_y  = [r.pred_O_t4 for r in historial if r.pred_O_t4 is not None]

    fig = plt.figure(figsize=(10, 3.6), facecolor="#1a1a2e")
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    for ax in (ax1, ax2):
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#aaaaaa", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    # ── Panel izquierdo: O(t) real vs predicción ──
    ax1.plot(ticks, O_vals, color="#BB86FC", lw=1.4, label="O(t) real")
    if pred_x:
        pred_x_shifted = [x + 4 for x in pred_x]
        ax1.plot(pred_x_shifted, pred_y, color="#03DAC6",
                 lw=1.0, ls="--", label="O pred. T+4", alpha=0.85)

    if len(ticks) > WARM_UP_TICKS:
        ax1.axvline(WARM_UP_TICKS, color="#555", ls=":", lw=0.8)
        ax1.text(WARM_UP_TICKS + 1, 2, "warm-up",
                 color="#666", fontsize=6, va="bottom")

    ax1.set_xlabel("Tick (1 tick = 15 min)", color="#aaa", fontsize=7)
    ax1.set_ylabel("Ocupación (%)", color="#aaa", fontsize=7)
    ax1.set_title("Ocupación O(t) vs Predicción", color="#e0e0e0", fontsize=8, pad=4)
    ax1.set_ylim(0, 105)
    ax1.legend(fontsize=6, facecolor="#1a1a2e", labelcolor="#ccc",
               loc="upper left", framealpha=0.7)

    # ── Panel derecho: I(t) con bandas de nivel ──
    bandas = [
        (0,  25,  "#4CAF50", "Bajo"),
        (25, 50,  "#FF9800", "Medio"),
        (50, 75,  "#FF5722", "Alto"),
        (75, 100, "#F44336", "Crítico"),
    ]
    for y0, y1, color, _ in bandas:
        ax2.axhspan(y0, y1, alpha=0.08, color=color)
        ax2.axhline(y1, color=color, lw=0.4, ls="--", alpha=0.4)

    ax2.plot(ticks, I_vals, color="#BB86FC", lw=1.4)
    if I_vals:
        ax2.axhline(I_vals[-1], color="#03DAC6", lw=0.7, ls=":", alpha=0.6)

    ax2.set_xlabel("Tick (1 tick = 15 min)", color="#aaa", fontsize=7)
    ax2.set_ylabel("Indicador I", color="#aaa", fontsize=7)
    ax2.set_title("Indicador Compuesto I(t)", color="#e0e0e0", fontsize=8, pad=4)
    ax2.set_ylim(0, 100)

    fig.tight_layout(pad=1.2)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Panel de control
# ─────────────────────────────────────────────────────────────────────────────

def _renderizar_sidebar(modelo):
    with st.sidebar:
        st.markdown(
            "<h2 style='color:#BB86FC;margin-bottom:0;font-size:1.1rem;'>"
            "🏥 Simulador Hospitalario</h2>"
            "<p style='color:#666;font-size:0.7rem;margin-top:2px;'>"
            "UTP · IA en Salud · 2026</p>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        st.markdown(
            "<p style='color:#aaa;font-size:0.75rem;font-weight:600;"
            "letter-spacing:0.08em;'>CONFIGURACIÓN</p>",
            unsafe_allow_html=True,
        )

        escenario = st.selectbox(
            "Escenario",
            options=["normal", "alta_demanda", "crisis"],
            format_func=lambda x: {
                "normal":       "Normal (λ=1.5 pac/tick)",
                "alta_demanda": "Alta demanda (λ=3.0 pac/tick)",
                "crisis":       "Crisis (λ=5.0 pac/tick)",
            }[x],
            key="cfg_escenario",
        )
        modo_asistido = st.toggle(
            "Modo asistido",
            value=False,
            key="cfg_modo_asistido",
            help=(
                "Activado: el gestor confirma cada acción de traslado/desborde. "
                "Desactivado: el sistema ejecuta todo automáticamente."
            ),
        )
        velocidad = st.slider(
            "Velocidad de simulación",
            min_value=1, max_value=10, value=3, step=1,
            key="cfg_velocidad",
            format="%dx",
        )
        seed = st.number_input(
            "Semilla (reproducibilidad)",
            min_value=0, max_value=9999, value=99, step=1,
            key="cfg_seed",
        )

        st.markdown("---")
        st.markdown(
            "<p style='color:#aaa;font-size:0.75rem;font-weight:600;"
            "letter-spacing:0.08em;'>CONTROL</p>",
            unsafe_allow_html=True,
        )

        sim = st.session_state["sim"]
        col_a, col_b = st.columns(2)

        with col_a:
            if st.button("▶ Iniciar", use_container_width=True):
                st.session_state["sim"] = crear_estado(
                    escenario=escenario,
                    modo_asistido=modo_asistido,
                    seed=int(seed),
                )
                st.session_state["autoavance"] = True
                st.session_state["ultimo_snap"] = None
                st.session_state["ultimo_resultado"] = None
                st.rerun()

        with col_b:
            sim = st.session_state["sim"]
            if sim and sim.activa and not sim.finalizada:
                label  = "⏸ Pausar" if not sim.pausada else "▶ Reanudar"
                if st.button(label, use_container_width=True):
                    sim.pausada = not sim.pausada
                    st.session_state["autoavance"] = not sim.pausada
                    st.rerun()

        if sim and (sim.finalizada or not sim.activa):
            if st.button("⟳ Reiniciar", use_container_width=True):
                st.session_state["sim"] = None
                st.session_state["autoavance"] = False
                st.rerun()

        st.markdown("---")

        # Progreso
        if sim:
            progreso = sim.tick_actual / sim.ticks_total
            st.markdown(
                f"<p style='font-size:0.72rem;color:#aaa;margin-bottom:3px;'>"
                f"Tick {sim.tick_actual} / {sim.ticks_total} "
                f"({sim.tick_actual * 0.25:.1f} h simuladas)</p>",
                unsafe_allow_html=True,
            )
            st.progress(progreso)

            if sim.finalizada:
                st.success("✅ Simulación completada")
        else:
            st.markdown(
                "<p style='color:#555;font-size:0.78rem;'>"
                "Configura los parámetros y pulsa Iniciar.</p>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Leyenda de niveles
        st.markdown(
            "<p style='color:#aaa;font-size:0.72rem;font-weight:600;"
            "letter-spacing:0.08em;margin-bottom:6px;'>NIVELES DEL INDICADOR I</p>",
            unsafe_allow_html=True,
        )
        for nivel, color in COLOR_NIVEL.items():
            rangos = {"Bajo": "0–25", "Medio": "26–50",
                      "Alto": "51–75", "Crítico": "76–100"}
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;"
                f"margin-bottom:4px;'>"
                f"<span style='width:10px;height:10px;border-radius:50%;"
                f"background:{color};display:inline-block;'></span>"
                f"<span style='font-size:0.73rem;color:#ccc;'>"
                f"<b>{nivel}</b> ({rangos[nivel]})</span></div>",
                unsafe_allow_html=True,
            )

    return escenario, modo_asistido, velocidad, int(seed)


# ─────────────────────────────────────────────────────────────────────────────
# PANEL DE MÉTRICAS EN TIEMPO REAL
# ─────────────────────────────────────────────────────────────────────────────

def _renderizar_metricas(snap: dict, historial):
    if not snap:
        return

    ind = snap["indicador"]
    nivel    = ind.get("nivel", "Bajo")
    color_nv = COLOR_NIVEL.get(nivel, "#aaa")

    # Badge de nivel
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:6px;'>"
        f"<span style='font-size:0.75rem;color:#aaa;'>Nivel actual:</span>"
        f"<span class='nivel-badge' style='background:{color_nv};color:#1a1a2e;'>"
        f"{nivel}</span></div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Ocupación O", f"{ind['O']:.1f}%")
    with col2:
        st.metric("Indicador I", f"{ind['I']:.1f}")
    with col3:
        st.metric("Espera norm. E", f"{ind['E']:.1f}")
    with col4:
        st.metric("Desborde P", f"{ind['P']:.1f}")
    with col5:
        st.metric("Críticos sin cama C", f"{ind['C']:.1f}")

    # Predicción T+4
    if historial:
        ultimos_con_pred = [r for r in historial if r.pred_O_t4 is not None]
        if ultimos_con_pred:
            pred = ultimos_con_pred[-1].pred_O_t4
            delta = pred - ind["O"]
            st.metric(
                "Predicción O (T+4 · 1h)",
                f"{pred:.1f}%",
                delta=f"{delta:+.1f} pp",
                delta_color="inverse",
            )


# ─────────────────────────────────────────────────────────────────────────────
# PANEL DE COLA DE ESPERA
# ─────────────────────────────────────────────────────────────────────────────

def _renderizar_cola(snap: dict):
    cola = snap.get("cola_espera", [])
    if not cola:
        st.markdown(
            "<p style='color:#4CAF50;font-size:0.8rem;'>✓ Sin pacientes en espera</p>",
            unsafe_allow_html=True,
        )
        return

    colores_prioridad = {
        "P1": "#F44336", "P2": "#FF9800",
        "P3": "#4CAF50", "P4": "#2196F3",
    }

    filas = ""
    for p in cola[:15]:   # máximo 15 filas visibles
        color = colores_prioridad.get(p["prioridad"], "#aaa")
        alerta = " ⚠" if p["ticks_espera"] >= 2 and p["prioridad"] in ("P1", "P2") else ""
        filas += f"""
        <tr style="border-bottom:1px solid #2a2a4a;">
            <td style="padding:4px 8px;">
                <span style="
                    background:{color};color:#1a1a2e;
                    border-radius:4px;padding:1px 6px;
                    font-size:0.72rem;font-weight:700;">{p['prioridad']}</span>
            </td>
            <td style="padding:4px 8px;font-size:0.72rem;color:#ccc;font-family:monospace;">
                {p['id']}
            </td>
            <td style="padding:4px 8px;font-size:0.72rem;color:#ccc;">{p['area_req']}</td>
            <td style="padding:4px 8px;font-size:0.72rem;color:#ccc;">{p['espera_h']:.2f} h</td>
            <td style="padding:4px 8px;font-size:0.72rem;color:#FF5722;">{alerta}</td>
        </tr>"""

    resto = f"<p style='color:#666;font-size:0.7rem;margin-top:4px;'>... y {len(cola)-15} más</p>" if len(cola) > 15 else ""

    st.markdown(f"""
    <div style="background:#16213e;border-radius:8px;padding:10px;border:1px solid #2a2a4a;">
        <p style="color:#BB86FC;font-size:0.78rem;font-weight:700;margin-bottom:8px;">
            Cola de espera — {len(cola)} paciente{'s' if len(cola)!=1 else ''}
        </p>
        <table style="width:100%;border-collapse:collapse;">
            <thead>
                <tr style="border-bottom:1px solid #3a3a5a;">
                    <th style="text-align:left;padding:3px 8px;color:#aaa;font-size:0.7rem;font-weight:600;">Prior.</th>
                    <th style="text-align:left;padding:3px 8px;color:#aaa;font-size:0.7rem;font-weight:600;">ID</th>
                    <th style="text-align:left;padding:3px 8px;color:#aaa;font-size:0.7rem;font-weight:600;">Área req.</th>
                    <th style="text-align:left;padding:3px 8px;color:#aaa;font-size:0.7rem;font-weight:600;">Espera</th>
                    <th style="padding:3px 8px;"></th>
                </tr>
            </thead>
            <tbody>{filas}</tbody>
        </table>
        {resto}
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PANEL DE MODO ASISTIDO — confirmación de acciones pendientes
# ─────────────────────────────────────────────────────────────────────────────

def _renderizar_panel_asistido(sim) -> bool:
    """Renderiza el panel de confirmación del modo asistido.

    Retorna True si hay acciones pendientes que bloquean el avance.
    """
    if not sim or not sim.modo_asistido or not sim.acciones_pendientes:
        return False

    pendientes = sim.acciones_pendientes
    st.markdown(
        "<div style='background:#1e1e3a;border:1px solid #BB86FC;"
        "border-radius:10px;padding:14px 18px;margin-bottom:12px;'>"
        "<p style='color:#BB86FC;font-weight:700;font-size:0.9rem;"
        "margin-bottom:8px;'>🤝 Modo asistido — acciones propuestas</p>"
        "<p style='color:#aaa;font-size:0.75rem;'>"
        "El sistema propone las siguientes redistribuciones. "
        "Selecciona las que deseas confirmar y pulsa Ejecutar.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    seleccionadas = []
    for i, accion in enumerate(pendientes):
        checked = st.checkbox(
            accion.descripcion,
            value=True,
            key=f"accion_{sim.tick_actual}_{i}",
        )
        if checked:
            seleccionadas.append(i)

    col_exec, col_skip = st.columns([1, 3])
    with col_exec:
        if st.button("✅ Ejecutar seleccionadas", use_container_width=True):
            n = confirmar_acciones_pendientes(sim, seleccionadas)
            st.success(f"{n} acción(es) ejecutada(s).")
            st.session_state["autoavance"] = True
            st.rerun()
    with col_skip:
        if st.button("⏭ Omitir todas", use_container_width=True):
            sim.acciones_pendientes = []
            st.session_state["autoavance"] = True
            st.rerun()

    return True


# ─────────────────────────────────────────────────────────────────────────────
# PANEL DE RESUMEN FINAL
# ─────────────────────────────────────────────────────────────────────────────

def _renderizar_resumen_final(sim):
    resumen = calcular_resumen(sim)
    if not resumen:
        return

    st.markdown("---")
    st.markdown(
        "<h3 style='color:#BB86FC;font-size:1rem;'>📋 Resumen de evaluación (régimen estable)</h3>",
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("O media", f"{resumen['O_media']:.1f}%")
        st.metric("O P5–P95", f"{resumen['O_p5']:.0f}%–{resumen['O_p95']:.0f}%")
    with col2:
        st.metric("I media", f"{resumen['I_media']:.1f}")
        st.metric("Nivel modal", resumen["nivel_I_modal"])
    with col3:
        st.metric("Traslados totales", resumen["traslados_total"])
        st.metric("Alertas RES-04", resumen["alertas_total"])
    with col4:
        rmse = resumen.get("pred_rmse")
        st.metric("RMSE predicción", f"{rmse:.3f} pp" if rmse else "—")
        st.metric("Altas totales", resumen["altas_total"])

    cumple_O = resumen["cumple_O_rango"]
    cumple_I = resumen["cumple_I_nivel"]
    st.markdown(
        f"<div style='display:flex;gap:16px;margin-top:8px;'>"
        f"<span style='color:{'#4CAF50' if cumple_O else '#F44336'};font-size:0.82rem;'>"
        f"{'✅' if cumple_O else '❌'} CE-B rango O</span>"
        f"<span style='color:{'#4CAF50' if cumple_I else '#F44336'};font-size:0.82rem;'>"
        f"{'✅' if cumple_I else '❌'} CE-B nivel I</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def main():
    _init_session()
    modelo = cargar_modelo()

    escenario, modo_asistido, velocidad, seed = _renderizar_sidebar(modelo)

    # Título
    st.markdown(
        "<h1 style='color:#e0e0e0;font-size:1.4rem;font-weight:700;"
        "margin-bottom:2px;'>Simulador Inteligente de Ocupación Hospitalaria</h1>"
        "<p style='color:#666;font-size:0.78rem;margin-bottom:16px;'>"
        "Maestría IA y CD · UTP · 2026 · CRISP-DM/S</p>",
        unsafe_allow_html=True,
    )

    sim = st.session_state["sim"]

    # ── Sin simulación activa ─────────────────────────────────────────────────
    if sim is None:
        st.markdown(
            "<div style='background:#16213e;border:1px solid #2a2a4a;"
            "border-radius:10px;padding:30px;text-align:center;margin-top:40px;'>"
            "<p style='font-size:2rem;margin-bottom:8px;'>🏥</p>"
            "<p style='color:#BB86FC;font-size:1rem;font-weight:700;'>"
            "Configura y pulsa Iniciar para comenzar la simulación</p>"
            "<p style='color:#666;font-size:0.8rem;margin-top:6px;'>"
            "Selecciona el escenario, modo y velocidad en el panel lateral.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Autoavance tick-a-tick ────────────────────────────────────────────────
    if st.session_state["autoavance"] and sim.activa and not sim.pausada:
        # Bloquear si hay acciones pendientes en modo asistido
        if sim.modo_asistido and sim.acciones_pendientes:
            st.session_state["autoavance"] = False
        else:
            resultado, acciones_pend = avanzar_tick(sim, modelo)
            if resultado:
                st.session_state["ultimo_resultado"] = resultado
                st.session_state["ultimo_snap"] = snapshot_hospital(sim)
            if sim.modo_asistido and acciones_pend:
                st.session_state["autoavance"] = False
            elif not sim.finalizada:
                delay = max(0.02, 0.3 / velocidad)
                time.sleep(delay)
                st.rerun()

    snap      = st.session_state["ultimo_snap"]
    resultado = st.session_state["ultimo_resultado"]

    # ── Panel asistido (bloquea el resto si hay pendientes) ───────────────────
    hay_pendientes = _renderizar_panel_asistido(sim)

    # ── Métricas en tiempo real ───────────────────────────────────────────────
    if snap:
        _renderizar_metricas(snap, sim.historial)

    st.markdown("---")

    # ── Layout de dos columnas: grid hospital | cola + gráfico ───────────────
    col_hosp, col_derecha = st.columns([3, 2])

    with col_hosp:
        st.markdown(
            "<p style='color:#BB86FC;font-size:0.82rem;font-weight:700;"
            "letter-spacing:0.05em;margin-bottom:6px;'>DISTRIBUCIÓN DE CAMAS</p>",
            unsafe_allow_html=True,
        )
        if snap:
            grid_html = _html_grid_hospital(snap)
            st.components.v1.html(grid_html, height=420, scrolling=True)
        else:
            st.markdown(
                "<p style='color:#555;font-size:0.8rem;'>"
                "El hospital aparecerá aquí al iniciar la simulación.</p>",
                unsafe_allow_html=True,
            )

    with col_derecha:
        st.markdown(
            "<p style='color:#BB86FC;font-size:0.82rem;font-weight:700;"
            "letter-spacing:0.05em;margin-bottom:6px;'>COLA DE ESPERA</p>",
            unsafe_allow_html=True,
        )
        if snap:
            _renderizar_cola(snap)

    # ── Gráfico de evolución temporal ─────────────────────────────────────────
    if sim.historial:
        st.markdown("---")
        st.markdown(
            "<p style='color:#BB86FC;font-size:0.82rem;font-weight:700;"
            "letter-spacing:0.05em;margin-bottom:6px;'>EVOLUCIÓN TEMPORAL</p>",
            unsafe_allow_html=True,
        )
        fig = _fig_tiempo_real(sim.historial, sim.escenario)
        if fig:
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # ── Resumen final (solo cuando finaliza) ──────────────────────────────────
    if sim.finalizada:
        _renderizar_resumen_final(sim)

    # ── Rerun si la simulación sigue corriendo ────────────────────────────────
    if sim.activa and not sim.pausada and not sim.finalizada and not hay_pendientes:
        time.sleep(max(0.02, 0.3 / velocidad))
        st.rerun()


if __name__ == "__main__":
    main()
