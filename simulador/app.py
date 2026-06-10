"""
app.py
======
Simulador Inteligente de Ocupación Hospitalaria — F6
Autor: Juan Camilo García Braham · UTP · IA en Salud · 2026
Ejecución: streamlit run app.py

Flujo de páginas (D_F6_008):
  PÁGINA 0 — Configurar hospital   → define camas por área y parámetros
  PÁGINA 1 — Simulador en curso    → grid en vivo, cola, métricas
  PÁGINA 2 — Resultados finales    → resumen CE-B, gráficos
"""

import time, os, sys
import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from motor_simulacion import (
    ConfigHospital, crear_estado, avanzar_tick,
    confirmar_acciones_pendientes, calcular_resumen, snapshot_hospital,
    COLOR_NIVEL, COLOR_CAMA, TICKS_TOTAL, WARM_UP_TICKS,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN STREAMLIT
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Simulador Hospitalario · UTP",
    page_icon="🏥", layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #1a1a2e; color: #e0e0e0;
    font-family: 'Liberation Sans','DejaVu Sans',sans-serif;
}
[data-testid="stSidebar"] { background-color: #16213e; border-right:1px solid #2a2a4a; }
[data-testid="stSidebar"] * { color:#e0e0e0 !important; }
[data-testid="stMetric"] {
    background:#16213e; border:1px solid #2a2a4a;
    border-radius:8px; padding:10px 14px;
}
[data-testid="stMetricLabel"] { color:#aaaaaa !important; font-size:0.72rem; }
[data-testid="stMetricValue"] { color:#BB86FC !important; font-size:1.3rem; font-weight:700; }
.stButton > button {
    background:#BB86FC; color:#1a1a2e; border:none;
    border-radius:6px; font-weight:700; padding:6px 18px;
}
.stButton > button:hover { opacity:0.85; }
div[data-testid="stNumberInput"] label,
div[data-testid="stSlider"] label,
div[data-testid="stSelectbox"] label { color:#aaaaaa !important; font-size:0.78rem; }
.streamlit-expanderHeader { color:#BB86FC !important; font-weight:600; }
hr { border-color:#2a2a4a; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE MODELO (cacheado)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    return joblib.load(os.path.join(BASE_DIR, "modelo_final_f4b.pkl"))

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def _init():
    defaults = {
        "pagina":       "config",   # "config" | "simular" | "resultados"
        "sim":          None,
        "ultimo_snap":  None,
        "autoavance":   False,
        "cfg":          ConfigHospital(),
        "escenario":    "normal",
        "modo_asistido":False,
        "velocidad":    3,
        "seed":         99,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()

# ─────────────────────────────────────────────────────────────────────────────
# GRID HTML DEL HOSPITAL — D_F6_003 / D_F6_007
# Panel de ficha de paciente con JS: click en cama → detalle lateral
# ─────────────────────────────────────────────────────────────────────────────
def _html_grid(snap: dict, tick_actual: int, ticks_total: int) -> str:
    if not snap:
        return ""

    ind = snap["indicador"]
    nivel     = ind.get("nivel", "Bajo")
    color_nv  = COLOR_NIVEL.get(nivel, "#aaa")
    O_val     = ind.get("O", 0.0)
    I_val     = ind.get("I", 0.0)
    cola_snap = snap.get("cola_espera", [])

    # ── Leyenda ──────────────────────────────────────────────────────────────
    leyenda = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:5px;'
        f'margin-right:12px;font-size:0.7rem;color:#ccc;">'
        f'<span style="width:11px;height:11px;border-radius:3px;'
        f'background:{c};display:inline-block;"></span>{lbl}</span>'
        for lbl, c in [("Libre", COLOR_CAMA["libre"]),
                       ("Ocupada", COLOR_CAMA["ocupada"]),
                       ("En limpieza", COLOR_CAMA["en_limpieza"]),
                       ("Temporal", COLOR_CAMA["temporal"])]
    )

    # ── Celdas por área ───────────────────────────────────────────────────────
    areas_html = ""
    for area in snap["areas"]:
        nombre     = area["nombre"]
        cap        = area["capacidad_total"]
        cap_dis    = area["cap_disponible"]
        camas      = area["camas"]
        ocupadas   = sum(1 for c in camas if c["estado"] in ("ocupada", "temporal"))

        color_estado = (
            "#F44336" if cap_dis <= 0
            else "#FF9800" if cap_dis <= max(1, cap * 0.2)
            else "#4CAF50"
        )
        desborde_txt = "⚠ DESBORDE" if cap_dis <= 0 else f"Disp: {cap_dis}"

        celdas = ""
        for cama in camas:
            estado = cama["estado"]
            color  = COLOR_CAMA.get(estado, "#555")
            pac    = cama["paciente"]

            if pac:
                # Datos para el panel JS (escapados para onclick)
                edad_txt = f"{pac['edad']} años · " if pac.get("edad") else ""
                js_data  = (
                    f"ID: {pac['id']} | "
                    f"{pac['prioridad']} | "
                    f"{edad_txt}"
                    f"Área req: {pac['area_req']} | "
                    f"Estancia: {pac['estancia_h']:.1f}h | "
                    f"Espera acum: {pac['espera_h']:.1f}h"
                ).replace("'", "\\'")
                prioridad_color = {
                    "P1": "#F44336", "P2": "#FF9800",
                    "P3": "#4CAF50", "P4": "#2196F3"
                }.get(pac["prioridad"], "#aaa")

                cama_id      = cama["id"]
                pac_id       = pac["id"]
                pac_prior    = pac["prioridad"]
                pac_area     = pac["area_req"]
                pac_estancia = f"{pac['estancia_h']:.1f} h"
                pac_espera   = f"{pac['espera_h']:.1f} h"
                onclick = (
                    f"document.getElementById('ficha-id').innerText='{pac_id}';"
                    f"document.getElementById('ficha-prior').innerText='{pac_prior}';"
                    f"document.getElementById('ficha-prior').style.background='{prioridad_color}';"
                    f"document.getElementById('ficha-area').innerText='{pac_area}';"
                    f"document.getElementById('ficha-estancia').innerText='{pac_estancia}';"
                    f"document.getElementById('ficha-espera').innerText='{pac_espera}';"
                    f"document.getElementById('ficha-cama').innerText='{nombre} · {cama_id}';"
                    f"document.getElementById('panel-paciente').style.display='block';"
                )
                celdas += (
                    f'<div onclick="{onclick}" title="{js_data}" style="'
                    f'width:18px;height:18px;border-radius:3px;background:{color};'
                    f'cursor:pointer;border:1px solid rgba(255,255,255,0.1);'
                    f'transition:transform 0.1s;" '
                    f'onmouseover="this.style.transform=\'scale(1.5)\'" '
                    f'onmouseout="this.style.transform=\'scale(1)\'"></div>'
                )
            else:
                tooltip = "En limpieza" if estado == "en_limpieza" else "Libre"
                onclick_clear = (
                    "document.getElementById('panel-paciente').style.display='none';"
                )
                celdas += (
                    f'<div onclick="{onclick_clear}" title="{tooltip}" style="'
                    f'width:18px;height:18px;border-radius:3px;background:{color};'
                    f'cursor:default;border:1px solid rgba(255,255,255,0.08);'
                    f'transition:transform 0.1s;" '
                    f'onmouseover="this.style.transform=\'scale(1.4)\'" '
                    f'onmouseout="this.style.transform=\'scale(1)\'"></div>'
                )

        areas_html += f"""
        <div style="margin-bottom:14px;">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
            <span style="font-size:0.78rem;font-weight:700;color:#BB86FC;">{nombre}</span>
            <span style="font-size:0.68rem;color:#aaa;background:#2a2a4a;
                         border-radius:4px;padding:1px 6px;">{ocupadas}/{cap}</span>
            <span style="font-size:0.68rem;font-weight:600;color:{color_estado};">{desborde_txt}</span>
          </div>
          <div style="display:flex;flex-wrap:wrap;gap:4px;">{celdas}</div>
        </div>"""

    # ── Cola de espera (HTML dentro del grid) ─────────────────────────────────
    if cola_snap:
        filas_cola = ""
        for p in cola_snap[:12]:
            cp = {"P1":"#F44336","P2":"#FF9800","P3":"#4CAF50","P4":"#2196F3"}.get(p["prioridad"],"#aaa")
            alerta = " ⚠" if p["ticks_espera"] >= 2 and p["prioridad"] in ("P1","P2") else ""
            filas_cola += (
                f'<tr style="border-bottom:1px solid #2a2a4a;">'
                f'<td style="padding:3px 6px;">'
                f'<span style="background:{cp};color:#1a1a2e;border-radius:4px;'
                f'padding:1px 5px;font-size:0.68rem;font-weight:700;">{p["prioridad"]}</span></td>'
                f'<td style="padding:3px 6px;font-size:0.68rem;color:#ccc;font-family:monospace;">{p["id"]}</td>'
                f'<td style="padding:3px 6px;font-size:0.68rem;color:#ccc;">{p["area_req"]}</td>'
                f'<td style="padding:3px 6px;font-size:0.68rem;color:#FF9800;">{p["espera_h"]:.1f}h{alerta}</td>'
                f'</tr>'
            )
        resto_cola = (f'<p style="color:#666;font-size:0.66rem;margin-top:3px;">… y {len(cola_snap)-12} más</p>'
                      if len(cola_snap) > 12 else "")
        cola_html = f"""
        <div style="margin-top:14px;background:#1a1a3a;border-radius:8px;padding:10px;border:1px solid #3a3a6a;">
          <p style="color:#BB86FC;font-size:0.75rem;font-weight:700;margin-bottom:6px;">
            🪑 Cola de espera — {len(cola_snap)} paciente{'s' if len(cola_snap)!=1 else ''}</p>
          <table style="width:100%;border-collapse:collapse;">
            <thead><tr style="border-bottom:1px solid #3a3a6a;">
              <th style="text-align:left;padding:2px 6px;color:#888;font-size:0.65rem;">Prior.</th>
              <th style="text-align:left;padding:2px 6px;color:#888;font-size:0.65rem;">ID</th>
              <th style="text-align:left;padding:2px 6px;color:#888;font-size:0.65rem;">Área req.</th>
              <th style="text-align:left;padding:2px 6px;color:#888;font-size:0.65rem;">Espera</th>
            </tr></thead>
            <tbody>{filas_cola}</tbody>
          </table>
          {resto_cola}
        </div>"""
    else:
        cola_html = '<p style="color:#4CAF50;font-size:0.75rem;margin-top:10px;">✓ Sin pacientes en espera</p>'

    # ── Barra de progreso interna ─────────────────────────────────────────────
    pct = int(tick_actual / max(ticks_total, 1) * 100)
    progreso_html = f"""
    <div style="margin-bottom:12px;">
      <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
        <span style="font-size:0.68rem;color:#aaa;">Tick {tick_actual} / {ticks_total} · {tick_actual*0.25:.1f}h simuladas</span>
        <span style="font-size:0.68rem;color:#BB86FC;">{pct}%</span>
      </div>
      <div style="background:#2a2a4a;border-radius:4px;height:5px;">
        <div style="background:#BB86FC;width:{pct}%;height:5px;border-radius:4px;transition:width 0.3s;"></div>
      </div>
    </div>"""

    # ── Métricas top (dentro del HTML, actualizan en cada rerun) ──────────────
    metricas_html = f"""
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px;">
      <div style="background:#16213e;border:1px solid #2a2a4a;border-radius:8px;
                  padding:8px 14px;min-width:90px;">
        <div style="font-size:0.65rem;color:#aaa;">Ocupación O</div>
        <div style="font-size:1.1rem;font-weight:700;color:#BB86FC;">{O_val:.1f}%</div>
      </div>
      <div style="background:#16213e;border:1px solid #2a2a4a;border-radius:8px;
                  padding:8px 14px;min-width:90px;">
        <div style="font-size:0.65rem;color:#aaa;">Indicador I</div>
        <div style="font-size:1.1rem;font-weight:700;color:#BB86FC;">{I_val:.1f}</div>
      </div>
      <div style="background:{color_nv}22;border:1px solid {color_nv}55;border-radius:8px;
                  padding:8px 14px;min-width:90px;">
        <div style="font-size:0.65rem;color:#aaa;">Nivel</div>
        <div style="font-size:1.1rem;font-weight:700;color:{color_nv};">{nivel}</div>
      </div>
      <div style="background:#16213e;border:1px solid #2a2a4a;border-radius:8px;
                  padding:8px 14px;min-width:90px;">
        <div style="font-size:0.65rem;color:#aaa;">En cola</div>
        <div style="font-size:1.1rem;font-weight:700;color:#FF9800;">{len(cola_snap)}</div>
      </div>
    </div>"""

    # ── Panel de ficha de paciente (oculto hasta click) ───────────────────────
    panel_paciente = """
    <div id="panel-paciente" style="
        display:none; margin-top:14px;
        background:#1e1e3a; border:1px solid #BB86FC55;
        border-radius:10px; padding:14px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
        <span style="color:#BB86FC;font-size:0.78rem;font-weight:700;">📋 Ficha del paciente</span>
        <span onclick="document.getElementById('panel-paciente').style.display='none'"
              style="cursor:pointer;color:#666;font-size:0.75rem;">✕ cerrar</span>
      </div>
      <table style="width:100%;border-collapse:collapse;">
        <tr><td style="color:#888;font-size:0.7rem;padding:3px 0;">ID</td>
            <td id="ficha-id" style="color:#e0e0e0;font-size:0.72rem;font-family:monospace;padding:3px 0;"></td></tr>
        <tr><td style="color:#888;font-size:0.7rem;padding:3px 0;">Prioridad</td>
            <td style="padding:3px 0;"><span id="ficha-prior"
              style="border-radius:4px;padding:1px 8px;font-size:0.7rem;font-weight:700;color:#1a1a2e;"></span></td></tr>
        <tr><td style="color:#888;font-size:0.7rem;padding:3px 0;">Área req.</td>
            <td id="ficha-area" style="color:#e0e0e0;font-size:0.72rem;padding:3px 0;"></td></tr>
        <tr><td style="color:#888;font-size:0.7rem;padding:3px 0;">Cama</td>
            <td id="ficha-cama" style="color:#e0e0e0;font-size:0.72rem;padding:3px 0;"></td></tr>
        <tr><td style="color:#888;font-size:0.7rem;padding:3px 0;">Estancia</td>
            <td id="ficha-estancia" style="color:#e0e0e0;font-size:0.72rem;padding:3px 0;"></td></tr>
        <tr><td style="color:#888;font-size:0.7rem;padding:3px 0;">Espera acum.</td>
            <td id="ficha-espera" style="color:#e0e0e0;font-size:0.72rem;padding:3px 0;"></td></tr>
      </table>
    </div>"""

    return f"""
    <div style="background:#16213e;border-radius:10px;padding:16px;border:1px solid #2a2a4a;">
      {progreso_html}
      {metricas_html}
      <div style="margin-bottom:10px;display:flex;flex-wrap:wrap;gap:3px;">{leyenda}</div>
      <p style="color:#666;font-size:0.66rem;margin-bottom:10px;">
        Haz clic en una cama ocupada (🔴) para ver la ficha del paciente.</p>
      {areas_html}
      {cola_html}
      {panel_paciente}
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA MATPLOTLIB — evolución temporal
# ─────────────────────────────────────────────────────────────────────────────
def _fig_evolucion(historial, mostrar_todo=False):
    if not historial or len(historial) < 3:
        return None
    h = historial if mostrar_todo else historial[-min(60, len(historial)):]
    ticks  = [r.tick for r in h]
    O_vals = [r.O_t  for r in h]
    I_vals = [r.I_t  for r in h]
    pred_x = [r.tick + 4 for r in h if r.pred_O_t4 is not None]
    pred_y = [r.pred_O_t4  for r in h if r.pred_O_t4 is not None]

    fig = plt.figure(figsize=(11, 3.5), facecolor="#1a1a2e")
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.32)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    for ax in (ax1, ax2):
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#aaaaaa", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

    ax1.plot(ticks, O_vals, color="#BB86FC", lw=1.4, label="O(t) real")
    if pred_x:
        ax1.plot(pred_x, pred_y, color="#03DAC6", lw=1.0, ls="--",
                 label="O pred. T+4", alpha=0.85)
    if mostrar_todo and len(historial) > WARM_UP_TICKS:
        ax1.axvline(WARM_UP_TICKS, color="#555", ls=":", lw=0.8)
        ax1.text(WARM_UP_TICKS+1, 2, "warm-up", color="#666", fontsize=6, va="bottom")
    ax1.set_xlabel("Tick", color="#aaa", fontsize=7)
    ax1.set_ylabel("Ocupación (%)", color="#aaa", fontsize=7)
    ax1.set_title("O(t) real vs Predicción T+4", color="#e0e0e0", fontsize=8, pad=4)
    ax1.set_ylim(0, 105)
    ax1.legend(fontsize=6, facecolor="#1a1a2e", labelcolor="#ccc",
               loc="upper left", framealpha=0.7)

    for y0, y1, color in [(0,25,"#4CAF50"),(25,50,"#FF9800"),
                           (50,75,"#FF5722"),(75,100,"#F44336")]:
        ax2.axhspan(y0, y1, alpha=0.08, color=color)
        ax2.axhline(y1, color=color, lw=0.4, ls="--", alpha=0.4)
    ax2.plot(ticks, I_vals, color="#BB86FC", lw=1.4)
    if I_vals:
        ax2.axhline(I_vals[-1], color="#03DAC6", lw=0.7, ls=":", alpha=0.6)
    ax2.set_xlabel("Tick", color="#aaa", fontsize=7)
    ax2.set_ylabel("Indicador I", color="#aaa", fontsize=7)
    ax2.set_title("Indicador Compuesto I(t)", color="#e0e0e0", fontsize=8, pad=4)
    ax2.set_ylim(0, 100)
    fig.tight_layout(pad=1.1)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA 0 — CONFIGURAR HOSPITAL (D_F6_008)
# ─────────────────────────────────────────────────────────────────────────────
def pagina_config():
    st.markdown(
        "<h1 style='color:#e0e0e0;font-size:1.5rem;font-weight:700;margin-bottom:4px;'>"
        "🏥 Simulador Inteligente de Ocupación Hospitalaria</h1>"
        "<p style='color:#666;font-size:0.78rem;margin-bottom:24px;'>"
        "Maestría IA y Ciencia de Datos · UTP · 2026 · CRISP-DM/S</p>",
        unsafe_allow_html=True,
    )

    col_izq, col_der = st.columns([1, 1], gap="large")

    with col_izq:
        st.markdown(
            "<p style='color:#BB86FC;font-size:0.85rem;font-weight:700;"
            "letter-spacing:0.06em;margin-bottom:14px;'>INFRAESTRUCTURA DEL HOSPITAL</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='background:#16213e;border:1px solid #2a2a4a;border-radius:10px;"
            "padding:6px 16px 14px;margin-bottom:8px;'>"
            "<p style='color:#aaa;font-size:0.72rem;margin-bottom:0;margin-top:8px;'>"
            "Piso 1 — Urgencias y UCI</p></div>",
            unsafe_allow_html=True,
        )
        uci = st.number_input("Camas UCI (P1 críticos)", min_value=2, max_value=30,
                               value=st.session_state["cfg"].uci, step=1)
        urgencias = st.number_input("Camas Urgencias (P1/P2/P3)", min_value=5, max_value=60,
                                     value=st.session_state["cfg"].urgencias, step=1)
        sala = st.number_input("Camas Sala de espera (P3/P4)", min_value=2, max_value=30,
                                value=st.session_state["cfg"].sala_espera, step=1)

        st.markdown(
            "<div style='background:#16213e;border:1px solid #2a2a4a;border-radius:10px;"
            "padding:6px 16px 14px;margin-top:10px;margin-bottom:8px;'>"
            "<p style='color:#aaa;font-size:0.72rem;margin-bottom:0;margin-top:8px;'>"
            "Piso 2 — Hospitalización y Observación</p></div>",
            unsafe_allow_html=True,
        )
        hosp = st.number_input("Camas Hospitalización (P2/P3)", min_value=10, max_value=100,
                                value=st.session_state["cfg"].hospitalizacion, step=1)
        obs  = st.number_input("Camas Observación (P2/P3/P4)", min_value=5, max_value=40,
                                value=st.session_state["cfg"].observacion, step=1)

        total = uci + urgencias + hosp + obs + sala
        st.markdown(
            f"<p style='color:#BB86FC;font-size:0.82rem;margin-top:8px;'>"
            f"Total de camas: <strong>{total}</strong></p>",
            unsafe_allow_html=True,
        )

    with col_der:
        st.markdown(
            "<p style='color:#BB86FC;font-size:0.85rem;font-weight:700;"
            "letter-spacing:0.06em;margin-bottom:14px;'>PARÁMETROS DE SIMULACIÓN</p>",
            unsafe_allow_html=True,
        )
        escenario = st.selectbox(
            "Escenario de demanda",
            options=["normal", "alta_demanda", "crisis"],
            index=["normal","alta_demanda","crisis"].index(st.session_state["escenario"]),
            format_func=lambda x: {
                "normal":       "Normal — λ=1.5 pac/tick (O esperada ~43%)",
                "alta_demanda": "Alta demanda — λ=3.0 pac/tick (O esperada ~62%)",
                "crisis":       "Crisis — λ=5.0 pac/tick (O esperada ~81%)",
            }[x],
        )
        modo_asistido = st.toggle(
            "Modo asistido",
            value=st.session_state["modo_asistido"],
            help=(
                "Activado: tú confirmas cada traslado o desborde propuesto por el sistema.\n"
                "Desactivado: el sistema actúa de forma completamente autónoma."
            ),
        )
        velocidad = st.slider("Velocidad de simulación", 1, 10,
                               st.session_state["velocidad"], step=1, format="%dx")
        seed = st.number_input("Semilla aleatoria (reproducibilidad)",
                                min_value=0, max_value=9999,
                                value=st.session_state["seed"], step=1)

        st.markdown("<br>", unsafe_allow_html=True)

        # Vista previa del hospital
        st.markdown(
            "<p style='color:#aaa;font-size:0.75rem;margin-bottom:8px;'>"
            "Vista previa del hospital configurado:</p>",
            unsafe_allow_html=True,
        )
        preview_rows = ""
        for nombre, n, piso in [
            ("UCI",             uci,      "Piso 1"),
            ("Urgencias",       urgencias,"Piso 1"),
            ("Sala de espera",  sala,     "Piso 1"),
            ("Hospitalización", hosp,     "Piso 2"),
            ("Observación",     obs,      "Piso 2"),
        ]:
            celdas = "".join(
                f'<span style="display:inline-block;width:9px;height:9px;'
                f'border-radius:2px;background:#4CAF50;margin:1px;"></span>'
                for _ in range(min(n, 40))
            )
            sufijo = f" +{n-40}" if n > 40 else ""
            preview_rows += (
                f'<tr>'
                f'<td style="color:#aaa;font-size:0.68rem;padding:3px 8px 3px 0;'
                f'white-space:nowrap;">{nombre}</td>'
                f'<td style="font-size:0.66rem;color:#555;padding:3px 8px 3px 0;">{piso}</td>'
                f'<td style="padding:3px 0;">{celdas}'
                f'<span style="color:#666;font-size:0.66rem;">{sufijo} ({n})</span></td>'
                f'</tr>'
            )
        st.markdown(
            f'<div style="background:#16213e;border:1px solid #2a2a4a;'
            f'border-radius:8px;padding:12px;">'
            f'<table style="border-collapse:collapse;width:100%;">{preview_rows}</table>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        if st.button("▶ Iniciar simulación", use_container_width=True):
            st.session_state["cfg"] = ConfigHospital(
                uci=int(uci), urgencias=int(urgencias),
                hospitalizacion=int(hosp), observacion=int(obs),
                sala_espera=int(sala),
            )
            st.session_state["escenario"]     = escenario
            st.session_state["modo_asistido"] = modo_asistido
            st.session_state["velocidad"]     = velocidad
            st.session_state["seed"]          = int(seed)
            st.session_state["sim"] = crear_estado(
                escenario=escenario, modo_asistido=modo_asistido,
                seed=int(seed), config=st.session_state["cfg"],
            )
            st.session_state["ultimo_snap"] = None
            st.session_state["autoavance"]  = True
            st.session_state["pagina"]      = "simular"
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA 1 — SIMULADOR EN CURSO
# ─────────────────────────────────────────────────────────────────────────────
def pagina_simular(modelo):
    sim = st.session_state["sim"]

    # Cabecera con controles
    col_titulo, col_ctrl = st.columns([3, 2])
    with col_titulo:
        escenario_lbl = {
            "normal": "Normal (λ=1.5)", "alta_demanda": "Alta demanda (λ=3.0)",
            "crisis": "Crisis (λ=5.0)",
        }.get(sim.escenario, sim.escenario)
        modo_lbl = "🤝 Asistido" if sim.modo_asistido else "🤖 Automático"
        st.markdown(
            f"<h2 style='color:#e0e0e0;font-size:1.1rem;font-weight:700;margin-bottom:2px;'>"
            f"🏥 Simulador en curso — {escenario_lbl} · {modo_lbl}</h2>",
            unsafe_allow_html=True,
        )
    with col_ctrl:
        c1, c2, c3 = st.columns(3)
        with c1:
            label = "⏸ Pausar" if (sim.activa and not sim.pausada) else "▶ Reanudar"
            if st.button(label, use_container_width=True):
                sim.pausada = not sim.pausada
                st.session_state["autoavance"] = not sim.pausada
                st.rerun()
        with c2:
            if st.button("⚙ Config.", use_container_width=True):
                st.session_state["pagina"]    = "config"
                st.session_state["sim"]       = None
                st.session_state["autoavance"]= False
                st.rerun()
        with c3:
            velocidad = st.session_state["velocidad"]

    st.markdown("---")

    # ── Avanzar tick ─────────────────────────────────────────────────────────
    hay_pendientes = False
    if st.session_state["autoavance"] and sim.activa and not sim.pausada:
        if sim.modo_asistido and sim.acciones_pendientes:
            st.session_state["autoavance"] = False
            hay_pendientes = True
        else:
            resultado, acciones_pend = avanzar_tick(sim, modelo)
            if resultado:
                st.session_state["ultimo_snap"] = snapshot_hospital(sim)
            if sim.modo_asistido and acciones_pend:
                st.session_state["autoavance"] = False
                hay_pendientes = True
            elif sim.finalizada:
                st.session_state["autoavance"] = False
                time.sleep(0.05)

    snap = st.session_state["ultimo_snap"]

    # ── Panel modo asistido ───────────────────────────────────────────────────
    if hay_pendientes or (sim.modo_asistido and sim.acciones_pendientes):
        hay_pendientes = True
        with st.container():
            st.markdown(
                "<div style='background:#1e1e3a;border:1px solid #BB86FC;"
                "border-radius:10px;padding:14px 18px;margin-bottom:12px;'>"
                "<p style='color:#BB86FC;font-weight:700;font-size:0.88rem;margin-bottom:6px;'>"
                "🤝 Modo asistido — acciones propuestas</p>"
                "<p style='color:#aaa;font-size:0.72rem;margin-bottom:10px;'>"
                "El sistema propone estas redistribuciones. "
                "Selecciona las que deseas ejecutar.</p></div>",
                unsafe_allow_html=True,
            )
            pendientes = sim.acciones_pendientes
            seleccionadas = []
            for i, accion in enumerate(pendientes):
                if st.checkbox(accion.descripcion, value=True,
                               key=f"accion_{sim.tick_actual}_{i}"):
                    seleccionadas.append(i)

            ca, cb = st.columns([1, 3])
            with ca:
                if st.button("✅ Ejecutar", use_container_width=True):
                    confirmar_acciones_pendientes(sim, seleccionadas)
                    st.session_state["autoavance"] = True
                    st.rerun()
            with cb:
                if st.button("⏭ Omitir todas", use_container_width=True):
                    sim.acciones_pendientes = []
                    st.session_state["autoavance"] = True
                    st.rerun()

    # ── Layout: grid + gráfico ────────────────────────────────────────────────
    col_grid, col_graf = st.columns([3, 2], gap="medium")

    with col_grid:
        st.markdown(
            "<p style='color:#BB86FC;font-size:0.78rem;font-weight:700;"
            "letter-spacing:0.05em;margin-bottom:6px;'>DISTRIBUCIÓN EN TIEMPO REAL</p>",
            unsafe_allow_html=True,
        )
        if snap:
            grid_html = _html_grid(snap, sim.tick_actual, sim.ticks_total)
            st.components.v1.html(grid_html, height=580, scrolling=True)
        else:
            st.markdown(
                "<p style='color:#555;font-size:0.8rem;'>Iniciando simulación…</p>",
                unsafe_allow_html=True,
            )

    with col_graf:
        st.markdown(
            "<p style='color:#BB86FC;font-size:0.78rem;font-weight:700;"
            "letter-spacing:0.05em;margin-bottom:6px;'>EVOLUCIÓN TEMPORAL</p>",
            unsafe_allow_html=True,
        )
        if sim.historial:
            fig = _fig_evolucion(sim.historial, mostrar_todo=False)
            if fig:
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            # Últimas llegadas del tick más reciente
            if sim.historial:
                ultimo = sim.historial[-1]
                st.markdown(
                    f"<div style='background:#16213e;border:1px solid #2a2a4a;"
                    f"border-radius:8px;padding:10px 14px;margin-top:8px;'>"
                    f"<p style='color:#aaa;font-size:0.68rem;margin-bottom:6px;'>"
                    f"Último tick (#{ultimo.tick})</p>"
                    f"<div style='display:flex;gap:12px;flex-wrap:wrap;'>"
                    f"<span style='font-size:0.72rem;color:#03DAC6;'>🚑 Llegadas: <b>{ultimo.llegadas_t}</b></span>"
                    f"<span style='font-size:0.72rem;color:#4CAF50;'>✓ Altas: <b>{ultimo.altas_t}</b></span>"
                    f"<span style='font-size:0.72rem;color:#FF9800;'>⟶ Traslados: <b>{ultimo.traslados_t}</b></span>"
                    f"<span style='font-size:0.72rem;color:#F44336;'>⚠ Alertas: <b>{ultimo.n_alertas}</b></span>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )

    # ── Finalización ─────────────────────────────────────────────────────────
    if sim.finalizada:
        st.success("✅ Simulación completada — revisa el resumen a continuación")
        if st.button("📋 Ver resultados completos", use_container_width=False):
            st.session_state["pagina"] = "resultados"
            st.rerun()
        pagina_resumen_inline(sim)
        return

    # ── Rerun si sigue corriendo ──────────────────────────────────────────────
    if sim.activa and not sim.pausada and not hay_pendientes:
        delay = max(0.05, 0.35 / velocidad)
        time.sleep(delay)
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA 2 — RESULTADOS FINALES
# ─────────────────────────────────────────────────────────────────────────────
def pagina_resumen_inline(sim):
    resumen = calcular_resumen(sim)
    if not resumen:
        return
    st.markdown("---")
    st.markdown(
        "<h3 style='color:#BB86FC;font-size:0.95rem;'>📊 Resumen — régimen estable "
        f"(ticks {WARM_UP_TICKS}–{sim.ticks_total})</h3>",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("O media",   f"{resumen['O_media']:.1f}%")
        st.metric("O P5–P95",  f"{resumen['O_p5']:.0f}%–{resumen['O_p95']:.0f}%")
    with c2:
        st.metric("I media",   f"{resumen['I_media']:.1f}")
        st.metric("Nivel modal", resumen["nivel_I_modal"])
    with c3:
        st.metric("Traslados", resumen["traslados_total"])
        st.metric("Alertas RES-04", resumen["alertas_total"])
    with c4:
        rmse = resumen.get("pred_rmse")
        st.metric("RMSE pred.", f"{rmse:.3f} pp" if rmse else "—")
        st.metric("Altas totales", resumen["altas_total"])

    cumple_O = resumen["cumple_O_rango"]
    cumple_I = resumen["cumple_I_nivel"]
    st.markdown(
        f"<div style='display:flex;gap:16px;margin-top:6px;'>"
        f"<span style='color:{'#4CAF50' if cumple_O else '#F44336'};font-size:0.8rem;'>"
        f"{'✅' if cumple_O else '❌'} CE-B rango O</span>"
        f"<span style='color:{'#4CAF50' if cumple_I else '#F44336'};font-size:0.8rem;'>"
        f"{'✅' if cumple_I else '❌'} CE-B nivel I</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    fig = _fig_evolucion(sim.historial, mostrar_todo=True)
    if fig:
        st.markdown("<br>", unsafe_allow_html=True)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

def pagina_resultados():
    sim = st.session_state.get("sim")
    if not sim:
        st.session_state["pagina"] = "config"; st.rerun()
    st.markdown(
        "<h2 style='color:#e0e0e0;font-size:1.1rem;font-weight:700;margin-bottom:4px;'>"
        "📋 Resultados de la simulación</h2>",
        unsafe_allow_html=True,
    )
    if st.button("⟳ Nueva simulación"):
        st.session_state["pagina"] = "config"
        st.session_state["sim"]    = None
        st.rerun()
    pagina_resumen_inline(sim)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
def main():
    modelo = cargar_modelo()
    pagina = st.session_state["pagina"]
    if pagina == "config":
        pagina_config()
    elif pagina == "simular":
        pagina_simular(modelo)
    elif pagina == "resultados":
        pagina_resultados()

if __name__ == "__main__":
    main()
