"""
motor_simulacion.py
===================
Simulador Inteligente de Ocupación Hospitalaria
Fase 6 — Despliegue (CRISP-DM/S)

Autor  : Juan Camilo García Braham
Curso  : IA en Salud · Maestría IA y CD · UTP
Año    : 2026
Stack  : Python 3.12 · numpy 2.4 · streamlit 1.45.1

Decisiones de diseño activas — F6
───────────────────────────────────
D_F6_001  Stack aprobado: streamlit 1.45.1 sobre el stack base de F1–F5.
D_F6_002  Motor tick-a-tick: estado vive en st.session_state["sim"].
          Una llamada a avanzar_tick() procesa exactamente un tick.
D_F6_003  Representación del hospital como grid HTML con panel JS.
D_F6_006  Animación: st.rerun() completo por tick.
D_F6_007  Info paciente: panel lateral JS dentro del grid HTML.
D_F6_008  Pantalla de configuración separada antes del simulador.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np

from generador_pacientes import generar_llegadas_tick, LAMBDA_POR_ESCENARIO
from sistema_experto import (
    SistemaExperto, Hospital, ResultadoTick,
    calcular_indicador_I, Piso, Area, Cama,
)

# ── Constantes heredadas de F5 ────────────────────────────────────────────────
TICKS_TOTAL:    int = 200
WARM_UP_TICKS:  int = 20
HORIZONTE_PRED: int = 4
N_LAGS:         int = 4
SEED_DEFAULT:   int = 99

COLOR_NIVEL: dict[str, str] = {
    "Bajo":    "#4CAF50",
    "Medio":   "#FF9800",
    "Alto":    "#FF5722",
    "Crítico": "#F44336",
}
COLOR_CAMA: dict[str, str] = {
    "libre":       "#4CAF50",
    "ocupada":     "#F44336",
    "en_limpieza": "#FF9800",
    "temporal":    "#BB86FC",
}
UMBRALES: dict[str, dict] = {
    "normal":       {"O_min": 35.0, "O_max": 55.0,  "I_niveles": ["Bajo", "Medio"]},
    "alta_demanda": {"O_min": 60.0, "O_max": 88.0,  "I_niveles": ["Medio", "Alto", "Crítico"]},
    "crisis":       {"O_min": 80.0, "O_max": 100.0, "I_niveles": ["Alto", "Crítico"]},
}

# ── Dataclasses ───────────────────────────────────────────────────────────────
@dataclass
class RegistroTick:
    tick: int
    O_t: float; E_t: float; P_t: float; C_t: float; I_t: float
    nivel_I: str
    llegadas_t: int; altas_t: int; traslados_t: int; cola_t: int; n_alertas: int
    pred_O_t4: float | None = None

@dataclass
class ConfigHospital:
    """Capacidades configurables por el usuario (D_F6_008)."""
    uci:          int = 10
    urgencias:    int = 20
    hospitalizacion: int = 40
    observacion:  int = 15
    sala_espera:  int = 10

@dataclass
class EstadoSimulacion:
    escenario: str; modo_asistido: bool; seed: int
    ticks_total: int = TICKS_TOTAL
    hospital:    Hospital | None             = field(default=None, repr=False)
    se:          SistemaExperto | None       = field(default=None, repr=False)
    rng_sim:     np.random.Generator | None  = field(default=None, repr=False)
    rng_gen:     np.random.Generator | None  = field(default=None, repr=False)
    tick_actual: int  = 0
    activa:      bool = False
    pausada:     bool = False
    finalizada:  bool = False
    historial:   list[RegistroTick]  = field(default_factory=list)
    historial_O: list[float]         = field(default_factory=list)
    acciones_pendientes: list[Any]   = field(default_factory=list)
    esc_alta:    float = 0.0
    esc_crisis:  float = 0.0
    config:      ConfigHospital      = field(default_factory=ConfigHospital)

# ── Fábrica parametrizable ────────────────────────────────────────────────────
def _crear_hospital_custom(cfg: ConfigHospital,
                           rng: np.random.Generator | None = None) -> Hospital:
    """Construye el hospital con capacidades definidas por el usuario.

    Mantiene la topología de dos pisos y las reglas de prioridad de D008/F2.
    Solo varía el número de camas por área.
    """
    from sistema_experto import Hospital
    h = Hospital(rng=rng)

    p1 = h.agregar_piso("Piso 1 — Urgencias y UCI")
    p2 = h.agregar_piso("Piso 2 — Hospitalización y Observación")

    area_uci  = h.agregar_area("UCI",             p1.id_piso, cfg.uci,          acepta_desborde=False)
    area_urg  = h.agregar_area("Urgencias",        p1.id_piso, cfg.urgencias,    acepta_desborde=False)
    area_hosp = h.agregar_area("Hospitalización",  p2.id_piso, cfg.hospitalizacion, acepta_desborde=False)
    area_obs  = h.agregar_area("Observación",      p2.id_piso, cfg.observacion,  acepta_desborde=True)
    area_sala = h.agregar_area("Sala_de_espera",   p1.id_piso, cfg.sala_espera,  acepta_desborde=True)

    for _ in range(cfg.uci):           h.agregar_cama(area_uci.id_area,  "UCI")
    for _ in range(cfg.urgencias):     h.agregar_cama(area_urg.id_area,  "normal")
    for _ in range(cfg.hospitalizacion): h.agregar_cama(area_hosp.id_area, "normal")
    for _ in range(cfg.observacion):   h.agregar_cama(area_obs.id_area,  "observacion")
    for _ in range(cfg.sala_espera):   h.agregar_cama(area_sala.id_area, "normal")

    return h

# ── API pública ───────────────────────────────────────────────────────────────
def crear_estado(escenario: str = "normal", modo_asistido: bool = False,
                 seed: int = SEED_DEFAULT, ticks_total: int = TICKS_TOTAL,
                 config: ConfigHospital | None = None) -> EstadoSimulacion:
    if escenario not in LAMBDA_POR_ESCENARIO:
        raise ValueError(f"Escenario '{escenario}' no válido.")
    cfg = config or ConfigHospital()
    rng_sim = np.random.default_rng(seed)
    rng_gen = np.random.default_rng(seed + 500)
    hospital = _crear_hospital_custom(cfg, rng=rng_sim)
    se = SistemaExperto(hospital=hospital)
    return EstadoSimulacion(
        escenario=escenario, modo_asistido=modo_asistido,
        seed=seed, ticks_total=ticks_total,
        hospital=hospital, se=se, rng_sim=rng_sim, rng_gen=rng_gen,
        activa=True,
        esc_alta=1.0   if escenario == "alta_demanda" else 0.0,
        esc_crisis=1.0 if escenario == "crisis"       else 0.0,
        config=cfg,
    )

def avanzar_tick(estado: EstadoSimulacion, modelo_pred
                 ) -> tuple[ResultadoTick | None, list[Any]]:
    if not estado.activa or estado.pausada or estado.finalizada:
        return None, []
    tick = estado.tick_actual
    if tick >= estado.ticks_total:
        estado.finalizada = True; estado.activa = False
        return None, []

    nuevos = generar_llegadas_tick(tick, escenario=estado.escenario, rng=estado.rng_gen)
    resultado: ResultadoTick = estado.se.procesar_tick(
        tick, nuevos_pacientes=nuevos, modo_asistido=estado.modo_asistido)

    altas_t = sum(1 for a in resultado.acciones_ejecutadas if a.tipo == "alta")
    traslados_t = sum(1 for a in resultado.acciones_ejecutadas
                      if a.tipo in ("traslado", "desborde"))

    acciones_pendientes: list[Any] = []
    if estado.modo_asistido and resultado.acciones_pendientes:
        acciones_pendientes = resultado.acciones_pendientes
        estado.acciones_pendientes = acciones_pendientes

    cola_t = len(estado.hospital.pacientes_esperando())
    pred_O_t4: float | None = None
    estado.historial_O.append(resultado.componente_O)

    if len(estado.historial_O) > N_LAGS and modelo_pred is not None:
        fv = np.array([[
            resultado.componente_O, resultado.componente_E,
            resultado.componente_P, resultado.componente_C,
            resultado.indicador_I, float(len(nuevos)),
            float(altas_t), float(traslados_t), float(cola_t),
            estado.historial_O[-2], estado.historial_O[-3],
            estado.historial_O[-4], estado.historial_O[-5],
            estado.esc_alta, estado.esc_crisis,
        ]])
        pred_O_t4 = float(modelo_pred.predict(fv)[0])

    estado.historial.append(RegistroTick(
        tick=tick, O_t=resultado.componente_O, E_t=resultado.componente_E,
        P_t=resultado.componente_P, C_t=resultado.componente_C,
        I_t=resultado.indicador_I, nivel_I=resultado.nivel_I,
        llegadas_t=len(nuevos), altas_t=altas_t,
        traslados_t=traslados_t, cola_t=cola_t,
        n_alertas=len(resultado.alertas), pred_O_t4=pred_O_t4,
    ))

    estado.tick_actual += 1
    if estado.tick_actual >= estado.ticks_total:
        estado.finalizada = True; estado.activa = False

    return resultado, acciones_pendientes

def confirmar_acciones_pendientes(estado: EstadoSimulacion,
                                   indices: list[int]) -> int:
    if not estado.acciones_pendientes:
        return 0
    seleccion = [estado.acciones_pendientes[i]
                 for i in indices if 0 <= i < len(estado.acciones_pendientes)]
    if not seleccion:
        estado.acciones_pendientes = []
        return 0
    tick = max(0, estado.tick_actual - 1)
    ejecutadas = estado.se.confirmar_acciones(seleccion, tick)
    estado.acciones_pendientes = []
    return len(ejecutadas)

def calcular_resumen(estado: EstadoSimulacion) -> dict:
    regime = [r for r in estado.historial if r.tick >= WARM_UP_TICKS]
    if not regime:
        return {}
    from collections import Counter
    O_vals = np.array([r.O_t for r in regime])
    I_vals = np.array([r.I_t for r in regime])
    nivel_counts = Counter(r.nivel_I for r in regime)
    pairs = [(regime[i].pred_O_t4, regime[i + HORIZONTE_PRED].O_t)
             for i in range(len(regime) - HORIZONTE_PRED)
             if regime[i].pred_O_t4 is not None]
    pred_rmse = None
    if pairs:
        ps, rs = zip(*pairs)
        pred_rmse = float(np.sqrt(np.mean((np.array(ps) - np.array(rs)) ** 2)))
    umb = UMBRALES[estado.escenario]
    O_media = float(np.mean(O_vals))
    nivel_modal = nivel_counts.most_common(1)[0][0]
    nivel_norm = nivel_modal  # sin normalizacion: UMBRALES y SE usan tildes consistentemente
    return {
        "O_media": O_media, "O_std": float(np.std(O_vals)),
        "O_p5": float(np.percentile(O_vals, 5)),
        "O_p95": float(np.percentile(O_vals, 95)),
        "I_media": float(np.mean(I_vals)),
        "nivel_I_modal": nivel_modal,
        "cola_max": int(max(r.cola_t for r in regime)),
        "traslados_total": sum(r.traslados_t for r in regime),
        "alertas_total": sum(r.n_alertas for r in regime),
        "altas_total": sum(r.altas_t for r in regime),
        "pred_rmse": pred_rmse,
        "ticks_regime": len(regime),
        "cumple_O_rango": (O_media >= umb["O_min"] - 5.0 and
                           O_media <= umb["O_max"] + 5.0),
        "cumple_I_nivel": nivel_norm in umb["I_niveles"],
        "distribucion_niveles": dict(nivel_counts),
    }

def snapshot_hospital(estado: EstadoSimulacion) -> dict:
    h = estado.hospital
    if h is None:
        return {}
    indicador = calcular_indicador_I(h)
    areas_snap = []
    for piso in h.pisos.values():
        for area_id in piso.areas:
            area = h.areas[area_id]
            camas_snap = []
            for cid in area.todas_las_camas:
                cama = h.camas.get(cid)
                if cama is None:
                    continue
                pac_snap = None
                if cama.paciente_id:
                    p = h.pacientes.get(cama.paciente_id)
                    if p:
                        pac_snap = {
                            "id":         p.id_paciente[:8],
                            "prioridad":  p.prioridad_clinica,
                            "area_req":   p.area_requerida,
                            "espera_h":   round(p.tiempo_espera, 2),
                            "estancia_h": round(p.tiempo_en_sistema, 2),
                            "edad":       getattr(p, "edad", None),
                        }
                camas_snap.append({
                    "id":       cama.id_cama[:6],
                    "estado":   "temporal" if cama.es_temporal else cama.estado,
                    "paciente": pac_snap,
                })
            areas_snap.append({
                "nombre": area.nombre, "piso": piso.nombre,
                "capacidad_total": area.capacidad_total,
                "cap_disponible":  area.capacidad_disponible,
                "camas": camas_snap,
            })
    cola = sorted(h.pacientes_esperando(),
                  key=lambda p: (p.prioridad_clinica, -p.tiempo_espera))
    cola_snap = [
        {"id": p.id_paciente[:8], "prioridad": p.prioridad_clinica,
         "area_req": p.area_requerida, "espera_h": round(p.tiempo_espera, 2),
         "ticks_espera": estado.tick_actual - p.tick_ingreso}
        for p in cola
    ]
    return {"areas": areas_snap, "cola_espera": cola_snap, "indicador": indicador}
