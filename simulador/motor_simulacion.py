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
D_F6_002  Motor tick-a-tick: estado de la simulación vive en st.session_state
          (dict EstadoSimulacion). Una llamada a avanzar_tick() procesa
          exactamente un tick y devuelve ResultadoTick. El loop de velocidad
          se gestiona en app.py, no aquí.
D_F6_003  Representación del hospital como grid HTML (app.py).

Relación con F5
───────────────
Este módulo extrae la lógica de ejecutar_simulacion_eval() de evaluar_escenarios_f5.py
y la convierte en una máquina de estados reanudable. La lógica interna del tick
(procesar_tick, confirmar_acciones, generar_llegadas_tick) no se modifica — solo
se separa el estado de la simulación del loop de 200 ticks.

API pública
───────────
    estado = crear_estado(escenario, modo_asistido, seed)
    resultado, estado = avanzar_tick(estado, modelo_pred)
    resumen = calcular_resumen(estado)
    snapshot = snapshot_hospital(estado)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from generador_pacientes import generar_llegadas_tick, LAMBDA_POR_ESCENARIO
from sistema_experto import (
    SistemaExperto,
    Hospital,
    crear_hospital_referencia,
    ResultadoTick,
    calcular_indicador_I,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES — heredadas de F5
# ─────────────────────────────────────────────────────────────────────────────

TICKS_TOTAL:    int   = 200    # D_F5_001 — 50 horas simuladas
WARM_UP_TICKS:  int   = 20     # D_F5_006 — ticks descartados del análisis
HORIZONTE_PRED: int   = 4      # D_F4B_002 — T+4 (1 hora)
N_LAGS:         int   = 4      # D_F4B_005
SEED_DEFAULT:   int   = 99     # D_F5_003

# Orden fijo de features para el modelo — D_F5_007
FEATURES_ORDER: list[str] = [
    "O_t", "E_t", "P_t", "C_t", "I_t",
    "llegadas_t", "altas_t", "traslados_t", "cola_espera_t",
    "O_lag1", "O_lag2", "O_lag3", "O_lag4",
    "escenario_alta_demanda", "escenario_crisis",
]

# Umbrales CE-B ajustados — D_F5_008
UMBRALES: dict[str, dict] = {
    "normal":       {"O_min": 35.0, "O_max": 55.0,  "I_niveles": ["Bajo", "Medio"]},
    "alta_demanda": {"O_min": 60.0, "O_max": 88.0,  "I_niveles": ["Medio", "Alto", "Crítico"]},
    "crisis":       {"O_min": 80.0, "O_max": 100.0, "I_niveles": ["Alto", "Crítico"]},
}

# Colores semánticos por nivel I — usados por app.py
COLOR_NIVEL: dict[str, str] = {
    "Bajo":    "#4CAF50",
    "Medio":   "#FF9800",
    "Alto":    "#FF5722",
    "Crítico": "#F44336",
}

# Colores semánticos por estado de cama — usados por app.py (grid HTML)
COLOR_CAMA: dict[str, str] = {
    "libre":       "#4CAF50",   # verde
    "ocupada":     "#F44336",   # rojo
    "en_limpieza": "#FF9800",   # naranja
    "temporal":    "#BB86FC",   # acento violeta del proyecto
}

# ─────────────────────────────────────────────────────────────────────────────
# ESTADO DE SIMULACIÓN — D_F6_002
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RegistroTick:
    """Una fila del historial tick-a-tick (equivalente al de F5)."""
    tick:        int
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
    pred_O_t4:   float | None = None


@dataclass
class EstadoSimulacion:
    """Estado completo y reanudable de una simulación en curso.

    Vive en st.session_state["sim"]. avanzar_tick() muta este objeto
    y devuelve el ResultadoTick del tick procesado.
    """
    # Configuración (inmutable durante la simulación)
    escenario:      str
    modo_asistido:  bool
    seed:           int
    ticks_total:    int = TICKS_TOTAL

    # Objetos de simulación
    hospital:       Hospital | None                = field(default=None, repr=False)
    se:             SistemaExperto | None          = field(default=None, repr=False)
    rng_sim:        np.random.Generator | None     = field(default=None, repr=False)
    rng_gen:        np.random.Generator | None     = field(default=None, repr=False)

    # Cursor temporal
    tick_actual:    int = 0
    activa:         bool = False
    pausada:        bool = False
    finalizada:     bool = False

    # Historial y lags para el modelo predictivo
    historial:      list[RegistroTick]             = field(default_factory=list)
    historial_O:    list[float]                    = field(default_factory=list)

    # Acciones pendientes de confirmación (modo asistido)
    acciones_pendientes: list[Any]                 = field(default_factory=list)

    # One-hot del escenario (precalculado)
    esc_alta:       float = 0.0
    esc_crisis:     float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# API PÚBLICA
# ─────────────────────────────────────────────────────────────────────────────

def crear_estado(
    escenario: str = "normal",
    modo_asistido: bool = False,
    seed: int = SEED_DEFAULT,
    ticks_total: int = TICKS_TOTAL,
) -> EstadoSimulacion:
    """Inicializa un EstadoSimulacion listo para avanzar_tick().

    Construye el hospital de referencia (D008) y los generadores de
    aleatoriedad con la semilla indicada. No ejecuta ningún tick.
    """
    if escenario not in LAMBDA_POR_ESCENARIO:
        raise ValueError(f"Escenario '{escenario}' no válido.")

    rng_sim = np.random.default_rng(seed)
    rng_gen = np.random.default_rng(seed + 500)

    hospital = crear_hospital_referencia(rng=rng_sim)
    se       = SistemaExperto(hospital=hospital)

    estado = EstadoSimulacion(
        escenario     = escenario,
        modo_asistido = modo_asistido,
        seed          = seed,
        ticks_total   = ticks_total,
        hospital      = hospital,
        se            = se,
        rng_sim       = rng_sim,
        rng_gen       = rng_gen,
        activa        = True,
        esc_alta      = 1.0 if escenario == "alta_demanda" else 0.0,
        esc_crisis    = 1.0 if escenario == "crisis"       else 0.0,
    )
    return estado


def avanzar_tick(
    estado: EstadoSimulacion,
    modelo_pred,
) -> tuple[ResultadoTick | None, list[Any]]:
    """Procesa exactamente un tick y actualiza el estado en su lugar.

    Retorna (resultado_tick, acciones_pendientes).
    - resultado_tick: None si la simulación ya finalizó o está pausada.
    - acciones_pendientes: lista no vacía solo en modo asistido cuando
      hay propuestas esperando confirmación del gestor.

    D_F6_002: este método no contiene loops. El loop de velocidad
    (1x–10x) vive en app.py con st.empty() + time.sleep().
    """
    if not estado.activa or estado.pausada or estado.finalizada:
        return None, []

    tick = estado.tick_actual
    if tick >= estado.ticks_total:
        estado.finalizada = True
        estado.activa     = False
        return None, []

    # Generar llegadas del tick
    nuevos = generar_llegadas_tick(
        tick, escenario=estado.escenario, rng=estado.rng_gen
    )

    # Procesar tick en el sistema experto
    resultado: ResultadoTick = estado.se.procesar_tick(
        tick,
        nuevos_pacientes=nuevos,
        modo_asistido=estado.modo_asistido,
    )

    # Contadores de acciones ejecutadas
    altas_t = sum(1 for a in resultado.acciones_ejecutadas if a.tipo == "alta")
    traslados_t = sum(
        1 for a in resultado.acciones_ejecutadas
        if a.tipo in ("traslado", "desborde")
    )

    # Modo asistido — acumular pendientes; NO confirmar aquí
    # La confirmación la hace app.py al recibir la selección del gestor
    acciones_pendientes: list[Any] = []
    if estado.modo_asistido and resultado.acciones_pendientes:
        acciones_pendientes = resultado.acciones_pendientes
        estado.acciones_pendientes = acciones_pendientes

    cola_t = len(estado.hospital.pacientes_esperando())

    # Predicción del modelo (D_F5_004 — primeros N_LAGS ticks sin predicción)
    pred_O_t4: float | None = None
    estado.historial_O.append(resultado.componente_O)

    if len(estado.historial_O) > N_LAGS and modelo_pred is not None:
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
            estado.historial_O[-2],   # O_lag1
            estado.historial_O[-3],   # O_lag2
            estado.historial_O[-4],   # O_lag3
            estado.historial_O[-5],   # O_lag4
            estado.esc_alta,
            estado.esc_crisis,
        ]])
        pred_O_t4 = float(modelo_pred.predict(features_vec)[0])

    # Registrar en historial
    estado.historial.append(RegistroTick(
        tick        = tick,
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

    estado.tick_actual += 1

    if estado.tick_actual >= estado.ticks_total:
        estado.finalizada = True
        estado.activa     = False

    return resultado, acciones_pendientes


def confirmar_acciones_pendientes(
    estado: EstadoSimulacion,
    indices_seleccionados: list[int],
) -> int:
    """Confirma las acciones pendientes seleccionadas por el gestor.

    Parámetros
    ----------
    estado                : EstadoSimulacion con acciones_pendientes no vacías
    indices_seleccionados : índices (0-based) dentro de acciones_pendientes
                            que el gestor aprueba

    Retorna el número de acciones efectivamente ejecutadas.
    """
    if not estado.acciones_pendientes:
        return 0

    seleccion = [
        estado.acciones_pendientes[i]
        for i in indices_seleccionados
        if 0 <= i < len(estado.acciones_pendientes)
    ]
    if not seleccion:
        estado.acciones_pendientes = []
        return 0

    tick = max(0, estado.tick_actual - 1)   # tick que acaba de procesarse
    ejecutadas = estado.se.confirmar_acciones(seleccion, tick)
    estado.acciones_pendientes = []
    return len(ejecutadas)


def calcular_resumen(estado: EstadoSimulacion) -> dict:
    """Calcula métricas de régimen estable (equivalente a ResultadoEscenario de F5).

    Solo incluye ticks >= WARM_UP_TICKS. Devuelve dict con las métricas
    clave para el panel de resumen de app.py.
    """
    regime = [r for r in estado.historial if r.tick >= WARM_UP_TICKS]
    if not regime:
        return {}

    import numpy as np
    from collections import Counter

    O_vals  = np.array([r.O_t for r in regime])
    I_vals  = np.array([r.I_t for r in regime])
    niveles = [r.nivel_I for r in regime]
    nivel_counts = Counter(niveles)

    # RMSE predicción vs real (alineado en T+4)
    pred_rmse: float | None = None
    pairs = [
        (regime[i].pred_O_t4, regime[i + HORIZONTE_PRED].O_t)
        for i in range(len(regime) - HORIZONTE_PRED)
        if regime[i].pred_O_t4 is not None
    ]
    if pairs:
        preds, reals = zip(*pairs)
        pred_rmse = float(
            np.sqrt(np.mean((np.array(preds) - np.array(reals)) ** 2))
        )

    umb = UMBRALES[estado.escenario]
    O_media = float(np.mean(O_vals))
    nivel_modal = nivel_counts.most_common(1)[0][0]
    nivel_norm  = nivel_modal.replace("\u00ed", "i").replace("\u00e9", "e")

    return {
        "O_media":        O_media,
        "O_std":          float(np.std(O_vals)),
        "O_p5":           float(np.percentile(O_vals, 5)),
        "O_p95":          float(np.percentile(O_vals, 95)),
        "I_media":        float(np.mean(I_vals)),
        "nivel_I_modal":  nivel_modal,
        "cola_max":       int(max(r.cola_t      for r in regime)),
        "traslados_total":sum(r.traslados_t      for r in regime),
        "alertas_total":  sum(r.n_alertas        for r in regime),
        "altas_total":    sum(r.altas_t          for r in regime),
        "pred_rmse":      pred_rmse,
        "ticks_regime":   len(regime),
        "cumple_O_rango": (O_media >= umb["O_min"] - 5.0 and
                           O_media <= umb["O_max"] + 5.0),
        "cumple_I_nivel": nivel_norm in umb["I_niveles"],
        "distribucion_niveles": dict(nivel_counts),
    }


def snapshot_hospital(estado: EstadoSimulacion) -> dict:
    """Extrae una vista serializable del estado actual del hospital.

    Devuelve un dict estructurado por área con la información necesaria
    para que app.py renderice el grid HTML sin acceder al Hospital directamente.

    Estructura:
        {
          "areas": [
            {
              "nombre": str,
              "piso":   str,
              "capacidad_total": int,
              "camas": [
                {
                  "id":      str (primeros 6 chars),
                  "estado":  "libre"|"ocupada"|"en_limpieza"|"temporal",
                  "paciente": {  # None si libre/en_limpieza
                    "id":        str,
                    "prioridad": str,
                    "area_req":  str,
                    "espera_h":  float,
                    "estancia_h":float,
                  } | None
                },
                ...
              ]
            },
            ...
          ],
          "cola_espera": [  # pacientes en estado=esperando
            {
              "id": str, "prioridad": str, "area_req": str,
              "espera_h": float, "ticks_espera": int,
            },
            ...
          ],
          "indicador": {
            "I": float, "nivel": str, "O": float,
            "E": float, "P": float, "C": float,
          }
        }
    """
    h = estado.hospital
    if h is None:
        return {}

    indicador = calcular_indicador_I(h)

    # Construir lista de áreas con sus camas
    areas_snap = []
    for piso in h.pisos.values():
        for area_id in piso.areas:
            area = h.areas[area_id]
            camas_snap = []
            for cid in area.todas_las_camas:
                cama = h.camas.get(cid)
                if cama is None:
                    continue
                paciente_snap = None
                if cama.paciente_id:
                    p = h.pacientes.get(cama.paciente_id)
                    if p:
                        paciente_snap = {
                            "id":         p.id_paciente[:8],
                            "prioridad":  p.prioridad_clinica,
                            "area_req":   p.area_requerida,
                            "espera_h":   round(p.tiempo_espera, 2),
                            "estancia_h": round(p.tiempo_en_sistema, 2),
                        }
                estado_cama = (
                    "temporal" if cama.es_temporal else cama.estado
                )
                camas_snap.append({
                    "id":       cama.id_cama[:6],
                    "estado":   estado_cama,
                    "paciente": paciente_snap,
                })

            areas_snap.append({
                "nombre":          area.nombre,
                "piso":            piso.nombre,
                "capacidad_total": area.capacidad_total,
                "cap_disponible":  area.capacidad_disponible,
                "camas":           camas_snap,
            })

    # Cola de espera (ordenada por prioridad y tiempo de espera — igual que SE)
    cola = sorted(
        h.pacientes_esperando(),
        key=lambda p: (p.prioridad_clinica, -p.tiempo_espera),
    )
    cola_snap = [
        {
            "id":          p.id_paciente[:8],
            "prioridad":   p.prioridad_clinica,
            "area_req":    p.area_requerida,
            "espera_h":    round(p.tiempo_espera, 2),
            "ticks_espera": estado.tick_actual - p.tick_ingreso,
        }
        for p in cola
    ]

    return {
        "areas":       areas_snap,
        "cola_espera": cola_snap,
        "indicador":   indicador,
    }
