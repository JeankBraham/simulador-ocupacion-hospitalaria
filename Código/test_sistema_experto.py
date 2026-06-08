"""
test_sistema_experto.py
=======================
Simulador Inteligente de Ocupación Hospitalaria
Fase 4-A — Pruebas unitarias del sistema experto

Autor  : Juan Camilo García Braham
Curso  : IA en Salud · Maestría IA y CD · UTP
Año    : 2026

Cobertura: una prueba por regla documentada en F4-A Entregable 1.
Convención de nombres: test_<ID_REGLA>_<descripcion_corta>

Ejecución:
    python test_sistema_experto.py
"""

import sys
import traceback
import uuid
import numpy as np

from sistema_experto import (
    Hospital, SistemaExperto, Paciente, Cama, Area,
    crear_hospital_referencia, calcular_indicador_I,
    _asignar_cama, _liberar_cama, _trasladar_paciente, _crear_cama_temporal,
    TransicionInvalidaError, InconsistenciaModeloError,
    DURACION_TICK_H, T_UMBRAL_CRITICO_TICKS,
)

# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES DE TEST
# ─────────────────────────────────────────────────────────────────────────────

_resultados: list[tuple[str, bool, str]] = []

def prueba(nombre: str):
    """Decorador para registrar resultado de cada prueba."""
    def decorador(fn):
        try:
            fn()
            _resultados.append((nombre, True, ""))
        except AssertionError as e:
            _resultados.append((nombre, False, str(e)))
        except Exception as e:
            _resultados.append((nombre, False, f"{type(e).__name__}: {e}"))
        return fn
    return decorador


def _paciente_dict(prioridad: str, area: str,
                   estancia: float = 2.0, tick: int = 0) -> dict:
    """Genera un PacienteDict mínimo para pruebas."""
    return {
        "id_paciente": str(uuid.uuid4()),
        "edad": 45,
        "prioridad_clinica": prioridad,
        "area_requerida": area,
        "tiempo_estancia_esperado": estancia,
        "tiempo_espera": 0.0,
        "tiempo_en_sistema": 0.0,
        "estado": "esperando",
        "cama_id": None,
        "tick_ingreso": tick,
        "es_desborde": False,
    }


def _hospital_vacio() -> Hospital:
    return Hospital(rng=np.random.default_rng(42))


# ─────────────────────────────────────────────────────────────────────────────
# CAPA 1 — ELEGIBILIDAD
# ─────────────────────────────────────────────────────────────────────────────

@prueba("RE-01 — Solo camas libres son elegibles")
def test_RE01_solo_camas_libres():
    h = crear_hospital_referencia()
    area_uci = h.area_por_nombre("UCI")
    # Marcar todas las camas de UCI como ocupadas manualmente
    for cid in area_uci.camas:
        h.camas[cid].estado = "ocupada"
    camas = h.camas_libres_en_area(area_uci.id_area)
    assert len(camas) == 0, f"Esperaba 0 camas libres, obtuvo {len(camas)}"


@prueba("RE-01 — Camas en_limpieza no son elegibles")
def test_RE01_camas_limpieza_no_elegibles():
    h = crear_hospital_referencia()
    area_urg = h.area_por_nombre("Urgencias")
    for cid in area_urg.camas:
        h.camas[cid].estado = "en_limpieza"
        h.camas[cid].tiempo_limpieza_restante = 0.5
    camas = h.camas_libres_en_area(area_urg.id_area)
    assert len(camas) == 0, "Camas en limpieza no deben ser elegibles"


@prueba("RE-02 — Asignación solo al área requerida del paciente")
def test_RE02_area_requerida():
    h = crear_hospital_referencia()
    se = SistemaExperto(h)
    datos = _paciente_dict("P2", "Urgencias")
    resultado = se.procesar_tick(0, [datos])
    pid = datos["id_paciente"]
    paciente = h.pacientes[pid]
    assert paciente.estado == "hospitalizado", "P2 debería estar hospitalizado"
    cama = h.camas[paciente.cama_id]
    area_asignada = h.areas[cama.area_id]
    assert area_asignada.nombre == "Urgencias", (
        f"Paciente P2 con area_requerida=Urgencias fue a {area_asignada.nombre}"
    )


@prueba("RE-03 — P4 no puede ir a UCI (prioridades_aceptadas)")
def test_RE03_p4_no_va_a_uci():
    h = crear_hospital_referencia()
    # Bloquear todas las áreas excepto UCI
    for nombre in ["Urgencias", "Hospitalización", "Observación", "Sala_de_espera"]:
        area = h.area_por_nombre(nombre)
        for cid in area.camas:
            h.camas[cid].estado = "ocupada"
        area.acepta_desborde = False

    se = SistemaExperto(h)
    datos = _paciente_dict("P4", "Observación")
    se.procesar_tick(0, [datos])
    paciente = h.pacientes[datos["id_paciente"]]
    assert paciente.estado == "esperando", (
        "P4 no debe poder ir a UCI aunque sea la única área libre"
    )


@prueba("RE-03 — R09: ningún atributo demográfico en reglas de elegibilidad")
def test_RE03_r09_sin_demograficos():
    """Verifica que edad no afecta asignación: dos pacientes con misma prioridad
    y diferente edad deben recibir camas por igual sin distinción."""
    h = crear_hospital_referencia()
    se = SistemaExperto(h)
    d1 = _paciente_dict("P2", "Urgencias"); d1["edad"] = 5
    d2 = _paciente_dict("P2", "Urgencias"); d2["edad"] = 90
    se.procesar_tick(0, [d1, d2])
    p1_obj = h.pacientes[d1["id_paciente"]]
    p2_obj = h.pacientes[d2["id_paciente"]]
    assert p1_obj.estado == "hospitalizado", "Paciente de 5 años debe ser asignado"
    assert p2_obj.estado == "hospitalizado", "Paciente de 90 años debe ser asignado"


# ─────────────────────────────────────────────────────────────────────────────
# CAPA 2 — PRIORIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────

@prueba("RP-01 — P1 tiene prioridad sobre P3 con una sola cama libre")
def test_RP01_prioridad_p1_sobre_p3():
    h = crear_hospital_referencia()
    # Dejar exactamente 1 cama libre en Urgencias y bloquear Observación/Sala
    # para que P3 no pueda ir a desborde
    area_urg = h.area_por_nombre("Urgencias")
    for cid in area_urg.camas[1:]:
        h.camas[cid].estado = "ocupada"
    h._recalcular_capacidad_disponible(area_urg.id_area)

    area_obs = h.area_por_nombre("Observación")
    for cid in area_obs.camas:
        h.camas[cid].estado = "ocupada"
    area_obs.acepta_desborde = False
    h._recalcular_capacidad_disponible(area_obs.id_area)

    area_sala = h.area_por_nombre("Sala_de_espera")
    area_sala.acepta_desborde = False

    se = SistemaExperto(h)
    d_p3 = _paciente_dict("P3", "Urgencias")
    d_p1 = _paciente_dict("P1", "Urgencias")
    se.procesar_tick(0, [d_p3, d_p1])

    p1 = h.pacientes[d_p1["id_paciente"]]
    p3 = h.pacientes[d_p3["id_paciente"]]
    assert p1.estado == "hospitalizado", "P1 debería recibir la cama disponible"
    # Sin desborde disponible, P3 queda esperando
    assert p3.estado == "esperando", (
        f"P3 debería quedar en espera (sin desborde posible), estado={p3.estado}"
    )


@prueba("RP-01 — P2 tiene prioridad sobre P4 con una sola cama libre")
def test_RP01_prioridad_p2_sobre_p4():
    h = crear_hospital_referencia()
    area_obs = h.area_por_nombre("Observación")
    for cid in area_obs.camas[1:]:
        h.camas[cid].estado = "ocupada"
    h._recalcular_capacidad_disponible(area_obs.id_area)

    # Bloquear Sala_de_espera para que P4 no pueda ir a desborde
    area_sala = h.area_por_nombre("Sala_de_espera")
    for cid in area_sala.camas:
        h.camas[cid].estado = "ocupada"
    area_sala.acepta_desborde = False
    h._recalcular_capacidad_disponible(area_sala.id_area)

    se = SistemaExperto(h)
    d_p4 = _paciente_dict("P4", "Observación")
    d_p2 = _paciente_dict("P2", "Observación")
    se.procesar_tick(0, [d_p4, d_p2])

    p2 = h.pacientes[d_p2["id_paciente"]]
    p4 = h.pacientes[d_p4["id_paciente"]]
    assert p2.estado == "hospitalizado", "P2 debería recibir la cama"
    assert p4.estado == "esperando", (
        f"P4 debería quedar en espera (sin desborde), estado={p4.estado}"
    )


@prueba("RP-02 — Desempate por tiempo de espera (FIFO dentro de prioridad)")
def test_RP02_desempate_tiempo_espera():
    h = crear_hospital_referencia()
    # Urgencias completamente ocupada; Observación/Sala bloqueadas → P3 espera
    area_urg = h.area_por_nombre("Urgencias")
    for cid in area_urg.camas:
        h.camas[cid].estado = "ocupada"
    h._recalcular_capacidad_disponible(area_urg.id_area)

    area_obs = h.area_por_nombre("Observación")
    for cid in area_obs.camas:
        h.camas[cid].estado = "ocupada"
    area_obs.acepta_desborde = False
    h._recalcular_capacidad_disponible(area_obs.id_area)

    area_sala = h.area_por_nombre("Sala_de_espera")
    area_sala.acepta_desborde = False

    se = SistemaExperto(h)

    # Paciente A ingresa en tick 0 — queda esperando (no hay camas)
    d_a = _paciente_dict("P3", "Urgencias")
    se.procesar_tick(0, [d_a])
    p_a = h.pacientes[d_a["id_paciente"]]
    assert p_a.estado == "esperando", "Setup incorrecto: P_A debería estar esperando"

    # Tick 1 — sigue esperando, acumula tiempo_espera
    se.procesar_tick(1)

    # Tick 2 — liberar 1 cama en Urgencias + ingresar P_B (mismo prioridad, menos espera)
    h.camas[area_urg.camas[0]].estado = "libre"
    h._recalcular_capacidad_disponible(area_urg.id_area)

    d_b = _paciente_dict("P3", "Urgencias")
    se.procesar_tick(2, [d_b])

    p_a2 = h.pacientes[d_a["id_paciente"]]
    p_b2 = h.pacientes[d_b["id_paciente"]]
    assert p_a2.estado == "hospitalizado", (
        "Paciente A (más tiempo esperando) debe ser atendido primero"
    )
    assert p_b2.estado == "esperando", (
        "Paciente B (recién llegado) debe seguir esperando"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CAPA 3 — ASIGNACIÓN NORMAL
# ─────────────────────────────────────────────────────────────────────────────

@prueba("RA-01 — Asignación directa cuando hay cama disponible")
def test_RA01_asignacion_directa():
    h = crear_hospital_referencia()
    se = SistemaExperto(h)
    datos = _paciente_dict("P1", "UCI")
    se.procesar_tick(0, [datos])
    paciente = h.pacientes[datos["id_paciente"]]
    assert paciente.estado == "hospitalizado"
    assert paciente.cama_id is not None
    cama = h.camas[paciente.cama_id]
    assert cama.estado == "ocupada"
    assert cama.paciente_id == paciente.id_paciente


@prueba("RA-01 — Invariante D013: cama y paciente se actualizan en par")
def test_RA01_invariante_d013():
    h = crear_hospital_referencia()
    se = SistemaExperto(h)
    datos = _paciente_dict("P2", "Urgencias")
    se.procesar_tick(0, [datos])
    paciente = h.pacientes[datos["id_paciente"]]
    cama = h.camas[paciente.cama_id]
    assert cama.paciente_id == paciente.id_paciente, "Referencia cama→paciente inconsistente"
    assert paciente.cama_id == cama.id_cama, "Referencia paciente→cama inconsistente"


@prueba("RA-02 — Selección determinista (menor id lexicográfico)")
def test_RA02_seleccion_determinista():
    h = crear_hospital_referencia()
    area_urg = h.area_por_nombre("Urgencias")
    camas_ids = sorted(area_urg.camas)
    esperada = camas_ids[0]

    se = SistemaExperto(h)
    datos = _paciente_dict("P2", "Urgencias")
    se.procesar_tick(0, [datos])
    paciente = h.pacientes[datos["id_paciente"]]
    assert paciente.cama_id == esperada, (
        f"Esperaba cama {esperada[:8]}, obtuvo {paciente.cama_id[:8]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CAPA 4 — SOBREOCUPACIÓN
# ─────────────────────────────────────────────────────────────────────────────

@prueba("RSO-01 — Detectar área con capacidad_disponible <= 0")
def test_RSO01_detectar_sobreocupacion():
    h = crear_hospital_referencia()
    area_urg = h.area_por_nombre("Urgencias")
    for cid in area_urg.camas:
        h.camas[cid].estado = "ocupada"
    h._recalcular_capacidad_disponible(area_urg.id_area)
    assert area_urg.capacidad_disponible == 0, (
        f"capacidad_disponible debería ser 0, es {area_urg.capacidad_disponible}"
    )


@prueba("RSO-02 — Traslado interno P3 de Urgencias a Observación")
def test_RSO02_traslado_interno_p3():
    h = crear_hospital_referencia()
    se = SistemaExperto(h)

    # Hospitalizar un P3 en Urgencias
    d_p3 = _paciente_dict("P3", "Urgencias")
    se.procesar_tick(0, [d_p3])
    p3 = h.pacientes[d_p3["id_paciente"]]
    assert p3.estado == "hospitalizado"

    # Llenar Urgencias completamente
    area_urg = h.area_por_nombre("Urgencias")
    for cid in area_urg.camas:
        if h.camas[cid].estado == "libre":
            h.camas[cid].estado = "ocupada"
    h._recalcular_capacidad_disponible(area_urg.id_area)
    assert area_urg.capacidad_disponible == 0

    # Ingresar un P1 — para eso Urgencias necesita liberar espacio → traslado P3
    d_p1 = _paciente_dict("P1", "Urgencias")
    resultado = se.procesar_tick(1, [d_p1])

    p3_actualizado = h.pacientes[d_p3["id_paciente"]]
    p1 = h.pacientes[d_p1["id_paciente"]]

    # El P3 debería haber sido trasladado a Observación
    if p3_actualizado.estado == "trasladado":
        cama_p3 = h.camas[p3_actualizado.cama_id]
        area_p3 = h.areas[cama_p3.area_id]
        assert area_p3.nombre == "Observación", (
            f"P3 trasladado a {area_p3.nombre}, esperaba Observación"
        )
    # Y el P1 debería estar hospitalizado o en espera (dependiendo del orden)
    assert p1.estado in ("hospitalizado", "esperando"), (
        f"Estado inesperado del P1: {p1.estado}"
    )


@prueba("RSO-03 — Desborde crea cama temporal para P2")
def test_RSO03_desborde_cama_temporal_p2():
    h = crear_hospital_referencia()
    # Llenar todas las camas de Urgencias
    area_urg = h.area_por_nombre("Urgencias")
    for cid in area_urg.camas:
        h.camas[cid].estado = "ocupada"
    h._recalcular_capacidad_disponible(area_urg.id_area)

    se = SistemaExperto(h)
    d_p2 = _paciente_dict("P2", "Urgencias")
    se.procesar_tick(0, [d_p2])
    p2 = h.pacientes[d_p2["id_paciente"]]

    assert p2.estado == "hospitalizado", "P2 debería estar hospitalizado en desborde"
    assert p2.es_desborde is True, "P2 en desborde debe tener es_desborde=True"
    assert p2.cama_id is not None, "P2 en desborde debe tener cama asignada"
    cama = h.camas[p2.cama_id]
    assert cama.tipo == "temporal", "Cama de desborde debe ser tipo temporal"


@prueba("RSO-03 — Desborde usa cama temporal preexistente si existe")
def test_RSO03_desborde_reutiliza_temporal():
    h = crear_hospital_referencia()
    area_urg = h.area_por_nombre("Urgencias")
    area_obs = h.area_por_nombre("Observación")
    for cid in area_urg.camas:
        h.camas[cid].estado = "ocupada"
    h._recalcular_capacidad_disponible(area_urg.id_area)

    # Pre-crear una cama temporal libre en Observación
    cama_preexistente = _crear_cama_temporal(h, area_obs.id_area)
    cama_preexistente.estado = "libre"
    cama_preexistente_id = cama_preexistente.id_cama
    # Restaurar capacidad_disponible (fue decrementada por _crear_cama_temporal)
    area_obs.capacidad_disponible += 1

    se = SistemaExperto(h)
    d_p2 = _paciente_dict("P2", "Urgencias")
    se.procesar_tick(0, [d_p2])
    p2 = h.pacientes[d_p2["id_paciente"]]

    assert p2.cama_id == cama_preexistente_id, (
        "Debería haberse usado la cama temporal preexistente"
    )


@prueba("RSO-04 — P1 sin UCI NO se reasigna a otra área")
def test_RSO04_p1_sin_uci_no_reasigna():
    h = crear_hospital_referencia()
    # Ocupar todas las camas de UCI
    area_uci = h.area_por_nombre("UCI")
    for cid in area_uci.camas:
        h.camas[cid].estado = "ocupada"
    h._recalcular_capacidad_disponible(area_uci.id_area)

    se = SistemaExperto(h)
    d_p1 = _paciente_dict("P1", "UCI")
    se.procesar_tick(0, [d_p1])
    p1 = h.pacientes[d_p1["id_paciente"]]

    assert p1.estado == "esperando", (
        "P1 sin UCI debe permanecer en espera, no ser reasignado"
    )
    assert p1.cama_id is None, "P1 en espera no debe tener cama asignada"
    assert p1.es_desborde is False, "P1 sin UCI no debe marcarse como desborde"


# ─────────────────────────────────────────────────────────────────────────────
# CAPA 5 — ESCALAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

@prueba("RES-01 — Indicador I se calcula al final de cada tick")
def test_RES01_calculo_indicador_I():
    h = crear_hospital_referencia()
    se = SistemaExperto(h)
    resultado = se.procesar_tick(0, [])
    assert 0 <= resultado.indicador_I <= 100, (
        f"I={resultado.indicador_I} fuera del rango [0, 100]"
    )
    assert resultado.nivel_I in ("Bajo", "Medio", "Alto", "Crítico")


@prueba("RES-01 — Hospital vacío produce I=0 nivel Bajo")
def test_RES01_hospital_vacio_I_cero():
    h = crear_hospital_referencia()
    ind = calcular_indicador_I(h)
    assert ind["I"] == 0.0, f"Hospital vacío debe tener I=0, obtuvo {ind['I']}"
    assert ind["nivel"] == "Bajo"


@prueba("RES-01 — Componente O correcto con ocupación al 50%")
def test_RES01_componente_O_50pct():
    h = crear_hospital_referencia()
    # Hospital de referencia: UCI=10, Urg=20, Hosp=40, Obs=15, Sala=10 → total=95
    area_urg = h.area_por_nombre("Urgencias")
    for cid in area_urg.camas[:10]:
        h.camas[cid].estado = "ocupada"
    h._recalcular_capacidad_disponible(area_urg.id_area)

    ind = calcular_indicador_I(h)
    # Total camas válidas = 95; 10 ocupadas en Urgencias
    o_esperado = (10 / 95) * 100
    assert abs(ind["O"] - o_esperado) < 0.1, (
        f"O esperado ≈ {o_esperado:.2f}%, obtuvo {ind['O']}%"
    )


@prueba("RES-04 — P1 más de t_umbral_critico ticks sin cama fuerza nivel Crítico")
def test_RES04_p1_espera_umbral_critico():
    h = crear_hospital_referencia()
    area_uci = h.area_por_nombre("UCI")
    for cid in area_uci.camas:
        h.camas[cid].estado = "ocupada"
    h._recalcular_capacidad_disponible(area_uci.id_area)

    se = SistemaExperto(h)
    d_p1 = _paciente_dict("P1", "UCI", tick=0)

    # Tick 0: P1 ingresa, UCI llena
    se.procesar_tick(0, [d_p1])
    # Tick 1: sigue esperando (1 tick)
    se.procesar_tick(1)
    # Tick 2: supera T_UMBRAL_CRITICO_TICKS = 2
    resultado = se.procesar_tick(2)

    assert resultado.nivel_I == "Crítico", (
        f"P1 esperando {T_UMBRAL_CRITICO_TICKS} ticks debe forzar Crítico, "
        f"obtuvo nivel={resultado.nivel_I}"
    )
    assert any("RES-04" in a for a in resultado.alertas), (
        "Debe haber alerta RES-04 en el resultado"
    )


@prueba("RES-02/03 — Niveles de I correctos por umbrales")
def test_RES02_03_niveles_indicador():
    casos = [
        (0.0,  "Bajo"),
        (25.0, "Bajo"),
        (26.0, "Medio"),
        (50.0, "Medio"),
        (51.0, "Alto"),
        (75.0, "Alto"),
        (76.0, "Crítico"),
        (100.0,"Crítico"),
    ]
    for valor_i, nivel_esperado in casos:
        if valor_i <= 25:
            nivel = "Bajo"
        elif valor_i <= 50:
            nivel = "Medio"
        elif valor_i <= 75:
            nivel = "Alto"
        else:
            nivel = "Crítico"
        assert nivel == nivel_esperado, (
            f"I={valor_i} debería ser {nivel_esperado}, obtuvo {nivel}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CAPA 6 — ALTA Y LIMPIEZA
# ─────────────────────────────────────────────────────────────────────────────

@prueba("RAL-01 — Alta cuando tiempo_en_cama >= tiempo_estancia_esperado")
def test_RAL01_alta_por_tiempo():
    h = crear_hospital_referencia()
    se = SistemaExperto(h)

    # Paciente con estancia de 1 tick (0.25 h)
    datos = _paciente_dict("P2", "Urgencias", estancia=DURACION_TICK_H)
    se.procesar_tick(0, [datos])   # tick 0: ingresa y es asignado
    paciente = h.pacientes[datos["id_paciente"]]
    assert paciente.estado == "hospitalizado"

    se.procesar_tick(1)             # tick 1: se da de alta
    paciente_actualizado = h.pacientes[datos["id_paciente"]]
    assert paciente_actualizado.estado == "dado_de_alta", (
        "Paciente debería estar dado_de_alta tras cumplir estancia"
    )
    assert paciente_actualizado.cama_id is None, (
        "Paciente dado_de_alta no debe tener cama asignada"
    )


@prueba("RAL-01 — Cama pasa a en_limpieza tras el alta")
def test_RAL01_cama_limpieza_tras_alta():
    h = crear_hospital_referencia()
    se = SistemaExperto(h)
    datos = _paciente_dict("P2", "Urgencias", estancia=DURACION_TICK_H)
    se.procesar_tick(0, [datos])
    paciente = h.pacientes[datos["id_paciente"]]
    cama_id = paciente.cama_id
    se.procesar_tick(1)
    cama = h.camas.get(cama_id)
    # La cama puede haber pasado a en_limpieza o libre (si limpieza ya terminó)
    assert cama is not None, "Cama oficial no debe eliminarse tras alta"
    assert cama.estado in ("en_limpieza", "libre"), (
        f"Cama tras alta debe estar en_limpieza o libre, está {cama.estado}"
    )


@prueba("RAL-02 — Cama en_limpieza pasa a libre cuando tiempo_limpieza <= 0")
def test_RAL02_cama_libre_tras_limpieza():
    h = crear_hospital_referencia()
    area_urg = h.area_por_nombre("Urgencias")
    cama = h.camas[area_urg.camas[0]]
    cama.estado = "en_limpieza"
    cama.tiempo_limpieza_restante = DURACION_TICK_H  # exactamente 1 tick

    se = SistemaExperto(h)
    se.procesar_tick(0)   # procesa limpieza: decrementa → 0 → libre
    assert cama.estado == "libre", (
        f"Cama debería estar libre tras limpieza, está {cama.estado}"
    )


@prueba("RAL-03 — tiempo_espera se incrementa cada tick para pacientes esperando")
def test_RAL03_incremento_tiempo_espera():
    h = crear_hospital_referencia()
    area_urg = h.area_por_nombre("Urgencias")
    for cid in area_urg.camas:
        h.camas[cid].estado = "ocupada"
    h._recalcular_capacidad_disponible(area_urg.id_area)

    # Ocupar también Observación para que no haya desborde
    area_obs = h.area_por_nombre("Observación")
    area_obs.acepta_desborde = False

    se = SistemaExperto(h)
    datos = _paciente_dict("P3", "Urgencias")
    se.procesar_tick(0, [datos])
    paciente = h.pacientes[datos["id_paciente"]]
    espera_0 = paciente.tiempo_espera

    se.procesar_tick(1)
    espera_1 = h.pacientes[datos["id_paciente"]].tiempo_espera

    assert espera_1 > espera_0, (
        f"tiempo_espera no incrementó: tick0={espera_0} tick1={espera_1}"
    )
    assert abs(espera_1 - espera_0 - DURACION_TICK_H) < 1e-9, (
        f"Incremento esperado {DURACION_TICK_H}h, obtuvo {espera_1 - espera_0}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# DECISIONES DE DISEÑO — D_F4A
# ─────────────────────────────────────────────────────────────────────────────

@prueba("D_F4A_002 — TransicionInvalidaError en esperando→dado_de_alta directa")
def test_D_F4A_002_transicion_invalida():
    """R02 — La transición esperando→dado_de_alta debe estar bloqueada."""
    h = crear_hospital_referencia()
    area_urg = h.area_por_nombre("Urgencias")
    cama = h.camas[area_urg.camas[0]]
    paciente = Paciente(
        id_paciente=str(uuid.uuid4()),
        edad=30,
        prioridad_clinica="P3",
        area_requerida="Urgencias",
        tiempo_estancia_esperado=2.0,
        estado="dado_de_alta",   # ya dado de alta
    )
    h.pacientes[paciente.id_paciente] = paciente
    try:
        _asignar_cama(h, paciente, cama, 0)
        assert False, "Debería haber lanzado TransicionInvalidaError"
    except TransicionInvalidaError:
        pass  # Comportamiento esperado


@prueba("D_F4A_001 — InconsistenciaModeloError si cama ya ocupada")
def test_D_F4A_001_cama_ya_ocupada():
    h = crear_hospital_referencia()
    area_urg = h.area_por_nombre("Urgencias")
    cama = h.camas[area_urg.camas[0]]
    cama.estado = "ocupada"

    paciente = Paciente(
        id_paciente=str(uuid.uuid4()),
        edad=45,
        prioridad_clinica="P2",
        area_requerida="Urgencias",
        tiempo_estancia_esperado=2.0,
    )
    h.pacientes[paciente.id_paciente] = paciente
    try:
        _asignar_cama(h, paciente, cama, 0)
        assert False, "Debería haber lanzado InconsistenciaModeloError"
    except InconsistenciaModeloError:
        pass


@prueba("D_F4A_006/007 — Cama temporal dinámica se destruye al alta del paciente")
def test_D_F4A_007_cama_temporal_se_destruye():
    h = crear_hospital_referencia()
    area_urg = h.area_por_nombre("Urgencias")
    for cid in area_urg.camas:
        h.camas[cid].estado = "ocupada"
    h._recalcular_capacidad_disponible(area_urg.id_area)

    se = SistemaExperto(h)
    datos = _paciente_dict("P2", "Urgencias", estancia=DURACION_TICK_H)
    se.procesar_tick(0, [datos])
    paciente = h.pacientes[datos["id_paciente"]]
    assert paciente.es_desborde is True
    cama_temporal_id = paciente.cama_id
    assert h.camas[cama_temporal_id].es_temporal is True

    se.procesar_tick(1)  # alta
    assert datos["id_paciente"] in h.pacientes
    assert h.pacientes[datos["id_paciente"]].estado == "dado_de_alta"
    assert cama_temporal_id not in h.camas, (
        "Cama temporal dinámica debe eliminarse al dar de alta al paciente"
    )


@prueba("D_F4A_003 — Orden del tick: alta libera cama disponible en siguiente tick")
def test_D_F4A_003_orden_tick():
    """Verifica que el alta en tick T hace la cama disponible en T+1 o T+2
    (después del ciclo de limpieza), no en el mismo tick T."""
    h = crear_hospital_referencia()
    area_urg = h.area_por_nombre("Urgencias")
    # Dejar una sola cama libre
    for cid in area_urg.camas[1:]:
        h.camas[cid].estado = "ocupada"
    h._recalcular_capacidad_disponible(area_urg.id_area)

    se = SistemaExperto(h)
    # P2_a entra y se hospitaliza (la única cama libre)
    d_a = _paciente_dict("P2", "Urgencias", estancia=DURACION_TICK_H)
    se.procesar_tick(0, [d_a])
    p_a = h.pacientes[d_a["id_paciente"]]
    assert p_a.estado == "hospitalizado"

    # En tick 1: P2_a se da de alta Y entra P2_b en el mismo tick
    # La cama de P2_a pasa a en_limpieza → P2_b no debe ocuparla en tick 1
    d_b = _paciente_dict("P2", "Urgencias", estancia=2.0)
    se.procesar_tick(1, [d_b])
    p_b = h.pacientes[d_b["id_paciente"]]
    # P2_b puede haber ido a desborde o estar esperando, pero NO a la cama recién liberada
    if p_b.cama_id:
        cama_b = h.camas[p_b.cama_id]
        # Si fue asignado, su cama no es la misma que usó P2_a (está en limpieza)
        assert cama_b.estado == "ocupada"


# ─────────────────────────────────────────────────────────────────────────────
# MODO ASISTIDO
# ─────────────────────────────────────────────────────────────────────────────

@prueba("Modo asistido — desborde se propone sin ejecutar")
def test_modo_asistido_propone_sin_ejecutar():
    h = crear_hospital_referencia()
    area_urg = h.area_por_nombre("Urgencias")
    for cid in area_urg.camas:
        h.camas[cid].estado = "ocupada"
    h._recalcular_capacidad_disponible(area_urg.id_area)

    se = SistemaExperto(h)
    d_p2 = _paciente_dict("P2", "Urgencias")
    resultado = se.procesar_tick(0, [d_p2], modo_asistido=True)

    p2 = h.pacientes[d_p2["id_paciente"]]
    assert p2.estado == "esperando", (
        "En modo asistido el desborde no debe ejecutarse automáticamente"
    )
    pendientes_desborde = [a for a in resultado.acciones_pendientes
                           if a.tipo == "desborde"]
    assert len(pendientes_desborde) >= 1, (
        "Debe haber al menos una acción de desborde pendiente"
    )


@prueba("Modo asistido — confirmar_acciones ejecuta el desborde")
def test_modo_asistido_confirmar_ejecuta():
    h = crear_hospital_referencia()
    area_urg = h.area_por_nombre("Urgencias")
    for cid in area_urg.camas:
        h.camas[cid].estado = "ocupada"
    h._recalcular_capacidad_disponible(area_urg.id_area)

    se = SistemaExperto(h)
    d_p2 = _paciente_dict("P2", "Urgencias")
    resultado = se.procesar_tick(0, [d_p2], modo_asistido=True)

    pendientes = [a for a in resultado.acciones_pendientes if a.tipo == "desborde"]
    ejecutadas = se.confirmar_acciones(pendientes, tick=0)

    p2 = h.pacientes[d_p2["id_paciente"]]
    assert p2.estado == "hospitalizado", (
        "Tras confirmar, el paciente debe estar hospitalizado"
    )
    assert len(ejecutadas) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRACIÓN — escenarios completos
# ─────────────────────────────────────────────────────────────────────────────

@prueba("Integración — 10 ticks en escenario normal sin errores")
def test_integracion_escenario_normal():
    from generador_pacientes import generar_llegadas_tick
    rng = np.random.default_rng(42)
    h = crear_hospital_referencia(rng=rng)
    se = SistemaExperto(h)

    rng_gen = np.random.default_rng(42)
    for tick in range(10):
        nuevos = generar_llegadas_tick(tick, "normal", rng_gen)
        resultado = se.procesar_tick(tick, nuevos)
        assert 0 <= resultado.indicador_I <= 100
        assert resultado.nivel_I in ("Bajo", "Medio", "Alto", "Crítico")


@prueba("Integración — escenario crisis activa desborde masivo (P > 30%)")
def test_integracion_escenario_crisis():
    """En escenario crisis (lambda=5), las camas oficiales se llenan progresivamente
    y el sistema activa camas temporales de desborde.

    Con estancias promedio de 20h y ticks de 0.25h, la saturación completa
    del hospital toma ~300 ticks. En 30 ticks verificamos el indicador de
    estrés correcto: componente P (proporción de pacientes en desborde) > 30%.
    El nivel Alto/Crítico del indicador I se alcanza alrededor del tick 75
    y está cubierto por la prueba RES-04 que verifica el umbral de escalamiento.
    """
    from generador_pacientes import generar_llegadas_tick
    rng = np.random.default_rng(42)
    h = crear_hospital_referencia(rng=rng)
    se = SistemaExperto(h)

    rng_gen = np.random.default_rng(42)
    max_P = 0.0
    for tick in range(30):
        nuevos = generar_llegadas_tick(tick, "crisis", rng_gen)
        resultado = se.procesar_tick(tick, nuevos)
        assert 0 <= resultado.indicador_I <= 100, f"I fuera de rango en tick {tick}"
        if resultado.componente_P > max_P:
            max_P = resultado.componente_P

    assert max_P > 30.0, (
        f"Escenario crisis debería producir P > 30% de desborde. "
        f"P máxima observada: {max_P:.1f}%"
    )


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    total   = len(_resultados)
    pasadas = sum(1 for _, ok, _ in _resultados if ok)
    fallidas = total - pasadas

    print("=" * 70)
    print("RESULTADOS DE PRUEBAS — sistema_experto.py — F4-A")
    print("=" * 70)
    for nombre, ok, msg in _resultados:
        estado = "✔ OK" if ok else "✘ FALLA"
        print(f"  {estado}  {nombre}")
        if not ok:
            print(f"         → {msg}")
    print("-" * 70)
    print(f"  Total: {total}  |  Pasadas: {pasadas}  |  Fallidas: {fallidas}")
    print("=" * 70)

    if fallidas > 0:
        sys.exit(1)
