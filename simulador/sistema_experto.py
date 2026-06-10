"""
sistema_experto.py
==================
Simulador Inteligente de Ocupación Hospitalaria
Fase 4-A — Modelado — Sistema Experto (SEMMA·Model)

Autor  : Juan Camilo García Braham
Curso  : IA en Salud · Maestría IA y CD · UTP
Año    : 2026
Stack  : Python 3.12 · numpy 2.4 · scipy 1.17 · pandas 3.0

Decisiones de diseño activas — F4-A
─────────────────────────────────────
D_F4A_001  Actualización cama↔paciente siempre en funciones atómicas
           _asignar_cama() y _trasladar_paciente(). Garantiza invariante D013.
D_F4A_002  Transición esperando→dado_de_alta bloqueada con TransicionInvalidaError.
D_F4A_003  Orden de procesamiento en tick:
           (1) limpieza  (2) altas  (3) traslados  (4) asignaciones  (5) escalamiento
D_F4A_004  Selección de cama determinista (menor id_cama lexicográfico) dentro del área.
D_F4A_005  P1 sin UCI permanece en cola; no se reasigna. Activa RES-04.
D_F4A_006  Camas temporales en Área.camas_temporales (lista separada de camas oficiales).
D_F4A_007  Camas temporales dinámicas se destruyen al alta; preexistentes vuelven a libre.

Reglas implementadas
─────────────────────
Capa 1 — Elegibilidad : RE-01, RE-02, RE-03
Capa 2 — Priorización : RP-01, RP-02, RP-03
Capa 3 — Asignación   : RA-01, RA-02
Capa 4 — Sobreocupación: RSO-01, RSO-02, RSO-03, RSO-04
Capa 5 — Escalamiento : RES-01, RES-02, RES-03, RES-04
Capa 6 — Alta/limpieza: RAL-01, RAL-02, RAL-03

Riesgos activos monitoreados
──────────────────────────────
R02  Casos borde del sistema experto (transiciones inválidas, inconsistencias cama↔paciente)
R09  Sesgos en reglas (ningún atributo demográfico interviene en asignación)

Indicador compuesto (F1 — Entregable 3)
─────────────────────────────────────────
I = 0.4·O + 0.2·E + 0.2·P + 0.2·C
Niveles: Bajo(0-25) · Medio(26-50) · Alto(51-75) · Crítico(76-100)

Parámetros de tiempo (D001, D002, D003 — F1)
──────────────────────────────────────────────
1 tick = 0.25 h (15 min)          — D003
t_max_ref = 240 min = 4 h         — D001
t_umbral_critico = 30 min = 2 ticks — D002
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DEL DOMINIO
# ─────────────────────────────────────────────────────────────────────────────

DURACION_TICK_H: float = 0.25          # D003 — 1 tick = 15 min = 0.25 h
T_MAX_REF_H: float = 4.0               # D001 — 240 min en horas
T_UMBRAL_CRITICO_TICKS: int = 2        # D002 — 30 min = 2 ticks

# Mapeo prioridades_aceptadas por área — D010 (F2, aprobado)
PRIORIDADES_ACEPTADAS: dict[str, list[str]] = {
    "UCI":             ["P1"],
    "Urgencias":       ["P1", "P2", "P3"],
    "Hospitalización": ["P2", "P3"],
    "Observación":     ["P2", "P3", "P4"],
    "Sala_de_espera":  ["P3", "P4"],
}

# Jerarquía numérica de prioridad para comparaciones (menor = más urgente)
PESO_PRIORIDAD: dict[str, int] = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}

# Áreas de desborde autorizadas por prioridad — RSO-03
AREA_DESBORDE_POR_PRIORIDAD: dict[str, str] = {
    "P1": "UCI",            # P1 nunca se redirige (RSO-04 / D_F4A_005)
    "P2": "Observación",
    "P3": "Observación",
    "P4": "Sala_de_espera",
}

# Cadena de traslados internos permitidos — RSO-02
# Origen → destinos posibles (en orden de preferencia)
TRASLADOS_PERMITIDOS: dict[str, list[str]] = {
    "Urgencias":       ["Observación"],
    "Hospitalización": ["Observación"],
    "Observación":     ["Sala_de_espera"],
    "UCI":             [],               # UCI no traslada hacia abajo en sobreocupación
    "Sala_de_espera":  [],
}

# ─────────────────────────────────────────────────────────────────────────────
# EXCEPCIONES TIPADAS — R02
# ─────────────────────────────────────────────────────────────────────────────

class TransicionInvalidaError(Exception):
    """R02 — Intento de transición de estado no permitida (F2, Tabla 5 y 6)."""

class InconsistenciaModeloError(Exception):
    """R02 — Violación de invariante bidireccional cama↔paciente (D013)."""

# ─────────────────────────────────────────────────────────────────────────────
# DATACLASSES — entidades del dominio
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Cama:
    """Entidad Cama según diccionario de datos F2, Tabla 2.

    tipo: 'normal' | 'UCI' | 'observacion' | 'temporal'
    estado: 'libre' | 'ocupada' | 'en_limpieza'
    """
    id_cama: str
    tipo: Literal["normal", "UCI", "observacion", "temporal"]
    area_id: str                                # FK → Área.id_area
    estado: Literal["libre", "ocupada", "en_limpieza"] = "libre"
    paciente_id: str | None = None
    tiempo_limpieza_restante: float = 0.0       # horas; Uniforme[0.25, 1.0]
    es_temporal: bool = False                   # D_F4A_006 — cama dinámica


@dataclass
class Area:
    """Entidad Área según diccionario de datos F2, Tabla 3.

    camas             : camas oficiales (fijas, D008)
    camas_temporales  : camas dinámicas de desborde (D_F4A_006)
    capacidad_disponible puede ser negativo (D004)
    """
    id_area: str
    nombre: Literal["UCI", "Urgencias", "Hospitalización",
                    "Observación", "Sala_de_espera"]
    piso_id: str
    capacidad_total: int
    acepta_desborde: bool
    camas: list[str] = field(default_factory=list)            # id_cama oficiales
    camas_temporales: list[str] = field(default_factory=list) # D_F4A_006
    capacidad_disponible: int = 0                             # calculado; puede ser <0

    @property
    def prioridades_aceptadas(self) -> list[str]:
        """D010 — mapeo centralizado, no hardcoded por instancia."""
        return PRIORIDADES_ACEPTADAS.get(self.nombre, [])

    @property
    def todas_las_camas(self) -> list[str]:
        """Unión de camas oficiales y temporales para búsqueda."""
        return self.camas + self.camas_temporales


@dataclass
class Piso:
    """Entidad Piso según diccionario de datos F2, Tabla 4."""
    id_piso: str
    nombre: str
    areas: list[str] = field(default_factory=list)  # id_area


@dataclass
class Paciente:
    """Entidad Paciente según diccionario de datos F2, Tabla 1.

    Los campos calculados se actualizan exclusivamente por el sistema experto.
    Ningún atributo demográfico (edad) interviene en las reglas de asignación — R09.
    """
    id_paciente: str
    edad: int
    prioridad_clinica: Literal["P1", "P2", "P3", "P4"]
    area_requerida: str
    tiempo_estancia_esperado: float             # horas
    tiempo_espera: float = 0.0                  # horas — RAL-03
    tiempo_en_sistema: float = 0.0              # horas — RAL-03
    estado: Literal["esperando", "hospitalizado",
                    "trasladado", "dado_de_alta"] = "esperando"
    cama_id: str | None = None
    tick_ingreso: int = 0
    es_desborde: bool = False
    tick_asignacion: int | None = None          # para cálculo de tiempo_en_cama


# ─────────────────────────────────────────────────────────────────────────────
# ACCIÓN — estructura de salida del sistema experto
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Accion:
    """Representa una acción propuesta o ejecutada por el sistema experto.

    En modo automático se ejecuta inmediatamente.
    En modo asistido se acumula en una lista y se presenta al gestor.

    tipo: 'asignacion' | 'traslado' | 'desborde' | 'alta' | 'escalamiento'
    """
    tipo: Literal["asignacion", "traslado", "desborde", "alta", "escalamiento"]
    descripcion: str
    id_paciente: str
    id_cama_destino: str | None = None
    id_cama_origen: str | None = None
    area_destino: str | None = None
    ejecutada: bool = False


@dataclass
class ResultadoTick:
    """Resultado completo de procesar un tick."""
    tick: int
    acciones_ejecutadas: list[Accion] = field(default_factory=list)
    acciones_pendientes: list[Accion] = field(default_factory=list)  # modo asistido
    indicador_I: float = 0.0
    nivel_I: str = "Bajo"
    componente_O: float = 0.0
    componente_E: float = 0.0
    componente_P: float = 0.0
    componente_C: float = 0.0
    alertas: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# HOSPITAL — contenedor del estado global
# ─────────────────────────────────────────────────────────────────────────────

class Hospital:
    """Estado global del hospital.

    Mantiene los registros de pacientes, camas, áreas y pisos.
    Provee métodos de acceso O(1) mediante índices por id.
    Relaciones bidireccionales con redundancia controlada — D013.
    """

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.pacientes:  dict[str, Paciente] = {}
        self.camas:      dict[str, Cama]     = {}
        self.areas:      dict[str, Area]     = {}
        self.pisos:      dict[str, Piso]     = {}
        self._area_por_nombre: dict[str, str] = {}  # nombre → id_area (acceso rápido)
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.tick_actual: int = 0

    # ── Construcción del hospital ─────────────────────────────────────────────

    def agregar_piso(self, nombre: str) -> Piso:
        piso = Piso(id_piso=str(uuid.uuid4()), nombre=nombre)
        self.pisos[piso.id_piso] = piso
        return piso

    def agregar_area(self, nombre: str, piso_id: str,
                     capacidad: int, acepta_desborde: bool) -> Area:
        area = Area(
            id_area=str(uuid.uuid4()),
            nombre=nombre,
            piso_id=piso_id,
            capacidad_total=capacidad,
            capacidad_disponible=capacidad,
            acepta_desborde=acepta_desborde,
        )
        self.areas[area.id_area] = area
        self._area_por_nombre[nombre] = area.id_area
        self.pisos[piso_id].areas.append(area.id_area)
        return area

    def agregar_cama(self, area_id: str,
                     tipo: Literal["normal", "UCI", "observacion", "temporal"]
                     ) -> Cama:
        cama = Cama(id_cama=str(uuid.uuid4()), tipo=tipo, area_id=area_id)
        self.camas[cama.id_cama] = cama
        self.areas[area_id].camas.append(cama.id_cama)
        return cama

    def ingresar_paciente(self, datos: dict) -> Paciente:
        """Crea un Paciente desde un PacienteDict del generador (D_F3_002)."""
        p = Paciente(
            id_paciente=datos["id_paciente"],
            edad=datos["edad"],
            prioridad_clinica=datos["prioridad_clinica"],
            area_requerida=datos["area_requerida"],
            tiempo_estancia_esperado=datos["tiempo_estancia_esperado"],
            tick_ingreso=datos.get("tick_ingreso", self.tick_actual),
        )
        self.pacientes[p.id_paciente] = p
        return p

    # ── Acceso rápido por nombre de área ─────────────────────────────────────

    def area_por_nombre(self, nombre: str) -> Area | None:
        aid = self._area_por_nombre.get(nombre)
        return self.areas[aid] if aid else None

    # ── Consultas de estado ───────────────────────────────────────────────────

    def pacientes_esperando(self) -> list[Paciente]:
        return [p for p in self.pacientes.values() if p.estado == "esperando"]

    def pacientes_hospitalizados(self) -> list[Paciente]:
        return [p for p in self.pacientes.values()
                if p.estado in ("hospitalizado", "trasladado")]

    def camas_libres_en_area(self, area_id: str) -> list[Cama]:
        """RE-01 — solo camas libres (oficiales + temporales)."""
        area = self.areas[area_id]
        return sorted(
            [self.camas[cid] for cid in area.todas_las_camas
             if self.camas[cid].estado == "libre"],
            key=lambda c: c.id_cama,   # RA-02 — determinista
        )

    def _recalcular_capacidad_disponible(self, area_id: str) -> None:
        """Sincroniza capacidad_disponible con el estado real de camas oficiales."""
        area = self.areas[area_id]
        ocupadas_oficiales = sum(
            1 for cid in area.camas
            if self.camas[cid].estado == "ocupada"
        )
        area.capacidad_disponible = area.capacidad_total - ocupadas_oficiales


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES ATÓMICAS — D_F4A_001
# ─────────────────────────────────────────────────────────────────────────────

def _asignar_cama(hospital: Hospital, paciente: Paciente,
                  cama: Cama, tick: int, es_desborde: bool = False) -> None:
    """D_F4A_001 — Asignación atómica cama↔paciente.

    Actualiza ambas referencias en par (D013). Lanza InconsistenciaModeloError
    si la cama ya está ocupada o el paciente ya tiene cama asignada (R02).
    """
    if cama.estado != "libre":
        raise InconsistenciaModeloError(
            f"R02 — Cama {cama.id_cama} no está libre (estado={cama.estado})"
        )
    if paciente.cama_id is not None:
        raise InconsistenciaModeloError(
            f"R02 — Paciente {paciente.id_paciente} ya tiene cama asignada"
        )
    if paciente.estado == "dado_de_alta":
        raise TransicionInvalidaError(
            f"R02/D_F4A_002 — Paciente dado_de_alta no puede recibir cama"
        )

    # Actualización atómica
    cama.estado = "ocupada"
    cama.paciente_id = paciente.id_paciente
    paciente.cama_id = cama.id_cama
    paciente.estado = "hospitalizado"
    paciente.es_desborde = es_desborde
    paciente.tick_asignacion = tick

    # Sincronizar capacidad del área
    hospital._recalcular_capacidad_disponible(cama.area_id)


def _liberar_cama(hospital: Hospital, paciente: Paciente,
                  cama: Cama, rng: np.random.Generator) -> None:
    """D_F4A_001 — Liberación atómica cama↔paciente al alta o traslado.

    Si la cama es temporal dinámica (D_F4A_007), se elimina del registro.
    Si es preexistente, pasa a en_limpieza con tiempo Uniforme[0.25, 1.0].
    """
    if cama.estado != "ocupada":
        raise InconsistenciaModeloError(
            f"R02 — Cama {cama.id_cama} no está ocupada (estado={cama.estado})"
        )
    if cama.paciente_id != paciente.id_paciente:
        raise InconsistenciaModeloError(
            f"R02 — Cama {cama.id_cama} no está asignada al paciente "
            f"{paciente.id_paciente}"
        )

    area = hospital.areas[cama.area_id]

    # Desvinculación atómica
    cama.paciente_id = None
    paciente.cama_id = None

    if cama.es_temporal:
        # D_F4A_007 — cama dinámica: se destruye
        cama.estado = "libre"   # estado previo a eliminación
        if cama.id_cama in area.camas_temporales:
            area.camas_temporales.remove(cama.id_cama)
        del hospital.camas[cama.id_cama]
    else:
        # Cama oficial: ciclo de limpieza RAL-02
        cama.estado = "en_limpieza"
        cama.tiempo_limpieza_restante = float(rng.uniform(0.25, 1.0))

    hospital._recalcular_capacidad_disponible(cama.area_id)


def _trasladar_paciente(hospital: Hospital, paciente: Paciente,
                        cama_origen: Cama, cama_destino: Cama,
                        tick: int, rng: np.random.Generator) -> None:
    """D_F4A_001 — Traslado atómico: libera cama_origen y asigna cama_destino.

    RSO-02: actualiza cuatro referencias en par sin estado intermedio
    inválido. La secuencia es: capturar datos → liberar → reasignar.
    """
    if cama_destino.estado != "libre":
        raise InconsistenciaModeloError(
            f"R02 — Cama destino {cama_destino.id_cama} no está libre"
        )

    es_desborde_destino = (
        hospital.areas[cama_destino.area_id].nombre
        != paciente.area_requerida
    )

    # 1. Resetear estado de paciente para permitir reasignación
    paciente.estado = "esperando"
    paciente.cama_id = None
    cama_origen.estado = "en_limpieza"
    cama_origen.paciente_id = None
    cama_origen.tiempo_limpieza_restante = float(rng.uniform(0.25, 1.0))
    hospital._recalcular_capacidad_disponible(cama_origen.area_id)

    # 2. Asignar destino
    _asignar_cama(hospital, paciente, cama_destino, tick,
                  es_desborde=es_desborde_destino)
    paciente.estado = "trasladado"


def _crear_cama_temporal(hospital: Hospital, area_id: str) -> Cama:
    """D_F4A_006/007 — Crea y registra una cama temporal dinámica en el área."""
    area = hospital.areas[area_id]
    tipo_temporal: Literal["normal", "UCI", "observacion", "temporal"] = "temporal"
    cama = Cama(
        id_cama=str(uuid.uuid4()),
        tipo=tipo_temporal,
        area_id=area_id,
        estado="libre",
        es_temporal=True,
    )
    hospital.camas[cama.id_cama] = cama
    area.camas_temporales.append(cama.id_cama)
    # Las temporales no cuentan para capacidad_total; capacidad_disponible
    # puede volverse negativo (D004) para señalar desborde
    area.capacidad_disponible -= 1
    return cama


# ─────────────────────────────────────────────────────────────────────────────
# INDICADOR COMPUESTO I — RES-01 (F1, Entregable 3)
# ─────────────────────────────────────────────────────────────────────────────

def calcular_indicador_I(hospital: Hospital) -> dict:
    """RES-01 — Calcula I = 0.4·O + 0.2·E + 0.2·P + 0.2·C al final del tick.

    Fórmulas operacionales de F1 (D001, D007).
    Retorna dict con I, nivel y los cuatro componentes desagregados.
    """
    # ── Componente O — porcentaje de ocupación ────────────────────────────────
    camas_oficiales = [c for c in hospital.camas.values() if not c.es_temporal]
    total_camas = len(camas_oficiales)
    # Excluir camas en_limpieza del denominador (F1, sección 2.1)
    camas_validas = [c for c in camas_oficiales
                     if c.estado in ("libre", "ocupada")]
    camas_ocupadas = [c for c in camas_validas if c.estado == "ocupada"]

    if len(camas_validas) > 0:
        O = (len(camas_ocupadas) / len(camas_validas)) * 100
    else:
        O = 0.0

    # ── Componente E — tiempo promedio de espera normalizado ──────────────────
    esperando = hospital.pacientes_esperando()
    if esperando:
        t_prom_espera_h = sum(p.tiempo_espera for p in esperando) / len(esperando)
        t_prom_espera_min = t_prom_espera_h * 60
        E = min((t_prom_espera_min / (T_MAX_REF_H * 60)) * 100, 100)
    else:
        E = 0.0

    # ── Componente P — proporción pacientes en áreas temporales ──────────────
    hospitalizados = hospital.pacientes_hospitalizados()
    en_desborde = [p for p in hospitalizados if p.es_desborde]
    if hospitalizados:
        P = (len(en_desborde) / len(hospitalizados)) * 100
    else:
        P = 0.0

    # ── Componente C — pacientes críticos sin cama ────────────────────────────
    todos_criticos = [p for p in hospital.pacientes.values()
                      if p.prioridad_clinica in ("P1", "P2")
                      and p.estado != "dado_de_alta"]
    criticos_sin_cama = [p for p in todos_criticos if p.estado == "esperando"]
    if todos_criticos:
        C = (len(criticos_sin_cama) / len(todos_criticos)) * 100
    else:
        C = 0.0

    I = 0.4 * O + 0.2 * E + 0.2 * P + 0.2 * C

    # Nivel del indicador
    if I <= 25:
        nivel = "Bajo"
    elif I <= 50:
        nivel = "Medio"
    elif I <= 75:
        nivel = "Alto"
    else:
        nivel = "Crítico"

    return {
        "I": round(I, 2),
        "nivel": nivel,
        "O": round(O, 2),
        "E": round(E, 2),
        "P": round(P, 2),
        "C": round(C, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MOTOR DE TICK — lógica principal
# ─────────────────────────────────────────────────────────────────────────────

class SistemaExperto:
    """Motor principal del sistema experto.

    Expone dos modos de operación:
      - modo_automatico: ejecuta todas las acciones del tick sin intervención.
      - modo_asistido  : acumula acciones propuestas y las devuelve para
                         confirmación del gestor antes de ejecutarlas.
    """

    def __init__(self, hospital: Hospital) -> None:
        self.hospital = hospital
        self.rng = hospital.rng

    # ─────────────────────────────────────────────────────────────────────────
    # PASO 1 — Limpieza de camas (RAL-02)
    # ─────────────────────────────────────────────────────────────────────────

    def _procesar_limpieza(self) -> None:
        """RAL-02 — Decrementa tiempo de limpieza; libera cama cuando llega a 0."""
        for cama in list(self.hospital.camas.values()):
            if cama.estado == "en_limpieza":
                cama.tiempo_limpieza_restante -= DURACION_TICK_H
                if cama.tiempo_limpieza_restante <= 0:
                    cama.estado = "libre"
                    cama.tiempo_limpieza_restante = 0.0
                    self.hospital._recalcular_capacidad_disponible(cama.area_id)

    # ─────────────────────────────────────────────────────────────────────────
    # PASO 2 — Altas (RAL-01)
    # ─────────────────────────────────────────────────────────────────────────

    def _procesar_altas(self, tick: int) -> list[Accion]:
        """RAL-01 — Da de alta a pacientes que cumplieron tiempo de estancia."""
        acciones: list[Accion] = []
        for paciente in list(self.hospital.pacientes.values()):
            if paciente.estado not in ("hospitalizado", "trasladado"):
                continue
            if paciente.tick_asignacion is None:
                continue

            tiempo_en_cama_h = (tick - paciente.tick_asignacion) * DURACION_TICK_H
            if tiempo_en_cama_h >= paciente.tiempo_estancia_esperado:
                cama = self.hospital.camas[paciente.cama_id]
                _liberar_cama(self.hospital, paciente, cama, self.rng)
                paciente.estado = "dado_de_alta"

                acciones.append(Accion(
                    tipo="alta",
                    descripcion=(
                        f"Alta: paciente {paciente.id_paciente[:8]} "
                        f"({paciente.prioridad_clinica}) tras "
                        f"{tiempo_en_cama_h:.2f}h en cama"
                    ),
                    id_paciente=paciente.id_paciente,
                    id_cama_origen=cama.id_cama,
                    ejecutada=True,
                ))
        return acciones

    # ─────────────────────────────────────────────────────────────────────────
    # PASO 3 — Traslados internos por sobreocupación (RSO-02)
    # ─────────────────────────────────────────────────────────────────────────

    def _buscar_traslados_posibles(self) -> list[tuple[Paciente, Cama, Cama]]:
        """RSO-01/02 — Identifica traslados candidatos (paciente, origen, destino).

        Solo se propone trasladar pacientes P3/P4 de áreas con
        capacidad_disponible == 0 hacia áreas compatibles con espacio.
        Orden: P4 antes que P3; entre iguales, mayor tiempo_en_sistema.
        """
        candidatos: list[tuple[Paciente, Cama, Cama]] = []

        for area in self.hospital.areas.values():
            # RSO-01 — área en sobreocupación
            if area.capacidad_disponible > 0:
                continue

            destinos_posibles = TRASLADOS_PERMITIDOS.get(area.nombre, [])
            if not destinos_posibles:
                continue

            # Pacientes trasladables: P3/P4 hospitalizados en esta área
            candidatos_area = [
                p for p in self.hospital.pacientes.values()
                if p.estado in ("hospitalizado", "trasladado")
                and p.cama_id is not None
                and self.hospital.camas[p.cama_id].area_id == area.id_area
                and p.prioridad_clinica in ("P3", "P4")
            ]
            # Orden: P4 primero, luego P3; desempate por mayor tiempo_en_sistema
            candidatos_area.sort(
                key=lambda p: (PESO_PRIORIDAD[p.prioridad_clinica],
                               -p.tiempo_en_sistema)
            )

            for paciente in candidatos_area:
                for nombre_destino in destinos_posibles:
                    area_destino = self.hospital.area_por_nombre(nombre_destino)
                    if area_destino is None:
                        continue
                    # RE-03 — compatibilidad de prioridad en destino
                    if paciente.prioridad_clinica not in area_destino.prioridades_aceptadas:
                        continue
                    camas_libres = self.hospital.camas_libres_en_area(
                        area_destino.id_area
                    )
                    if not camas_libres:
                        continue

                    cama_origen = self.hospital.camas[paciente.cama_id]
                    candidatos.append((paciente, cama_origen, camas_libres[0]))
                    break  # un traslado por paciente por tick

        return candidatos

    def _procesar_traslados(self, tick: int,
                             modo_asistido: bool) -> list[Accion]:
        """RSO-02 — Ejecuta o propone traslados internos.

        Para evitar condición de carrera cuando múltiples traslados ocurren
        en el mismo tick, se re-verifica la disponibilidad de cama destino
        justo antes de ejecutar cada traslado individual. Si la cama ya fue
        ocupada por un traslado anterior en el mismo tick, se busca otra libre
        en el mismo área destino antes de descartar el traslado.
        """
        acciones: list[Accion] = []
        # Recolectar candidatos (paciente + cama_origen + nombre_área_destino)
        candidatos_raw = self._buscar_traslados_posibles()

        for paciente, cama_origen, cama_destino_inicial in candidatos_raw:
            area_destino = self.hospital.areas[cama_destino_inicial.area_id]

            # Re-verificar disponibilidad antes de ejecutar (anti condición de carrera)
            if cama_destino_inicial.estado != "libre":
                # Buscar otra cama libre en el mismo área destino
                alternativas = self.hospital.camas_libres_en_area(area_destino.id_area)
                if not alternativas:
                    continue   # No hay cama disponible en este tick; paciente espera
                cama_destino = alternativas[0]
            else:
                cama_destino = cama_destino_inicial

            desc = (
                f"Traslado: {paciente.id_paciente[:8]} "
                f"({paciente.prioridad_clinica}) → {area_destino.nombre}"
            )
            accion = Accion(
                tipo="traslado",
                descripcion=desc,
                id_paciente=paciente.id_paciente,
                id_cama_origen=cama_origen.id_cama,
                id_cama_destino=cama_destino.id_cama,
                area_destino=area_destino.nombre,
                ejecutada=False,
            )
            if not modo_asistido:
                _trasladar_paciente(
                    self.hospital, paciente, cama_origen, cama_destino,
                    tick, self.rng
                )
                accion.ejecutada = True
            acciones.append(accion)

        return acciones

    # ─────────────────────────────────────────────────────────────────────────
    # PASO 4 — Asignación de nuevos pacientes (RA-01, RSO-03/04)
    # ─────────────────────────────────────────────────────────────────────────

    def _ordenar_cola(self, cola: list[Paciente]) -> list[Paciente]:
        """RP-01/02/03 — Ordena la cola de espera por jerarquía de prioridad.

        Orden de clave:
          1. PESO_PRIORIDAD ascendente (0=P1 más urgente) — RP-01
          2. tiempo_espera negativo ascendente (mayor espera primero) — RP-02
          3. area_requerida alfabético como segundo desempate — RP-03
        """
        return sorted(
            cola,
            key=lambda p: (
                PESO_PRIORIDAD[p.prioridad_clinica],   # RP-01
                -p.tiempo_espera,                       # RP-02: negativo → mayor espera primero
                p.area_requerida,                       # RP-03
            )
        )

    def _intentar_asignacion_normal(self, paciente: Paciente,
                                     tick: int) -> Accion | None:
        """RA-01 — Intenta asignar al área requerida del paciente.

        RE-01: solo camas libres.
        RE-02: área requerida.
        RE-03: compatibilidad de prioridad (ya garantizada por D011).
        """
        area = self.hospital.area_por_nombre(paciente.area_requerida)
        if area is None:
            return None
        # RE-03 — verificación explícita (R09)
        if paciente.prioridad_clinica not in area.prioridades_aceptadas:
            return None

        camas = self.hospital.camas_libres_en_area(area.id_area)
        if not camas:
            return None

        cama = camas[0]   # RA-02 — determinista
        _asignar_cama(self.hospital, paciente, cama, tick)
        return Accion(
            tipo="asignacion",
            descripcion=(
                f"Asignación: {paciente.id_paciente[:8]} "
                f"({paciente.prioridad_clinica}) → {area.nombre}"
            ),
            id_paciente=paciente.id_paciente,
            id_cama_destino=cama.id_cama,
            area_destino=area.nombre,
            ejecutada=True,
        )

    def _intentar_desborde(self, paciente: Paciente,
                           tick: int) -> Accion | None:
        """RSO-03 — Busca o crea cama temporal de desborde.

        RSO-04 — P1 nunca se redirige; permanece en espera.
        """
        # RSO-04 / D_F4A_005 — P1 no se redirige
        if paciente.prioridad_clinica == "P1":
            return None

        nombre_desborde = AREA_DESBORDE_POR_PRIORIDAD.get(
            paciente.prioridad_clinica
        )
        if nombre_desborde is None:
            return None

        area_desborde = self.hospital.area_por_nombre(nombre_desborde)
        if area_desborde is None or not area_desborde.acepta_desborde:
            return None
        # RE-03 — prioridad compatible con área de desborde
        if paciente.prioridad_clinica not in area_desborde.prioridades_aceptadas:
            return None

        # Paso 1: buscar cama temporal libre preexistente
        camas_temp_libres = [
            self.hospital.camas[cid]
            for cid in area_desborde.camas_temporales
            if self.hospital.camas[cid].estado == "libre"
        ]
        if camas_temp_libres:
            cama = sorted(camas_temp_libres, key=lambda c: c.id_cama)[0]
        else:
            # Paso 2: crear cama temporal dinámica (D_F4A_006)
            cama = _crear_cama_temporal(self.hospital, area_desborde.id_area)

        _asignar_cama(self.hospital, paciente, cama, tick, es_desborde=True)
        return Accion(
            tipo="desborde",
            descripcion=(
                f"Desborde: {paciente.id_paciente[:8]} "
                f"({paciente.prioridad_clinica}) → {area_desborde.nombre} "
                f"(cama temporal)"
            ),
            id_paciente=paciente.id_paciente,
            id_cama_destino=cama.id_cama,
            area_destino=area_desborde.nombre,
            ejecutada=True,
        )

    def _procesar_asignaciones(self, tick: int,
                                modo_asistido: bool) -> list[Accion]:
        """RA-01/RSO-03/04 — Procesa la cola de espera completa.

        En modo asistido, las acciones de desborde se proponen pero no
        se ejecutan hasta confirmación del gestor.
        """
        acciones: list[Accion] = []
        cola = self._ordenar_cola(self.hospital.pacientes_esperando())

        for paciente in cola:
            # Intento 1: asignación normal al área requerida
            accion = self._intentar_asignacion_normal(paciente, tick)
            if accion:
                acciones.append(accion)
                continue

            # Intento 2: desborde (RSO-03) — no aplica a P1 (RSO-04)
            if not modo_asistido:
                accion = self._intentar_desborde(paciente, tick)
                if accion:
                    acciones.append(accion)
            else:
                # Modo asistido: proponer desborde sin ejecutar
                accion = self._proponer_desborde(paciente)
                if accion:
                    acciones.append(accion)

        return acciones

    def _proponer_desborde(self, paciente: Paciente) -> Accion | None:
        """Modo asistido — Genera propuesta de desborde sin ejecutar."""
        if paciente.prioridad_clinica == "P1":
            return None
        nombre_desborde = AREA_DESBORDE_POR_PRIORIDAD.get(
            paciente.prioridad_clinica
        )
        if nombre_desborde is None:
            return None
        area_desborde = self.hospital.area_por_nombre(nombre_desborde)
        if area_desborde is None:
            return None
        return Accion(
            tipo="desborde",
            descripcion=(
                f"PROPUESTA — Desborde: {paciente.id_paciente[:8]} "
                f"({paciente.prioridad_clinica}) → {nombre_desborde} "
                f"(requiere confirmación)"
            ),
            id_paciente=paciente.id_paciente,
            area_destino=nombre_desborde,
            ejecutada=False,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PASO 5 — Escalamiento y actualización de tiempos (RES-01..04, RAL-03)
    # ─────────────────────────────────────────────────────────────────────────

    def _actualizar_tiempos(self) -> None:
        """RAL-03 — Actualiza tiempo_espera y tiempo_en_sistema en cada tick."""
        for paciente in self.hospital.pacientes.values():
            if paciente.estado == "esperando":
                paciente.tiempo_espera += DURACION_TICK_H
            if paciente.estado in ("hospitalizado", "trasladado"):
                paciente.tiempo_en_sistema += DURACION_TICK_H

    def _evaluar_escalamiento(self, tick: int) -> list[str]:
        """RES-02/03/04 — Evalúa condiciones de escalamiento y retorna alertas."""
        alertas: list[str] = []
        for paciente in self.hospital.pacientes_esperando():
            if paciente.prioridad_clinica in ("P1", "P2"):
                ticks_esperando = tick - paciente.tick_ingreso
                if ticks_esperando >= T_UMBRAL_CRITICO_TICKS:
                    alertas.append(
                        f"RES-04 — {paciente.prioridad_clinica} "
                        f"{paciente.id_paciente[:8]} lleva "
                        f"{ticks_esperando} ticks sin cama (umbral={T_UMBRAL_CRITICO_TICKS})"
                    )
        return alertas

    # ─────────────────────────────────────────────────────────────────────────
    # MOTOR PRINCIPAL DE TICK
    # ─────────────────────────────────────────────────────────────────────────

    def procesar_tick(self, tick: int,
                      nuevos_pacientes: list[dict] | None = None,
                      modo_asistido: bool = False) -> ResultadoTick:
        """D_F4A_003 — Ejecuta un tick completo en orden:
        (1) limpieza → (2) altas → (3) traslados → (4) asignaciones → (5) escalamiento

        Parámetros
        ----------
        tick            : número de tick actual
        nuevos_pacientes: lista de PacienteDict del generador (puede ser vacía)
        modo_asistido   : si True, las acciones de desborde/traslado se proponen
                          pero no se ejecutan hasta confirmar_acciones()

        Retorna
        -------
        ResultadoTick con acciones ejecutadas, pendientes, indicador I y alertas.
        """
        self.hospital.tick_actual = tick
        resultado = ResultadoTick(tick=tick)

        # Ingresar nuevos pacientes a la cola
        if nuevos_pacientes:
            for datos in nuevos_pacientes:
                datos["tick_ingreso"] = tick
                self.hospital.ingresar_paciente(datos)

        # (1) Limpieza
        self._procesar_limpieza()

        # (2) Altas
        acciones_altas = self._procesar_altas(tick)
        resultado.acciones_ejecutadas.extend(acciones_altas)

        # (3) Traslados por sobreocupación
        acciones_traslados = self._procesar_traslados(tick, modo_asistido)
        for a in acciones_traslados:
            if a.ejecutada:
                resultado.acciones_ejecutadas.append(a)
            else:
                resultado.acciones_pendientes.append(a)

        # (4) Asignaciones nuevas
        acciones_asig = self._procesar_asignaciones(tick, modo_asistido)
        for a in acciones_asig:
            if a.ejecutada:
                resultado.acciones_ejecutadas.append(a)
            else:
                resultado.acciones_pendientes.append(a)

        # (5) Actualizar tiempos + escalamiento + calcular I
        self._actualizar_tiempos()
        resultado.alertas = self._evaluar_escalamiento(tick)

        indicador = calcular_indicador_I(self.hospital)
        resultado.indicador_I = indicador["I"]
        resultado.nivel_I = indicador["nivel"]
        resultado.componente_O = indicador["O"]
        resultado.componente_E = indicador["E"]
        resultado.componente_P = indicador["P"]
        resultado.componente_C = indicador["C"]

        # RES-04 — forzar nivel Crítico si hay alertas de umbral
        if resultado.alertas and resultado.indicador_I < 76:
            resultado.nivel_I = "Crítico"
            resultado.alertas.append(
                "RES-04 — Nivel forzado a Crítico por P1/P2 sin cama > umbral"
            )

        return resultado

    def confirmar_acciones(self, acciones: list[Accion],
                           tick: int) -> list[Accion]:
        """Modo asistido — Ejecuta las acciones previamente propuestas.

        El gestor selecciona cuáles confirmar pasando la sublista.
        Acciones no confirmadas se descartan (el paciente permanece en espera).
        """
        ejecutadas: list[Accion] = []
        for accion in acciones:
            if accion.ejecutada:
                continue

            paciente = self.hospital.pacientes.get(accion.id_paciente)
            if paciente is None or paciente.estado != "esperando":
                continue

            if accion.tipo == "traslado":
                if accion.id_cama_origen and accion.id_cama_destino:
                    cama_o = self.hospital.camas.get(accion.id_cama_origen)
                    cama_d = self.hospital.camas.get(accion.id_cama_destino)
                    if cama_o and cama_d:
                        _trasladar_paciente(
                            self.hospital, paciente, cama_o, cama_d,
                            tick, self.rng
                        )
                        accion.ejecutada = True
                        ejecutadas.append(accion)

            elif accion.tipo == "desborde":
                accion_real = self._intentar_desborde(paciente, tick)
                if accion_real:
                    accion.ejecutada = True
                    ejecutadas.append(accion)

        return ejecutadas


# ─────────────────────────────────────────────────────────────────────────────
# FÁBRICA — hospital de referencia F2 (D008)
# ─────────────────────────────────────────────────────────────────────────────

def crear_hospital_referencia(rng: np.random.Generator | None = None) -> Hospital:
    """Construye el hospital de 85 camas definido en D008 (F2).

    UCI=10, Urgencias=20, Hospitalización=40, Observación=15
    Sala_de_espera: 0 camas (es área de espera; los pacientes esperan
    pero no tienen cama asignada hasta desborde). Se agrega como área
    con capacidad simbólica para permitir RSO-03 con P4.
    """
    h = Hospital(rng=rng)

    p1 = h.agregar_piso("Piso 1 — Urgencias y UCI")
    p2 = h.agregar_piso("Piso 2 — Hospitalización y Observación")

    # Áreas y camas — D008
    area_uci  = h.agregar_area("UCI",             p1.id_piso, 10, acepta_desborde=False)
    area_urg  = h.agregar_area("Urgencias",        p1.id_piso, 20, acepta_desborde=False)
    area_hosp = h.agregar_area("Hospitalización",  p2.id_piso, 40, acepta_desborde=False)
    area_obs  = h.agregar_area("Observación",      p2.id_piso, 15, acepta_desborde=True)
    area_sala = h.agregar_area("Sala_de_espera",   p1.id_piso, 10, acepta_desborde=True)

    for _ in range(10):
        h.agregar_cama(area_uci.id_area,  "UCI")
    for _ in range(20):
        h.agregar_cama(area_urg.id_area,  "normal")
    for _ in range(40):
        h.agregar_cama(area_hosp.id_area, "normal")
    for _ in range(15):
        h.agregar_cama(area_obs.id_area,  "observacion")
    for _ in range(10):
        h.agregar_cama(area_sala.id_area, "normal")

    return h
