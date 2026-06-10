"""
generador_pacientes.py
======================
Simulador Inteligente de Ocupación Hospitalaria
Fase 3 — Preparación del dato (SEMMA·Explore+Modify)
Entregable 1: Módulo generador de pacientes sintéticos

Autor  : Juan Camilo García Braham
Curso  : IA en Salud · Maestría IA y CD · UTP
Año    : 2026
Stack  : Python 3.12 · numpy 2.4 · scipy 1.17 · pandas 3.0

Decisiones de diseño activas
─────────────────────────────
D_F3_001  SEED = 42 (semilla global documentada; reproducible)
D_F3_002  Salida como lista de dicts (portabilidad máxima; sin acoplamiento
          temprano a pandas o dataclasses)
D012      λ fija por escenario (Poisson estacionario; PMV académico)
D011      area_requerida: categórica condicional a prioridad_clinica
D010      Mapeo prioridades_aceptadas por área (aprobado F2)
D005      tiempo_estancia_esperado mínimo = 0.25 h (1 tick)

Parámetros de distribuciones (F2 — Entregable 2)
─────────────────────────────────────────────────
Llegada de pacientes / tick  : Poisson(λ) por escenario
Prioridad clínica            : Multinomial p=[0.05, 0.20, 0.45, 0.30]
Tiempo de estancia           : Log-normal truncada por área
Edad                         : Normal truncada µ=45 σ=20 [0,110]
Tiempo de limpieza           : Uniforme [0.25, 1.0] h (entidad Cama — referencia)
Área requerida               : Categórica condicional a prioridad
"""

from __future__ import annotations

import uuid
from typing import Literal, TypedDict

import numpy as np
from scipy.stats import truncnorm

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DEL DOMINIO
# ─────────────────────────────────────────────────────────────────────────────

# D_F3_001 — Semilla aleatoria global documentada
SEED: int = 42

# Escenarios válidos (F1 — Entregable 3, Tabla 13)
EscenariosValidos = Literal["normal", "alta_demanda", "crisis"]

# Tasas de llegada por escenario (pac/tick) — D012, F2 Tabla 8
LAMBDA_POR_ESCENARIO: dict[str, float] = {
    "normal":       1.5,   # ~6  pac/h — operación estable
    "alta_demanda": 3.0,   # ~12 pac/h — ocupación 85-95 %
    "crisis":       5.0,   # ~20 pac/h — ocupación >95 %
}

# Proporciones de prioridad clínica — F2 Tabla 9, R09
# P1=crítico · P2=urgente · P3=menos urgente · P4=no urgente
PRIORIDADES: list[str] = ["P1", "P2", "P3", "P4"]
PROB_PRIORIDAD: list[float] = [0.05, 0.20, 0.45, 0.30]

# Distribución de área requerida condicional a prioridad — D011, F2 Tabla 11
AREAS_POR_PRIORIDAD: dict[str, dict[str, float]] = {
    "P1": {"UCI": 0.80, "Urgencias": 0.20},
    "P2": {"Urgencias": 0.70, "Hospitalización": 0.30},
    "P3": {"Urgencias": 0.50, "Observación": 0.50},
    "P4": {"Observación": 0.60, "Sala_de_espera": 0.40},
}

# Parámetros log-normal de tiempo de estancia por área — F2 Tabla 10
# Formato: (µ_log, σ_log)  |  truncamiento global [0.25, 720.0] h — D005
ESTANCIA_LOG_NORMAL: dict[str, tuple[float, float]] = {
    "UCI":            (3.40, 0.60),   # media ~33 h · P95 ~90 h
    "Urgencias":      (1.80, 0.50),   # media  ~7 h · P95 ~18 h
    "Hospitalización":(2.90, 0.55),   # media ~20 h · P95 ~55 h
    "Observación":    (1.50, 0.45),   # media  ~5 h · P95 ~12 h
    "Sala_de_espera": (1.50, 0.45),   # asimilada a Observación (no especificada en F2)
}
ESTANCIA_MIN_H: float = 0.25   # D005 — 1 tick mínimo
ESTANCIA_MAX_H: float = 720.0  # 30 días; techo biológico razonable

# Distribución de edad — F2 sección 4
EDAD_MEDIA: float = 45.0
EDAD_SIGMA: float = 20.0
EDAD_MIN: int = 0
EDAD_MAX: int = 110

# ─────────────────────────────────────────────────────────────────────────────
# TYPE HINTS — estructura de salida (D_F3_002)
# ─────────────────────────────────────────────────────────────────────────────

class PacienteDict(TypedDict):
    """Estructura de un paciente generado sintéticamente.

    Los campos calculados (tiempo_espera, tiempo_en_sistema, cama_id,
    es_desborde) se inicializan con valores neutros; serán actualizados
    por el sistema experto en F4-A.
    """
    id_paciente:              str
    edad:                     int
    prioridad_clinica:        str   # P1 | P2 | P3 | P4
    area_requerida:           str
    tiempo_estancia_esperado: float # horas
    tiempo_espera:            float # horas — inicializado en 0.0
    tiempo_en_sistema:        float # horas — inicializado en 0.0
    estado:                   str   # esperando | hospitalizado | trasladado | dado_de_alta
    cama_id:                  None  # null al ingreso (F2 Tabla 5)
    tick_ingreso:             int
    es_desborde:              bool  # False al ingreso (F2 D007)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES INTERNAS
# ─────────────────────────────────────────────────────────────────────────────

def _generar_edad(rng: np.random.Generator, n: int) -> np.ndarray:
    """Muestrea n edades desde Normal truncada(µ=45, σ=20, [0, 110]).

    Usa scipy.stats.truncnorm con la parametrización estándar (a, b) en
    unidades de σ para evitar re-implementar el recorte manualmente.
    """
    a = (EDAD_MIN - EDAD_MEDIA) / EDAD_SIGMA   # = -2.25
    b = (EDAD_MAX - EDAD_MEDIA) / EDAD_SIGMA   # =  3.25
    # scipy.stats.truncnorm.rvs necesita random_state compatible con Generator
    muestras = truncnorm.rvs(a, b, loc=EDAD_MEDIA, scale=EDAD_SIGMA,
                             size=n, random_state=rng)
    return np.round(muestras).astype(int)


def _generar_prioridades(rng: np.random.Generator, n: int) -> np.ndarray:
    """Muestrea n prioridades desde Multinomial p=[0.05,0.20,0.45,0.30]."""
    indices = rng.choice(len(PRIORIDADES), size=n, p=PROB_PRIORIDAD)
    return np.array(PRIORIDADES)[indices]


def _generar_area_requerida(rng: np.random.Generator,
                            prioridad: str) -> str:
    """Muestrea un área desde la distribución categórica condicional (D011)."""
    opciones = AREAS_POR_PRIORIDAD[prioridad]
    areas     = list(opciones.keys())
    probs     = list(opciones.values())
    idx = rng.choice(len(areas), p=probs)
    return areas[idx]


def _generar_estancia(rng: np.random.Generator, area: str) -> float:
    """Muestrea tiempo de estancia desde Log-normal truncada (D005, F2 Tabla 10).

    Implementación:
        X = exp(µ_log + σ_log · Z)
    donde Z ~ N(0,1) estándar. Se aplica truncamiento por rechazo
    para mantener el dominio [0.25, 720.0] h. En la práctica la tasa
    de rechazo es mínima (<1 %) dado que los parámetros calibrados
    producen muy poco peso fuera del rango.
    """
    mu_log, sigma_log = ESTANCIA_LOG_NORMAL[area]
    while True:
        z = rng.standard_normal()
        valor = float(np.exp(mu_log + sigma_log * z))
        if ESTANCIA_MIN_H <= valor <= ESTANCIA_MAX_H:
            return round(valor, 4)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def generar_pacientes_lote(
    n_pacientes: int,
    tick_ingreso: int = 0,
    escenario: EscenariosValidos = "normal",
    rng: np.random.Generator | None = None,
) -> list[PacienteDict]:
    """Genera un lote de n_pacientes sintéticos para un tick dado.

    Parámetros
    ----------
    n_pacientes : int
        Cantidad de pacientes a generar. Debe ser ≥ 0.
    tick_ingreso : int
        Tick de simulación en que ingresan. Por defecto 0.
    escenario : {'normal', 'alta_demanda', 'crisis'}
        Solo se usa si se llama desde generar_llegadas_tick; aquí es
        informativo para el registro. No afecta las distribuciones
        del paciente individual (solo λ afecta el volumen).
    rng : np.random.Generator | None
        Generador de aleatoriedad. Si None, se crea uno con SEED=42
        (D_F3_001). Para simulaciones multi-tick se debe pasar el
        mismo rng para mantener secuencia reproducible.

    Retorna
    -------
    list[PacienteDict]
        Lista de dicts con todos los atributos de la entidad Paciente
        inicializados según F2. Campos calculados en valores neutros.

    Excepciones
    -----------
    ValueError
        Si n_pacientes < 0 o escenario no es válido.
    """
    if n_pacientes < 0:
        raise ValueError(f"n_pacientes debe ser ≥ 0, recibido: {n_pacientes}")
    if escenario not in LAMBDA_POR_ESCENARIO:
        raise ValueError(
            f"escenario '{escenario}' no válido. "
            f"Opciones: {list(LAMBDA_POR_ESCENARIO.keys())}"
        )
    if n_pacientes == 0:
        return []

    if rng is None:
        rng = np.random.default_rng(SEED)  # D_F3_001

    # Muestreo vectorizado donde es posible
    edades      = _generar_edad(rng, n_pacientes)
    prioridades = _generar_prioridades(rng, n_pacientes)

    pacientes: list[PacienteDict] = []

    for i in range(n_pacientes):
        prioridad = str(prioridades[i])
        area      = _generar_area_requerida(rng, prioridad)
        estancia  = _generar_estancia(rng, area)

        paciente: PacienteDict = {
            "id_paciente":              str(uuid.uuid4()),
            "edad":                     int(edades[i]),
            "prioridad_clinica":        prioridad,
            "area_requerida":           area,
            "tiempo_estancia_esperado": estancia,
            "tiempo_espera":            0.0,    # calculado por sistema experto F4-A
            "tiempo_en_sistema":        0.0,    # calculado por sistema experto F4-A
            "estado":                   "esperando",
            "cama_id":                  None,   # null al ingreso — F2 Tabla 5
            "tick_ingreso":             tick_ingreso,
            "es_desborde":              False,  # D007 — False al ingreso
        }
        pacientes.append(paciente)

    return pacientes


def generar_llegadas_tick(
    tick: int,
    escenario: EscenariosValidos = "normal",
    rng: np.random.Generator | None = None,
) -> list[PacienteDict]:
    """Simula las llegadas de pacientes en un tick usando Poisson(λ).

    El número de pacientes que llegan en el tick se muestrea desde
    Poisson(λ) donde λ depende del escenario (D012, F2 Tabla 8).
    Luego se llama a generar_pacientes_lote con el n resultante.

    Parámetros
    ----------
    tick : int
        Tick actual de la simulación (se registra en tick_ingreso).
    escenario : str
        Escenario de simulación. Determina λ.
    rng : np.random.Generator | None
        Generador compartido. Si None, se crea uno con SEED.

    Retorna
    -------
    list[PacienteDict]
        Lista de pacientes que llegan en este tick. Puede ser vacía.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    lam = LAMBDA_POR_ESCENARIO[escenario]
    n_llegadas = int(rng.poisson(lam))

    return generar_pacientes_lote(
        n_pacientes=n_llegadas,
        tick_ingreso=tick,
        escenario=escenario,
        rng=rng,
    )


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES DE SOPORTE PARA EDA (Entregable 2)
# ─────────────────────────────────────────────────────────────────────────────

def generar_lote_eda(
    n: int = 500,
    escenario: EscenariosValidos = "normal",
    seed: int = SEED,
) -> list[PacienteDict]:
    """Genera un lote fijo de n pacientes para el EDA.

    A diferencia de generar_llegadas_tick, aquí n es fijo (no Poisson)
    para que el EDA opere sobre una muestra de tamaño exacto y
    controlado. Crea su propio rng con la semilla indicada.

    Parámetros
    ----------
    n : int
        Tamaño del lote. Por defecto 500 (mínimo del criterio de F3).
    escenario : str
        Escenario de referencia (informativo).
    seed : int
        Semilla de reproducibilidad. Por defecto SEED=42 (D_F3_001).
    """
    rng = np.random.default_rng(seed)
    return generar_pacientes_lote(n_pacientes=n,
                                  tick_ingreso=0,
                                  escenario=escenario,
                                  rng=rng)


# ─────────────────────────────────────────────────────────────────────────────
# VALIDACIÓN INTERNA — ejecutar con  python generador_pacientes.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd

    print("=" * 60)
    print("VALIDACIÓN RÁPIDA — generador_pacientes.py")
    print(f"Semilla: SEED = {SEED}  (D_F3_001)")
    print("=" * 60)

    # 1. Lote fijo de 500 pacientes para EDA
    lote = generar_lote_eda(n=500)
    df = pd.DataFrame(lote)

    print(f"\n[1] Lote generado: {len(df)} pacientes")
    print(f"    Columnas: {list(df.columns)}")
    print(f"    Nulos totales: {df.isnull().sum().sum()} "
          f"(cama_id es null por diseño — no cuenta como error)")

    # 2. Proporciones de prioridad observadas vs esperadas
    print("\n[2] Proporciones de prioridad clínica:")
    obs = df["prioridad_clinica"].value_counts(normalize=True).sort_index()
    esp = dict(zip(PRIORIDADES, PROB_PRIORIDAD))
    for p in PRIORIDADES:
        print(f"    {p}  observado={obs.get(p, 0):.3f}  esperado={esp[p]:.3f}")

    # 3. Rango de edades
    print(f"\n[3] Edad: min={df['edad'].min()}  max={df['edad'].max()}  "
          f"media={df['edad'].mean():.1f}")
    fuera_edad = ((df["edad"] < EDAD_MIN) | (df["edad"] > EDAD_MAX)).sum()
    print(f"    Valores fuera de [0, 110]: {fuera_edad}")

    # 4. Rango de tiempo de estancia
    print(f"\n[4] Tiempo estancia (h): "
          f"min={df['tiempo_estancia_esperado'].min():.4f}  "
          f"max={df['tiempo_estancia_esperado'].max():.2f}")
    fuera_estancia = (
        (df["tiempo_estancia_esperado"] < ESTANCIA_MIN_H) |
        (df["tiempo_estancia_esperado"] > ESTANCIA_MAX_H)
    ).sum()
    print(f"    Valores fuera de [0.25, 720]: {fuera_estancia}")

    # 5. Distribución de área requerida
    print("\n[5] Distribución de área requerida:")
    print(df["area_requerida"].value_counts().to_string())

    # 6. Prueba de llegadas Poisson (5 ticks por escenario)
    print("\n[6] Llegadas Poisson — 5 ticks de prueba por escenario:")
    rng_prueba = np.random.default_rng(SEED)
    for esc in ["normal", "alta_demanda", "crisis"]:
        llegadas = [len(generar_llegadas_tick(t, esc, rng_prueba))
                    for t in range(5)]
        print(f"    {esc:15s}  λ={LAMBDA_POR_ESCENARIO[esc]}  "
              f"llegadas/tick={llegadas}  total={sum(llegadas)}")

    print("\n✔ Validación completada sin errores.")
