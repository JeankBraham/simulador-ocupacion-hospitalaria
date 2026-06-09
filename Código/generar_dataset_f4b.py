"""
generar_dataset_f4b.py
======================
Simulador Inteligente de Ocupacion Hospitalaria
Fase 4-B — Generacion del dataset de entrenamiento para el modelo predictivo

Autor  : Juan Camilo Garcia Braham
Curso  : IA en Salud · Maestria IA y CD · UTP
Anno   : 2026
Stack  : Python 3.12 · numpy 2.4 · pandas 3.0 · scikit-learn 1.8

Decisiones de diseno activas en este modulo
---------------------------------------------
D_F4B_001  Variable objetivo: O_{t+4} (componente O del indicador I)
D_F4B_002  Horizonte: T+4 ticks (1 hora)
D_F4B_003  Features extraidas de ResultadoTick sin modificar sistema_experto.py
D_F4B_004  Dataset generado mediante simulaciones multi-escenario con semillas distintas
D_F4B_005  4 lags de O incluidos como features (O_t-1..O_t-4)

Parametros de simulacion
-------------------------
TICKS_POR_SIM   : 200 ticks por simulacion (50 horas)
SEMILLAS        : [42, 123, 456, 789] — 4 semillas por escenario
ESCENARIOS      : ['normal', 'alta_demanda', 'crisis']
Total filas raw : ~(200-4) * 3 * 4 = 2.352 filas utiles
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from generador_pacientes import generar_llegadas_tick, LAMBDA_POR_ESCENARIO
from sistema_experto import SistemaExperto, crear_hospital_referencia

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DE SIMULACION
# ─────────────────────────────────────────────────────────────────────────────

TICKS_POR_SIM: int = 200          # 50 horas de simulacion
HORIZONTE: int = 4                # D_F4B_002 — T+4 ticks = 1 hora
N_LAGS: int = 4                   # D_F4B_005 — lags de O
ESCENARIOS: list[str] = ["normal", "alta_demanda", "crisis"]
SEMILLAS: list[int] = [42, 123, 456, 789]   # D_F4B_004


def ejecutar_simulacion(
    escenario: str,
    seed: int,
) -> pd.DataFrame:
    """Ejecuta una simulacion completa y retorna el historial tick a tick.

    Retorna
    -------
    DataFrame con columnas:
        tick, escenario, seed,
        O_t, E_t, P_t, C_t, I_t,
        llegadas_t, altas_t, traslados_t, cola_espera_t
    Una fila por tick procesado.
    """
    rng_sim = np.random.default_rng(seed)
    rng_gen = np.random.default_rng(seed + 1000)   # RNG separado para generador

    hospital = crear_hospital_referencia(rng=rng_sim)
    se = SistemaExperto(hospital=hospital)

    historial: list[dict] = []

    for tick in range(TICKS_POR_SIM):
        # Generar llegadas del tick actual
        nuevos = generar_llegadas_tick(tick, escenario=escenario, rng=rng_gen)

        # Procesar tick en modo automatico (D_F4B_003 — no modifica SE)
        resultado = se.procesar_tick(tick, nuevos_pacientes=nuevos, modo_asistido=False)

        # Contar acciones del tick
        altas_t = sum(1 for a in resultado.acciones_ejecutadas if a.tipo == "alta")
        traslados_t = sum(1 for a in resultado.acciones_ejecutadas if a.tipo in ("traslado", "desborde"))
        cola_t = len(hospital.pacientes_esperando())

        historial.append({
            "tick":          tick,
            "escenario":     escenario,
            "seed":          seed,
            "O_t":           resultado.componente_O,
            "E_t":           resultado.componente_E,
            "P_t":           resultado.componente_P,
            "C_t":           resultado.componente_C,
            "I_t":           resultado.indicador_I,
            "llegadas_t":    len(nuevos),
            "altas_t":       altas_t,
            "traslados_t":   traslados_t,
            "cola_espera_t": cola_t,
        })

    return pd.DataFrame(historial)


def construir_features(df_sim: pd.DataFrame) -> pd.DataFrame:
    """Construye el dataset de features y target a partir del historial.

    Para cada tick T (valido: T >= N_LAGS y T+HORIZONTE < TICKS_POR_SIM):
        - Features: estado en tick T + lags O_{t-1}..O_{t-4} + escenario one-hot
        - Target: O_{t+HORIZONTE}

    El one-hot del escenario usa drop_first=False para mantener legibilidad.
    Columnas dummy: escenario_alta_demanda, escenario_crisis
    (escenario_normal es la categoria de referencia, se elimina)

    D_F4B_005 — lags incluidos: O_{t-1}, O_{t-2}, O_{t-3}, O_{t-4}
    """
    registros: list[dict] = []

    # Trabajar por (escenario, seed) para calcular lags dentro de cada sim
    for (esc, seed), grupo in df_sim.groupby(["escenario", "seed"]):
        grupo = grupo.sort_values("tick").reset_index(drop=True)

        for idx in range(N_LAGS, len(grupo) - HORIZONTE):
            fila_actual = grupo.iloc[idx]
            fila_target = grupo.iloc[idx + HORIZONTE]

            reg: dict = {
                "tick":          int(fila_actual["tick"]),
                "escenario":     esc,
                "seed":          seed,
                # Features del estado en T
                "O_t":           fila_actual["O_t"],
                "E_t":           fila_actual["E_t"],
                "P_t":           fila_actual["P_t"],
                "C_t":           fila_actual["C_t"],
                "I_t":           fila_actual["I_t"],
                "llegadas_t":    fila_actual["llegadas_t"],
                "altas_t":       fila_actual["altas_t"],
                "traslados_t":   fila_actual["traslados_t"],
                "cola_espera_t": fila_actual["cola_espera_t"],
                # Lags D_F4B_005
                "O_lag1":        grupo.iloc[idx - 1]["O_t"],
                "O_lag2":        grupo.iloc[idx - 2]["O_t"],
                "O_lag3":        grupo.iloc[idx - 3]["O_t"],
                "O_lag4":        grupo.iloc[idx - 4]["O_t"],
                # Target D_F4B_001
                "target_O_t4":   fila_target["O_t"],
            }
            registros.append(reg)

    df_features = pd.DataFrame(registros)

    # One-hot del escenario — escenario_normal es categoria base (eliminada)
    dummies = pd.get_dummies(df_features["escenario"], prefix="escenario", dtype=float)
    if "escenario_normal" in dummies.columns:
        dummies = dummies.drop(columns=["escenario_normal"])
    df_features = pd.concat([df_features, dummies], axis=1)

    return df_features


def generar_dataset(ruta_salida: str = "dataset_f4b.parquet",
                    verbose: bool = True) -> pd.DataFrame:
    """Pipeline completo: simular → construir features → guardar.

    Retorna el DataFrame final listo para entrenamiento.
    """
    if verbose:
        print("=" * 60)
        print("GENERACION DATASET F4-B")
        print(f"Escenarios: {ESCENARIOS}")
        print(f"Semillas:   {SEMILLAS}")
        print(f"Ticks/sim:  {TICKS_POR_SIM}  |  Horizonte: T+{HORIZONTE}")
        print("=" * 60)

    frames_sim: list[pd.DataFrame] = []

    for esc in ESCENARIOS:
        for seed in SEMILLAS:
            if verbose:
                print(f"  Simulando  escenario={esc:15s}  seed={seed} ...", end=" ")
            df_sim = ejecutar_simulacion(esc, seed)
            frames_sim.append(df_sim)
            if verbose:
                print(f"OK ({len(df_sim)} ticks)")

    df_historial = pd.concat(frames_sim, ignore_index=True)
    df_dataset   = construir_features(df_historial)

    if verbose:
        print(f"\nDataset final: {len(df_dataset)} filas x {len(df_dataset.columns)} columnas")
        print(f"  Target (O_t+4): media={df_dataset['target_O_t4'].mean():.2f}  "
              f"std={df_dataset['target_O_t4'].std():.2f}  "
              f"min={df_dataset['target_O_t4'].min():.2f}  "
              f"max={df_dataset['target_O_t4'].max():.2f}")
        print(f"  Distribucion por escenario:")
        for esc in ESCENARIOS:
            n = (df_dataset["escenario"] == esc).sum()
            media = df_dataset.loc[df_dataset["escenario"] == esc, "target_O_t4"].mean()
            print(f"    {esc:15s}: {n} filas  |  target media={media:.2f}%")

    # Guardar en parquet (compacto, preserva dtypes)
    df_dataset.to_parquet(ruta_salida, index=False)
    if verbose:
        print(f"\nDataset guardado en: {ruta_salida}")

    return df_dataset


if __name__ == "__main__":
    df = generar_dataset(ruta_salida="/home/claude/dataset_f4b.parquet", verbose=True)
    print("\nColumnas del dataset:")
    for c in df.columns:
        print(f"  {c}")
