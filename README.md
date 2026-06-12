# Simulador Inteligente de Ocupación Hospitalaria

**Prototipo académico funcional** desarrollado para el curso *IA en Salud*  
Maestría en IA y Ciencia de Datos · Universidad Tecnológica de Pereira (UTP) · 2026  
Autor: **Juan Camilo García Braham**  
Framework: **CRISP-DM/S** (CRISP-DM + SEMMA + DAMA + MLOps)

---

## Descripción

Sistema de simulación hospitalaria que modela el comportamiento dinámico de un hospital colombiano durante hasta **50 horas simuladas** (200 ticks de 15 minutos). Integra:

- **Generador de pacientes sintéticos** con distribuciones probabilísticas calibradas (Poisson, Log-normal, Multinomial)
- **Sistema experto de asignación** con 19 reglas en 6 capas, modos automático y asistido, alineado con Resolución 5596/2015
- **Indicador compuesto I** = 0.4·O + 0.2·E + 0.2·P + 0.2·C con cuatro niveles (Bajo / Medio / Alto / Crítico)
- **Modelo predictivo** de ocupación a 1 hora (LinearRegression, RMSE = 2.64 pp)
- **Interfaz web Streamlit** con grid interactivo de camas, modo asistido, gráficas en tiempo real y pantalla de resultados

> Todos los datos son **100% sintéticos**. El sistema no procesa datos reales de pacientes.

---

## Estructura del repositorio

```
simulador_f6/
├── app.py                              # Punto de entrada — interfaz web Streamlit
├── motor_simulacion.py                 # Motor tick a tick (F6)
├── sistema_experto.py                  # Sistema experto, entidades y motor de dominio (F4-A)
├── generador_pacientes.py              # Generador de pacientes sintéticos (F3)
├── modelo_final_f4b.pkl                # Modelo predictivo serializado (F4-B)
│
├── pruebas_integracion_f6.py           # Suite de pruebas de integración (18/18 PASS)
├── generar_manual_usuario_f6.py        # Genera manual_usuario_f6.pdf
├── generar_documentacion_tecnica_f6.py # Genera documentacion_tecnica_f6.pdf
├── generar_informe_final_f6.py         # Genera informe_final_f6.pdf
│
└── docs/                               # PDFs de entregables por fase
    ├── Fase_2_Comprension_dato.pdf
    ├── Fase_4A_Sistema_experto.pdf
    ├── Fase_4B_Modelo_predictivo.pdf
    ├── Fase_5_Evaluacion.pdf
    ├── pruebas_integracion_f6.pdf
    ├── manual_usuario_f6.pdf
    ├── documentacion_tecnica_f6.pdf
    └── informe_final_f6.pdf
```

---

## Instalación

### Requisitos

- Python 3.10 o superior (probado con Python 3.12)
- pip o [uv](https://github.com/astral-sh/uv)

### Dependencias

```bash
# Con pip
pip install streamlit==1.45.1 numpy==2.4 pandas==3.0 \
            matplotlib==3.10 scikit-learn==1.8 joblib==1.5 \
            reportlab==4.4 seaborn==0.13

# Con uv (más rápido)
uv pip install streamlit==1.45.1 numpy==2.4 pandas==3.0 \
               matplotlib==3.10 scikit-learn==1.8 joblib==1.5 \
               reportlab==4.4 seaborn==0.13
```

### Verificar instalación

```bash
python -c "
import streamlit, numpy, pandas, matplotlib, sklearn, joblib
import sistema_experto, generador_pacientes, motor_simulacion
joblib.load('modelo_final_f4b.pkl')
print('Entorno OK')
"
```

---

## Uso

### Lanzar el simulador

```bash
streamlit run app.py
```

La aplicación se abre automáticamente en `http://localhost:8501`.

**Flujo de uso:**
1. **Configurar hospital** — define camas por área (UCI, Urgencias, Hospitalización, Observación, Sala de espera), escenario, modo y velocidad
2. **Simulador en curso** — observa el grid de camas en tiempo real; haz clic en una cama roja para ver la ficha del paciente
3. **Resultados** — métricas de régimen estable con verificación CE-B al completar los 200 ticks

### Escenarios disponibles

| Escenario     | λ (pac/tick) | Ocupación esperada | Modo recomendado |
|---------------|-------------:|--------------------|------------------|
| Normal        | 1.5          | 35 % – 55 %        | Automático       |
| Alta demanda  | 3.0          | 60 % – 88 %        | Automático       |
| Crisis        | 5.0          | 80 % – 100 %       | **Asistido**     |

### Reproducibilidad

Con la misma semilla, escenario y configuración los resultados son idénticos:

```python
import joblib
from motor_simulacion import crear_estado, avanzar_tick, calcular_resumen

modelo = joblib.load('modelo_final_f4b.pkl')
estado = crear_estado('normal', seed=99, ticks_total=200)
for _ in range(200):
    avanzar_tick(estado, modelo)
r = calcular_resumen(estado)
print(f"O_media = {r['O_media']:.1f}%")  # → 43.1%
```

---

## Pruebas

### Pruebas unitarias del sistema experto (F4-A)

```bash
python test_sistema_experto.py
# Total: 33 | Pasadas: 33 | Fallidas: 0
```

### Pruebas de integración (F6)

```bash
python pruebas_integracion_f6.py
# RESULTADO GLOBAL: 18/18 checks PASS
# PDF generado: pruebas_integracion_f6.pdf
```

---

## Resultados de evaluación

Evaluados con SEED=99, 200 ticks, warm-up de 20 ticks (Fase 5):

| Métrica              | Normal (E1) | Alta demanda (E2) | Crisis (E3) |
|----------------------|:-----------:|:-----------------:|:-----------:|
| O media              | 43.2 %      | 61.8 %            | 80.6 %      |
| I media              | 19.6        | 37.9              | 59.4        |
| Nivel I modal        | Bajo        | Medio             | Crítico     |
| Cola máxima (pac.)   | 0           | 4                 | 27          |
| Traslados totales    | 38          | 271               | 504         |
| Alertas RES-04       | 0           | 188               | 2 563       |
| RMSE predicción      | 2.689 pp    | 2.799 pp          | 2.183 pp    |
| CE-B rango O         | ✓           | ✓                 | ✓           |
| CE-B nivel I         | ✓           | ✓                 | ✓           |

---

## Regenerar documentación

```bash
python pruebas_integracion_f6.py          # Pruebas de integración + PDF
python generar_manual_usuario_f6.py       # Manual de usuario
python generar_documentacion_tecnica_f6.py # Documentación técnica
python generar_informe_final_f6.py        # Informe final del proyecto
```

---

## Arquitectura

```
Capa de presentación      app.py              (Streamlit — UI)
        |
Capa de motor             motor_simulacion.py  (loop tick a tick, estado)
        |
Capa de dominio           sistema_experto.py   (19 reglas, Hospital, SE)
                          generador_pacientes.py (Poisson, Log-normal...)
        |
Capa de predicción        modelo_final_f4b.pkl (StandardScaler + LinearRegression)
```

Las dependencias son estrictamente descendentes. Los módulos base (`sistema_experto.py`, `generador_pacientes.py`, `modelo_final_f4b.pkl`) no fueron modificados en F6.

---

## Ciclo de vida CRISP-DM/S

| Fase | SEMMA    | Contenido                                              | Estado     |
|------|----------|--------------------------------------------------------|------------|
| F1   | —        | Marco conceptual, criterios CE-A/B/C/D                | ✓ Completo |
| F2   | Sample   | Diccionario de datos, distribuciones, modelo entidades | ✓ Completo |
| F3   | Explore + Modify | Generador sintético, EDA, calibración         | ✓ Completo |
| F4-A | Model    | Sistema experto, 19 reglas, 33 pruebas unitarias       | ✓ Completo |
| F4-B | Model    | Modelo predictivo, RMSE=2.64 pp, modelo_final_f4b.pkl  | ✓ Completo |
| F5   | Assess   | 3 escenarios evaluados, R03 mitigado                   | ✓ Completo |
| F6   | —        | Despliegue web, pruebas integración, documentación     | ✓ Completo |

---

## Marco normativo

El sistema experto está alineado con:

- **Resolución 5596/2015** — Sistema de clasificación/triage en urgencias (prioridades P1–P4)
- **Resolución 3100/2019** — Condiciones de habilitación de servicios de salud
- **CONPES 3975/2019** — Política nacional de transformación digital e IA

Para los requisitos de un despliegue productivo real (datos reales, interoperabilidad HIS/EHR, protección de datos) ver la Sección 6 del informe final (`informe_final_f6.pdf`).

---

## Trabajo futuro

| ID    | Mejora                                     | Prioridad  |
|-------|--------------------------------------------|------------|
| TF-01 | Configuración dinámica de pisos y áreas    | Media      |
| TF-02 | Lambda variable con patrón diurno/nocturno | Alta       |
| TF-03 | Calibración con datos históricos reales    | Alta       |
| TF-04 | Guardado y carga de sesiones               | Media      |
| TF-05 | Modelo predictivo no lineal                | Baja       |
| TF-06 | Despliegue productivo multi-usuario        | Alta (v2.0)|
| TF-07 | Integración con HIS/EHR (HL7 FHIR)        | Alta (v2.0)|

---

## Licencia

Proyecto académico — Universidad Tecnológica de Pereira, 2026.  
Uso exclusivo para fines educativos e investigativos.