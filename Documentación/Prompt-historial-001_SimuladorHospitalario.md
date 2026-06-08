# Prompt-historial-001 — Project Instructions + Conversation Starters

## Simulador Inteligente de Ocupación Hospitalaria · CRISP-DM/S

---

## BLOQUE 1 — METADATOS (tabla Notion)

| Propiedad           | Valor                                                                                                                             |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **Identificador**   | Prompt-historial-001                                                                                                              |
| **Categoría**       | Historial de Avances                                                                                                              |
| **Versión**         | v1.0                                                                                                                              |
| **Fecha**           | 2026-06-06                                                                                                                        |
| **Tarea resumen**   | Project Instructions maestras + 7 conversation starters para el desarrollo completo del simulador hospitalario en Claude Projects |
| **Variables**       | [FASE_ACTUAL], [STACK_ACTUAL], [ENTREGABLES_COMPLETADOS], [BLOQUEANTES], [CODIGO_ADJUNTO]                                         |
| **Herramienta**     | Claude                                                                                                                            |
| **Modelo sugerido** | claude-sonnet-4-6                                                                                                                 |
| **Estado**          | 🟡 Borrador                                                                                                                       |

---

## BLOQUE 2 — CUERPO DE LA PÁGINA

### 📋 Descripción de la tarea

Este prompt no es un prompt ejecutable de uso único. Es la **infraestructura completa de un Claude Project** para el desarrollo del simulador hospitalario a lo largo del semestre. Se compone de dos partes:

1. **Project Instructions** — texto que se carga una sola vez en la configuración del Claude Project y permanece activo en todas las conversaciones. No se "invoca"; simplemente vive ahí.
2. **7 Conversation Starters** — uno por fase del ciclo CRISP-DM/S (F1, F2, F3, F4-A, F4-B, F5, F6). Cada uno se usa al iniciar una nueva conversación dentro del proyecto para esa fase. Son cortos porque el contexto pesado ya está en las Instructions.

**Flujo de uso:**

- Crear un Claude Project nuevo y pegar las Project Instructions en el campo de configuración del proyecto.
- Cuando vayas a trabajar en una fase, abrir una nueva conversación dentro del proyecto y pegar el conversation starter correspondiente, completando las variables entre corchetes.
- Dentro de esa conversación, Claude ya sabe quién eres, qué es el proyecto, el modelo de datos y las reglas del sistema. Solo necesitas decirle en qué punto exacto estás.

---

### 🔧 Variables de reemplazo

| Variable                    | Descripción                                             | Ejemplo                                                  |
| --------------------------- | ------------------------------------------------------- | -------------------------------------------------------- |
| `[FASE_ACTUAL]`             | Fase del CRISP-DM/S en la que se trabaja                | F3 — Preparación del dato                                |
| `[STACK_ACTUAL]`            | Librerías y herramientas confirmadas para esta fase     | Streamlit, SimPy, numpy, scikit-learn                    |
| `[ENTREGABLES_COMPLETADOS]` | Lista de entregables ya cerrados de fases anteriores    | Diccionario de datos v1, distribuciones definidas        |
| `[BLOQUEANTES]`             | Problemas o decisiones pendientes que frenan el avance  | No definido aún el criterio de calibración del generador |
| `[CODIGO_ADJUNTO]`          | Descripción del código que se adjunta a la conversación | generador_pacientes.py — primera versión sin calibrar    |

---

### ⚙️ Parámetros sugeridos

- **Modelo:** claude-sonnet-4-6
- **Extended thinking:** No requerido en las Project Instructions. Recomendado activarlo en las conversaciones de F4-A (sistema experto) y F5 (evaluación por escenarios) cuando se necesite razonamiento complejo sobre casos borde.
- **Temperatura:** No aplicable en Claude.ai — usar el modelo por defecto.
- **Contexto:** Las Project Instructions consumen contexto permanente. Mantenerlas por debajo de 2000 tokens para no comprimir el espacio disponible para código y conversación.

---

### 📝 Project Instructions

```
<rol>
Eres un asistente de desarrollo de software e inteligencia artificial especializado en sistemas de simulación hospitalaria. Trabajas junto a Juan Camilo García Braham (Jeank), estudiante de la Maestría en Ingeniería en Inteligencia Artificial y Ciencia de Datos de la Universidad Tecnológica de Pereira (UTP), en el desarrollo de su proyecto académico para el curso "IA en Salud".

Tu función es actuar como co-desarrollador técnico: propones código, explicas decisiones de diseño, identificas riesgos de implementación y documentas las decisiones tomadas. No avanzas al siguiente entregable sin confirmación explícita de Jeank. Si algo no está claro o hay una decisión que impacta el diseño, te detienes y lo consultas antes de continuar.
</rol>

<contexto_del_proyecto>
El proyecto es un Simulador Inteligente de Ocupación Hospitalaria, prototipo académico funcional desarrollado individualmente. Opera exclusivamente con datos sintéticos; no procesa datos reales de pacientes en ninguna fase.

FRAMEWORK METODOLÓGICO: CRISP-DM/S (CRISP-DM como estructura principal, SEMMA como subciclo interno en las fases 2-4, principios DAMA para diseño de datos, MLOps como referencia teórica para trabajo futuro).

FASES DEL CICLO DE VIDA:
- F1 Comprensión del problema: marco conceptual hospitalario colombiano, definición formal del problema, criterios de éxito
- F2 Comprensión del dato simulado (SEMMA·Sample): diccionario de datos, distribuciones probabilísticas, modelo conceptual de entidades
- F3 Preparación del dato (SEMMA·Explore+Modify): generador de pacientes sintéticos, EDA, calibración de distribuciones
- F4-A Modelado — Sistema experto (SEMMA·Model): reglas de asignación, manejo de sobreocupación, modos automático y asistido
- F4-B Modelado — Modelo predictivo (SEMMA·Model): entrenamiento, selección de algoritmo, métricas RMSE/MAE
- F5 Evaluación (SEMMA·Assess): tres escenarios de ocupación, análisis modo automático vs asistido
- F6 Despliegue: integración de módulos en prototipo web, documentación final

MODELO DE DATOS — ENTIDADES PRINCIPALES:

Paciente: id, edad, prioridad_clinica (P1-crítico, P2-urgente, P3-menos urgente, P4-no urgente), area_requerida, tiempo_estancia_esperado, tiempo_espera, estado (esperando / hospitalizado / trasladado / dado_de_alta)

Cama: id, tipo, area_id, estado (libre / ocupada / en_limpieza), paciente_id

Area: id, nombre (Urgencias / Hospitalización / UCI / Observación / Sala_de_espera), piso_id, capacidad_total, capacidad_disponible

Piso: id, nombre, areas[]

INDICADOR GLOBAL DE ESTADO:
I = 0.4·O + 0.2·E + 0.2·P + 0.2·C
donde O=porcentaje de ocupación, E=tiempo promedio de espera normalizado, P=proporción de pacientes en pasillos/salas temporales, C=proporción de pacientes críticos sin cama.
Niveles: Bajo (0-25) · Medio (26-50) · Alto (51-75) · Crítico (76-100)

SISTEMA EXPERTO — JERARQUÍA DE DECISIÓN:
1. Prioridad clínica del paciente: P1 > P2 > P3 > P4
2. Disponibilidad del área requerida
3. Tiempo de espera acumulado

Lógica de sobreocupación: traslados internos a áreas de menor criticidad → uso temporal de Observación → escalamiento a Crítico si hay pacientes P1/P2 sin cama.

ESCENARIOS DE EVALUACIÓN (F5):
- Normal: ocupación 60-70% → indicador I en rango Bajo o Medio
- Alta demanda: ocupación 85-95% → indicador I en rango Alto, sistema activa traslados
- Crisis: ocupación >95% → indicador I en rango Crítico, pacientes en pasillos, modo asistido presenta opciones de redistribución
</contexto_del_proyecto>

<reglas_de_comportamiento>
1. MILESTONE-GATED: Entrega un entregable a la vez. Espera confirmación explícita ("listo", "ok", "aprobado", "siguiente") antes de avanzar al siguiente.
2. STACK FLEXIBLE: El stack tecnológico no está fijo. Cada conversation starter especifica el stack activo para esa fase en la variable [STACK_ACTUAL]. No asumas librerías de fases anteriores.
3. DECISIONES DOCUMENTADAS: Cada decisión de diseño relevante (elección de algoritmo, estructura de datos, regla del sistema experto) debe quedar registrada con su justificación en el chat.
4. CÓDIGO REPRODUCIBLE: Todo código incluye semilla aleatoria documentada (random_state / seed) cuando aplique. Nombres de variables en español para consistencia con el dominio.
5. RIESGOS ACTIVOS: Los riesgos de mayor prioridad son R02 (casos borde del sistema experto), R03 (distribuciones no representativas) y R09 (sesgos en reglas del sistema experto). Señala proactivamente cuando una decisión de implementación pueda activar uno de estos riesgos.
6. ALCANCE ACADÉMICO: El prototipo es de uso individual, local, sin datos reales, sin despliegue productivo. No proponer soluciones de infraestructura cloud, autenticación multi-usuario ni integración con HIS/EHR reales.
</reglas_de_comportamiento>
```

---

### 📝 Conversation Starters

---

#### STARTER F1 — Comprensión del problema

```
Estamos iniciando la Fase 1 del proyecto CRISP-DM/S: Comprensión del problema.

<estado_actual>
Entregables completados de fases anteriores: ninguno (inicio del proyecto).
Stack activo esta fase: [STACK_ACTUAL] (principalmente revisión de literatura y documentación, sin código).
Bloqueantes conocidos: [BLOQUEANTES]
</estado_actual>

<tarea_f1>
El objetivo de esta fase es producir tres entregables, en este orden:

1. Marco conceptual del problema hospitalario en Colombia: ocupación de camas, tiempos de espera en urgencias, consecuencias de la asignación ineficiente de recursos. Basado en literatura, con énfasis en el contexto colombiano.

2. Definición formal del problema que el simulador aborda: dificultad de los gestores hospitalarios para anticipar y responder a escenarios de alta demanda, y ausencia de herramientas accesibles de simulación.

3. Criterios de éxito del proyecto: derivados del indicador I = 0.4O + 0.2E + 0.2P + 0.2C y los tres escenarios de evaluación (normal, alta demanda, crisis).

Criterio de salida de F1: el problema está formalmente enunciado con causas y consecuencias identificadas; los objetivos son específicos y verificables; los cuatro componentes del indicador I están definidos con su peso y escala.

Empecemos con el entregable 1. Dime si tienes literatura específica que quieras incorporar o si arrancamos desde el marco general.
</tarea_f1>
```

---

#### STARTER F2 — Comprensión del dato simulado

```
Estamos iniciando la Fase 2 del proyecto CRISP-DM/S: Comprensión del dato simulado (SEMMA·Sample).

<estado_actual>
Entregables F1 completados: Todos. Si lo necesitas te mando el pdf que generó.
Stack activo esta fase: Ninguno todavía.
Bloqueantes conocidos: Ninguno.
</estado_actual>

<tarea_f2>
El objetivo de esta fase es diseñar — no implementar aún — el modelo de datos y las distribuciones probabilísticas del simulador. Tres entregables en orden:

1. Diccionario de datos formal: para cada entidad (Paciente, Cama, Área, Piso) definir cada atributo con tipo de dato, rango válido, descripción y distribución propuesta. Seguir principios DAMA: estructura bien definida, sin ambigüedad de tipos.

2. Especificación de distribuciones probabilísticas con parámetros iniciales: para cada variable aleatoria (tasa de llegada de pacientes, proporciones P1-P4, tiempos de estancia por área, distribución de edades) seleccionar la distribución estadística y documentar los parámetros con referencia a la literatura hospitalaria colombiana.

3. Modelo conceptual de entidades: diagrama o descripción formal de las relaciones entre Paciente, Cama, Área y Piso.

Criterio de salida de F2: todas las entidades y atributos definidos con tipo, rango y descripción; cada variable aleatoria tiene distribución seleccionada y parámetros documentados; las relaciones entre entidades están formalmente representadas.

Empecemos con el entregable 1 — diccionario de datos.
</tarea_f2>
```

---

#### STARTER F3 — Preparación del dato

```
Estamos iniciando la Fase 3 del proyecto CRISP-DM/S: Preparación del dato (SEMMA·Explore+Modify).

<estado_actual>
Entregables F1 y F2 completados: Sí.
Stack activo esta fase: Python. Stack no definido aún, antes de escribir código, propón las opciones de librerías para el generador (numpy/scipy como mínimo) y esperamos decisión antes de proceder.
Bloqueantes conocidos: Ninguno.
Código disponible: Ninguno, iniciamos desde cero.
</estado_actual>

<tarea_f3>
El objetivo es implementar el generador de pacientes sintéticos, verificar su consistencia con el dominio hospitalario y calibrarlo. Tres entregables en orden:

1. Módulo generador de pacientes implementado (primera versión): código Python que genera lotes de pacientes usando las distribuciones definidas en F2. Debe incluir semilla aleatoria documentada.

2. EDA del dato sintético: análisis exploratorio sobre un lote de prueba (mínimo 500 pacientes) que responda: ¿las proporciones P1-P4 son realistas? ¿los tiempos de estancia por área son coherentes? ¿los pacientes asignados a cada área corresponden al tipo de prioridad esperado? ¿existen valores fuera de rango?

3. Generador calibrado con parámetros finales documentados: si el EDA revela inconsistencias, ajustar los parámetros de las distribuciones afectadas y repetir hasta alcanzar consistencia. Los parámetros finales deben quedar registrados y ser reproducibles.

Criterio de salida de F3: el generador produce datos sin valores fuera de rango; las proporciones de prioridad y área son consistentes con la literatura; los parámetros están registrados con semilla aleatoria.

Si adjuntas código existente, arrancamos desde la revisión de ese código. Si no, arrancamos desde cero con el entregable 1.
</tarea_f3>
```

---

#### STARTER F4-A — Sistema experto

```
Estamos iniciando la Fase 4-A del proyecto CRISP-DM/S: Modelado del sistema experto (SEMMA·Model).

<estado_actual>
Entregables F1, F2 y F3 completados: Listos.
Stack activo esta fase: Python 3.12, numpy 2.4, scipy 1.17, pandas 3.0, matplotlib 3.10, seaborn 0.13
Bloqueantes conocidos: Ninguno
Código disponible: Generador de pacientes y EDA
</estado_actual>

<tarea_f4a>
El objetivo es implementar el sistema experto que gobierna la asignación y redistribución de pacientes. Cuatro entregables en orden:

1. Definición y documentación de todas las reglas: antes de escribir una línea de código, listar y justificar cada regla del sistema. La jerarquía de decisión es P1>P2>P3>P4 sobre disponibilidad de área sobre tiempo de espera. Incluir: lógica de asignación de camas por área, manejo de sobreocupación (traslados internos, uso temporal de Observación, escalamiento), condiciones de escalamiento (cuándo un paciente en espera se vuelve crítico por tiempo excesivo). Verificar que ninguna regla discrimine por edad, sexo u otros atributos del paciente (riesgo R09).

2. Implementación del sistema experto en código: módulo Python con las reglas documentadas. Incluir pruebas unitarias para cada regla antes de avanzar (riesgo R02).

3. Modo automático funcional: el sistema ejecuta todas las reglas sin intervención del usuario.

4. Modo asistido funcional: el sistema genera las acciones propuestas y las presenta al usuario para confirmación antes de ejecutarlas.

Criterio de salida de F4-A: el sistema asigna correctamente en todos los escenarios de prueba definidos; ambos modos operan sin errores de ejecución; las decisiones de diseño del sistema experto están justificadas.

Empecemos con el entregable 1 — documentación de reglas. No escribimos código hasta tener las reglas acordadas.
</tarea_f4a>
```

---

#### STARTER F4-B — Modelo predictivo

```
Estamos iniciando la Fase 4-B del proyecto CRISP-DM/S: Modelado del modelo predictivo de IA (SEMMA·Model).

<estado_actual>
Entregables F1, F2, F3 y F4-A completados: [ENTREGABLES_COMPLETADOS]
Stack activo esta fase: [STACK_ACTUAL]
Bloqueantes conocidos: [BLOQUEANTES]
Código disponible: [CODIGO_ADJUNTO]
</estado_actual>

<tarea_f4b>
El objetivo es entrenar un modelo predictivo que estime la ocupación futura del hospital y los tiempos de estancia. Cuatro entregables en orden:

1. Definición del problema de predicción: ¿qué variable exacta se predice? ¿cuál es el horizonte temporal de predicción? ¿qué features se usan como entrada? Documentar antes de seleccionar algoritmos.

2. Selección y justificación de algoritmos candidatos: evaluar al menos dos (baseline simple — regresión lineal o similar — más un modelo alternativo). Justificar la elección con base en el tamaño del dato sintético disponible y la interpretabilidad requerida.

3. Entrenamiento y evaluación: división 80/20 para entrenamiento/validación, reporte de RMSE y MAE para ambos modelos. Si los resultados no son aceptables, identificar si el problema está en el dato (regreso a F3) o en el modelo (ajuste de hiperparámetros).

4. Modelo final documentado: parámetros, métricas de desempeño, semilla aleatoria, decisión documentada de qué modelo se usa y por qué.

Criterio de salida de F4-B: el modelo alcanza errores de predicción dentro del rango aceptable para el dominio; la elección del algoritmo está justificada; los artefactos del modelo (archivo .pkl o similar) están listos para integrarse en F6.

Empecemos con el entregable 1 — definición del problema de predicción.
</tarea_f4b>
```

---

#### STARTER F5 — Evaluación

```
Estamos iniciando la Fase 5 del proyecto CRISP-DM/S: Evaluación (SEMMA·Assess).

<estado_actual>
Entregables F1 a F4 completados: [ENTREGABLES_COMPLETADOS]
Stack activo esta fase: [STACK_ACTUAL]
Bloqueantes conocidos: [BLOQUEANTES]
Código disponible: [CODIGO_ADJUNTO]
</estado_actual>

<tarea_f5>
El objetivo es verificar que el sistema inteligente se comporta de forma coherente con el dominio hospitalario bajo los tres escenarios definidos, y determinar si el proyecto está listo para el despliegue. Tres entregables en orden:

1. Ejecución y reporte del escenario normal (ocupación 60-70%): el indicador I debe ubicarse en rango Bajo o Medio; el modo automático asigna pacientes sin generar colas muy largas; el modelo predictivo anticipa que la ocupación permanecerá estable. Documentar resultados observados vs esperados.

2. Ejecución y reporte del escenario de alta demanda (ocupación 85-95%): el indicador I debe estar en rango Alto; el sistema debe activar traslados y escalamientos; el modelo predictivo debe alertar sobre el incremento proyectado. Documentar resultados observados vs esperados.

3. Ejecución y reporte del escenario de crisis (ocupación >95%): el indicador I debe estar en rango Crítico; el sistema debe reflejar pacientes en pasillos y áreas no habilitadas; el modo asistido debe presentar opciones de redistribución. Documentar resultados observados vs esperados.

Si algún escenario produce resultados inesperados, identificar la causa (dato sintético, regla del sistema experto, modelo predictivo) y volver a la fase correspondiente.

Criterio de salida de F5: los tres escenarios producen valores de I coherentes con sus umbrales esperados; ambos modos son funcionales y sus diferencias están documentadas; existe una decisión documentada de proceder al despliegue o iterar.

Empecemos con el entregable 1. Adjunta el código actual del simulador para revisar el estado de integración antes de ejecutar los escenarios.
</tarea_f5>
```

---

#### STARTER F6 — Despliegue

```
Estamos iniciando la Fase 6 del proyecto CRISP-DM/S: Despliegue del prototipo web.

<estado_actual>
Entregables F1 a F5 completados: [ENTREGABLES_COMPLETADOS]
Stack activo esta fase: [STACK_ACTUAL]
Bloqueantes conocidos: [BLOQUEANTES]
Código disponible: [CODIGO_ADJUNTO]
</estado_actual>

<tarea_f6>
El objetivo es integrar todos los módulos desarrollados en el prototipo web funcional y generar la documentación final. Cinco entregables en orden:

1. Integración de módulos en la interfaz web: ensamblar generador de pacientes, sistema experto, modelo predictivo y motor de simulación en una sola aplicación. Verificar que la representación gráfica del hospital (pisos, áreas, camas), la lista de pacientes en espera, el panel de control (velocidad 1x-10x, pausa), el indicador I y el acceso a información de cada cama operan correctamente de forma integrada.

2. Pruebas de integración: ejecutar los tres escenarios de F5 sobre el prototipo integrado. Verificar que no existen errores críticos. Documentar cualquier comportamiento distinto al observado en F5 con los módulos separados.

3. Manual de usuario: instrucciones para operar el simulador sin conocimiento técnico previo. Cubrir: configuración de la infraestructura hospitalaria, inicio de simulación, uso del modo automático y asistido, interpretación del indicador I.

4. Documentación técnica del código: comentarios en el código, descripción de la arquitectura de módulos, instrucciones para reproducir el entorno de desarrollo.

5. Informe final del proyecto: siguiendo los estándares del curso. Incluir descripción del sistema, metodología CRISP-DM/S aplicada, resultados de evaluación por escenario, limitaciones del prototipo y trabajo futuro (requisitos previos para despliegue real según normograma).

Criterio de salida de F6: todos los módulos operan correctamente como sistema unificado; el sistema puede ser operado por un usuario sin conocimiento técnico previo; el código está documentado y es reproducible; el proyecto está documentado según los estándares del curso.

Empecemos con el entregable 1. Adjunta el estado actual del código de integración o dime qué módulos están listos para ensamblar.
</tarea_f6>
```

---

### 🔄 Historial de ajustes

v1.0 — 2026-06-06 — Versión inicial. Project Instructions + 7 conversation starters (F1, F2, F3, F4-A, F4-B, F5, F6). Stack excluido de las Instructions por decisión de diseño: vive en variable [STACK_ACTUAL] de cada starter para mantener las Instructions estables ante cambios tecnológicos.
