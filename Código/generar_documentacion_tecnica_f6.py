"""
generar_documentacion_tecnica_f6.py
=====================================
Simulador Inteligente de Ocupacion Hospitalaria
Fase 6 - Entregable 4: Documentacion Tecnica del Codigo

Autor  : Juan Camilo Garcia Braham
Curso  : IA en Salud - Maestria IA y CD - UTP
Anno   : 2026
Stack  : Python 3.12 - reportlab 4.4

Uso:
    python generar_documentacion_tecnica_f6.py

Produce:
    documentacion_tecnica_f6.pdf
"""

import os, sys
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether,
)
from reportlab.platypus.flowables import Flowable

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Paleta (identica a fases anteriores) ------------------------------------
DARK_BG    = colors.HexColor("#1a1a2e")
DARK_MID   = colors.HexColor("#16213e")
ACCENT     = colors.HexColor("#BB86FC")
ACCENT2    = colors.HexColor("#03DAC6")
WHITE      = colors.white
LIGHT_GRAY = colors.HexColor("#E0E0E0")
MID_GRAY   = colors.HexColor("#AAAAAA")
GREEN_OK   = colors.HexColor("#4CAF50")
RED_FAIL   = colors.HexColor("#F44336")
ORANGE     = colors.HexColor("#FF9800")
TABLE_HDR  = colors.HexColor("#2a2a4a")
TABLE_ALT  = colors.HexColor("#ECECEC")
BLUE_DEC   = colors.HexColor("#1565C0")
BLUE_LIGHT = colors.HexColor("#E3F2FD")
CODE_BG    = colors.HexColor("#F5F5F5")
CODE_BOR   = colors.HexColor("#DDDDDD")

W, H   = letter
MARGIN = 2.0 * cm

# --- Flowables ---------------------------------------------------------------
class LineaAccento(Flowable):
    def __init__(self, w, color=ACCENT, h=2):
        super().__init__()
        self.w, self.color, self.h = w, color, h
    def draw(self):
        self.canv.setFillColor(self.color)
        self.canv.rect(0, 0, self.w, self.h, fill=1, stroke=0)
    def wrap(self, *_): return self.w, self.h + 4

def BloqueCodigo(texto, ancho=None):
    """Bloque de codigo monoespacio con fondo gris claro."""
    aw = ancho or (W - 2 * MARGIN)
    estilo = ParagraphStyle(
        "code", fontName="Courier", fontSize=7.8,
        textColor=colors.HexColor("#1a1a2e"),
        leading=11, wordWrap="LTR", splitLongWords=True,
    )
    # Preservar saltos de linea convirtiendo \n en <br/>
    texto_html = texto.replace("&", "&amp;").replace("<", "&lt;") \
                      .replace(">", "&gt;").replace("\n", "<br/>")
    parrafo = Paragraph(texto_html, estilo)
    t = Table([[parrafo]], colWidths=[aw - 0.4*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), CODE_BG),
        ("TOPPADDING",    (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ("RIGHTPADDING",  (0,0), (-1,-1), 10),
        ("BOX",           (0,0), (-1,-1), 0.8, CODE_BOR),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
    ]))
    return t

def CajaDecision(texto, ancho=None):
    """Caja para notas de diseno con fondo azul claro."""
    aw = ancho or (W - 2 * MARGIN)
    estilo = ParagraphStyle(
        "dec", fontName="Helvetica", fontSize=8.5,
        textColor=colors.HexColor("#0d2a4a"),
        leading=13, wordWrap="LTR", splitLongWords=True,
    )
    parrafo = Paragraph(f"<b>[D]  </b>{texto}", estilo)
    t = Table([[parrafo]], colWidths=[aw - 0.6*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), BLUE_LIGHT),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ("RIGHTPADDING",  (0,0), (-1,-1), 10),
        ("BOX",           (0,0), (-1,-1), 1.2, BLUE_DEC),
        ("ROUNDEDCORNERS", [4]),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
    ]))
    return t

# --- Estilos -----------------------------------------------------------------
def estilos():
    return {
        "titulo_cover": ParagraphStyle("titulo_cover",
            fontName="Helvetica-Bold", fontSize=22,
            textColor=WHITE, alignment=TA_CENTER, leading=28, spaceAfter=6),
        "sub_cover": ParagraphStyle("sub_cover",
            fontName="Helvetica", fontSize=13,
            textColor=ACCENT, alignment=TA_CENTER, leading=18, spaceAfter=4),
        "meta_cover": ParagraphStyle("meta_cover",
            fontName="Helvetica", fontSize=10,
            textColor=LIGHT_GRAY, alignment=TA_CENTER, leading=14),
        "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=13,
            textColor=DARK_BG, spaceBefore=14, spaceAfter=6, leading=16),
        "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11,
            textColor=BLUE_DEC, spaceBefore=10, spaceAfter=4, leading=14),
        "h3": ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=10,
            textColor=colors.HexColor("#333333"),
            spaceBefore=6, spaceAfter=3, leading=13),
        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9,
            textColor=colors.HexColor("#222222"),
            leading=13, spaceAfter=4, alignment=TA_JUSTIFY),
        "body_bullet": ParagraphStyle("body_bullet",
            fontName="Helvetica", fontSize=9,
            textColor=colors.HexColor("#222222"),
            leading=13, spaceAfter=2, leftIndent=14),
        "caption": ParagraphStyle("caption",
            fontName="Helvetica-Oblique", fontSize=8,
            textColor=MID_GRAY, alignment=TA_CENTER, leading=10, spaceAfter=4),
        "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=8,
            textColor=colors.HexColor("#222222"), leading=11,
            wordWrap="LTR", splitLongWords=True),
        "cell_hdr": ParagraphStyle("cell_hdr",
            fontName="Helvetica-Bold", fontSize=8,
            textColor=WHITE, leading=11, alignment=TA_CENTER, wordWrap="LTR"),
        "cell_c": ParagraphStyle("cell_c", fontName="Helvetica", fontSize=8,
            textColor=colors.HexColor("#222222"), leading=11,
            alignment=TA_CENTER, wordWrap="LTR"),
        "cell_code": ParagraphStyle("cell_code",
            fontName="Courier", fontSize=7.8,
            textColor=colors.HexColor("#1a1a2e"), leading=11,
            wordWrap="LTR", splitLongWords=True),
    }

def p(texto, st, estilo="body"):
    if isinstance(texto, Paragraph): return texto
    return Paragraph(str(texto), st[estilo])

def tabla(datos, col_widths, st, hdr_bg=TABLE_HDR):
    rows_p = []
    for r_idx, row in enumerate(datos):
        rows_p.append([
            cell if isinstance(cell, Paragraph)
            else Paragraph(str(cell),
                           st["cell_hdr"] if r_idx == 0 else st["cell"])
            for cell in row
        ])
    t = Table(rows_p, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1,  0), hdr_bg),
        ("VALIGN",         (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, TABLE_ALT]),
        ("GRID",           (0, 0), (-1, -1), 0.3,
                           colors.HexColor("#CCCCCC")),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
        ("LEFTPADDING",    (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 6),
    ]))
    return t

# --- Header / Footer ---------------------------------------------------------
def _hf(canvas, doc, seccion=""):
    canvas.saveState()
    canvas.setFillColor(DARK_BG)
    canvas.rect(0, H - 1.2*cm, W, 1.2*cm, fill=1, stroke=0)
    canvas.setFont("Helvetica-Bold", 7.5)
    canvas.setFillColor(ACCENT)
    canvas.drawString(MARGIN, H - 0.75*cm,
                      "Simulador Inteligente de Ocupacion Hospitalaria")
    canvas.setFillColor(MID_GRAY)
    canvas.setFont("Helvetica", 7)
    canvas.drawRightString(W - MARGIN, H - 0.75*cm,
                           f"Documentacion Tecnica  |  {seccion}")
    canvas.setFillColor(DARK_BG)
    canvas.rect(0, 0, W, 1.0*cm, fill=1, stroke=0)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MID_GRAY)
    canvas.drawString(MARGIN, 0.35*cm,
                      "Juan Camilo Garcia Braham  |  UTP 2026")
    canvas.setFillColor(ACCENT)
    canvas.drawRightString(W - MARGIN, 0.35*cm, f"Pagina {doc.page}")
    canvas.restoreState()

# --- CONSTRUCCION DEL PDF ----------------------------------------------------
def construir_pdf(ruta_salida: str):
    st  = estilos()
    uw  = W - 2 * MARGIN

    doc = SimpleDocTemplate(
        ruta_salida, pagesize=letter,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=1.6*cm, bottomMargin=1.4*cm,
    )

    def hf(canvas, doc): _hf(canvas, doc, "Arquitectura y reproducibilidad")

    story = []

    # =========================================================================
    # PORTADA
    # =========================================================================
    story.append(Spacer(1, 2.8*cm))
    story.append(Paragraph(
        "SIMULADOR INTELIGENTE DE<br/>OCUPACI&Oacute;N HOSPITALARIA",
        st["titulo_cover"]))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Documentacion Tecnica del Codigo", st["sub_cover"]))
    story.append(Spacer(1, 0.15*cm))
    story.append(Paragraph("Fase 6 - Despliegue - Entregable 4", st["sub_cover"]))
    story.append(Spacer(1, 0.8*cm))
    story.append(LineaAccento(uw))
    story.append(Spacer(1, 0.5*cm))
    for txt in [
        "Proyecto  Prototipo academico funcional - Curso IA en Salud",
        "Autor  Juan Camilo Garcia Braham",
        "Programa  Maestria en IA y Ciencia de Datos",
        "Institucion  Universidad Tecnologica de Pereira (UTP)",
        "Marco  CRISP-DM/S - SEMMA - DAMA - MLOps",
        "Anno  2026",
    ]:
        story.append(Paragraph(txt, st["meta_cover"]))
    story.append(PageBreak())

    # =========================================================================
    # TABLA DE CONTENIDO
    # =========================================================================
    story.append(Paragraph("Tabla de Contenido", st["h1"]))
    story.append(LineaAccento(uw))
    story.append(Spacer(1, 0.3*cm))
    toc = [
        ["Seccion", "Contenido"],
        ["1", "Resumen de arquitectura del sistema"],
        ["2", "Descripcion de modulos — API publica"],
        ["3", "Diagrama de dependencias entre modulos"],
        ["4", "Convenciones de codigo y nomenclatura"],
        ["5", "Registro consolidado de decisiones de diseno (D_Fx)"],
        ["6", "Instrucciones para reproducir el entorno de desarrollo"],
        ["7", "Instrucciones para ejecutar las pruebas unitarias y de integracion"],
        ["8", "Limitaciones conocidas y trabajo futuro"],
    ]
    story.append(tabla(toc, [uw*0.08, uw*0.92], st))
    story.append(PageBreak())

    # =========================================================================
    # SECCION 1 — ARQUITECTURA
    # =========================================================================
    story.append(Paragraph("1. Resumen de Arquitectura del Sistema", st["h1"]))
    story.append(LineaAccento(uw))
    story.append(Spacer(1, 0.3*cm))
    story.append(p(
        "El simulador sigue una arquitectura de capas desacopladas: "
        "la capa de generacion de datos (F3), la capa de logica del dominio "
        "(F4-A sistema experto), la capa de prediccion (F4-B modelo predictivo), "
        "la capa de evaluacion (F5 motor de simulacion) y la capa de presentacion "
        "(F6 interfaz web). Cada capa solo depende de las capas inferiores, "
        "nunca en sentido contrario.", st))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("1.1 Capas del sistema", st["h2"]))
    capas = [
        ["Capa", "Modulo(s)", "Responsabilidad", "Fase"],
        ["Generacion de datos",
         "generador_pacientes.py",
         "Genera pacientes sinteticos con distribuciones probabilisticas "
         "calibradas. Expone generar_llegadas_tick().",
         "F3"],
        ["Logica del dominio",
         "sistema_experto.py",
         "Entidades (Paciente, Cama, Area, Piso, Hospital), motor de tick "
         "con 19 reglas en 6 capas, modos automatico y asistido.",
         "F4-A"],
        ["Prediccion",
         "modelo_final_f4b.pkl",
         "Pipeline sklearn StandardScaler + LinearRegression. "
         "Predice O en T+4 a partir de 15 features del tick actual.",
         "F4-B"],
        ["Motor de simulacion",
         "motor_simulacion.py",
         "Orquesta el loop tick a tick. Gestiona estado, lags, prediccion "
         "y snapshot del hospital para la UI.",
         "F6"],
        ["Presentacion",
         "app.py",
         "Interfaz Streamlit. Tres paginas: configuracion, simulador en "
         "curso, resultados. Sin logica de dominio propia.",
         "F6"],
    ]
    story.append(tabla(capas, [uw*0.18, uw*0.22, uw*0.48, uw*0.12], st))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("1.2 Principios de diseno aplicados", st["h2"]))
    principios = [
        ("Separacion de responsabilidades",
         "app.py no contiene logica de dominio. motor_simulacion.py no "
         "contiene codigo de UI. sistema_experto.py no sabe que existe "
         "una interfaz web."),
        ("Inmutabilidad de modulos base",
         "sistema_experto.py, generador_pacientes.py y modelo_final_f4b.pkl "
         "no fueron modificados en F6. La integracion ocurre exclusivamente "
         "en motor_simulacion.py."),
        ("Reproducibilidad garantizada",
         "Toda aleatoriedad esta controlada por semillas documentadas. "
         "SEED=99 (evaluacion, D_F5_003), SEED=42 (entrenamiento, D_F3_001). "
         "Con la misma semilla y configuracion, la simulacion produce "
         "los mismos resultados en cualquier entorno."),
        ("Estado centralizado",
         "El estado completo de la simulacion vive en EstadoSimulacion "
         "(dataclass). Streamlit lo serializa en st.session_state. "
         "No hay estado global mutable fuera de este objeto."),
        ("Atomicidad en mutaciones",
         "Toda mutacion que involucra mas de una entidad ocurre dentro "
         "de funciones atomicas (_asignar_cama, _trasladar_paciente, etc.). "
         "Esto previene estados inconsistentes ante excepciones (D_F4A_001)."),
    ]
    for nombre, desc in principios:
        story.append(KeepTogether([
            p(f"<b>{nombre}.</b> {desc}", st),
            Spacer(1, 0.1*cm),
        ]))
    story.append(PageBreak())

    # =========================================================================
    # SECCION 2 — API PUBLICA DE CADA MODULO
    # =========================================================================
    story.append(Paragraph("2. Descripcion de Modulos - API Publica", st["h1"]))
    story.append(LineaAccento(uw))
    story.append(Spacer(1, 0.3*cm))

    # --- 2.1 generador_pacientes.py
    story.append(Paragraph("2.1 generador_pacientes.py", st["h2"]))
    story.append(p(
        "Genera el flujo de llegada de pacientes sinteticos en cada tick "
        "de la simulacion. Las distribuciones de edad, prioridad clinica, "
        "area requerida y tiempo de estancia fueron calibradas en F3 "
        "con referencia a la normativa colombiana (Resolucion 5596/2015).", st))
    story.append(Spacer(1, 0.15*cm))
    api_gen = [
        ["Funcion / Constante", "Firma", "Descripcion"],
        ["LAMBDA_POR_ESCENARIO",
         "dict[str, float]",
         "Tasa de llegadas por escenario: normal=1.5, "
         "alta_demanda=3.0, crisis=5.0 pacientes/tick."],
        ["generar_llegadas_tick()",
         "generar_llegadas_tick(tick, escenario, rng) -> list[Paciente]",
         "Genera los pacientes que llegan en el tick indicado. "
         "El numero de llegadas sigue una distribucion Poisson(lambda). "
         "Cada paciente tiene id unico, edad, prioridad_clinica, "
         "area_requerida y tiempo_estancia_esperado asignados "
         "segun distribuciones empiricas."],
    ]
    story.append(tabla(api_gen, [uw*0.24, uw*0.36, uw*0.40], st))
    story.append(Spacer(1, 0.3*cm))

    # --- 2.2 sistema_experto.py
    story.append(Paragraph("2.2 sistema_experto.py", st["h2"]))
    story.append(p(
        "Modulo central del dominio. Contiene todas las entidades, "
        "el motor de tick y las 19 reglas organizadas en 6 capas "
        "(elegibilidad, priorizacion, asignacion, sobreocupacion, "
        "escalamiento, alta/limpieza).", st))
    story.append(Spacer(1, 0.15*cm))
    api_se = [
        ["Clase / Funcion", "Tipo", "Descripcion"],
        ["Paciente",    "dataclass",
         "Entidad paciente. Campos clave: id_paciente, prioridad_clinica "
         "(P1-P4), area_requerida, tiempo_espera, tiempo_en_sistema, "
         "estado, tick_ingreso."],
        ["Cama",        "dataclass",
         "Entidad cama. Campos: id_cama, tipo, area_id, estado "
         "(libre/ocupada/en_limpieza), paciente_id, es_temporal."],
        ["Area",        "dataclass",
         "Entidad area. Incluye capacidad_total, capacidad_disponible, "
         "todas_las_camas (oficiales + temporales), "
         "prioridades_aceptadas, acepta_desborde."],
        ["Hospital",    "class",
         "Estado global. Indices O(1) por id para pacientes, camas y areas. "
         "Fabrica: crear_hospital_referencia(). "
         "Metodo: pacientes_esperando()."],
        ["ResultadoTick", "dataclass",
         "Resultado de procesar_tick(). Contiene acciones_ejecutadas, "
         "acciones_pendientes, indicador_I, nivel_I, componentes "
         "O/E/P/C y alertas."],
        ["SistemaExperto", "class",
         "Motor de tick. Metodos publicos: procesar_tick(tick, "
         "nuevos_pacientes, modo_asistido=False) y "
         "confirmar_acciones(seleccion, tick)."],
        ["calcular_indicador_I()", "funcion",
         "Calcula I = 0.4*O + 0.2*E + 0.2*P + 0.2*C sobre el estado "
         "actual del hospital. Retorna dict con I, nivel y componentes."],
    ]
    story.append(tabla(api_se, [uw*0.22, uw*0.14, uw*0.64], st))
    story.append(Spacer(1, 0.3*cm))

    # --- 2.3 motor_simulacion.py
    story.append(Paragraph("2.3 motor_simulacion.py", st["h2"]))
    story.append(p(
        "Adapta el motor de F5 para operar tick a tick en lugar de en "
        "bloque. Introduce ConfigHospital (D_F6_008) para parametrizar "
        "la capacidad de cada area desde la UI.", st))
    story.append(Spacer(1, 0.15*cm))
    api_mot = [
        ["Clase / Funcion", "Tipo", "Descripcion"],
        ["ConfigHospital",  "dataclass",
         "Capacidades configurables: uci=10, urgencias=20, "
         "hospitalizacion=40, observacion=15, sala_espera=10. "
         "Valores por defecto = D008/F2."],
        ["EstadoSimulacion", "dataclass",
         "Estado completo y reanudable de una simulacion. Contiene "
         "hospital, se, rng_sim, rng_gen, tick_actual, historial, "
         "historial_O, acciones_pendientes y flags activa/pausada/finalizada."],
        ["RegistroTick",    "dataclass",
         "Una fila del historial. Campos: tick, O_t, E_t, P_t, C_t, "
         "I_t, nivel_I, llegadas_t, altas_t, traslados_t, cola_t, "
         "n_alertas, pred_O_t4."],
        ["crear_estado()",  "funcion",
         "crear_estado(escenario, modo_asistido, seed, ticks_total, config) "
         "-> EstadoSimulacion. Construye el hospital, inicializa SE y "
         "generadores RNG. No ejecuta ningun tick."],
        ["avanzar_tick()",  "funcion",
         "avanzar_tick(estado, modelo_pred) -> (ResultadoTick | None, list). "
         "Procesa exactamente un tick: genera llegadas, llama a "
         "procesar_tick(), calcula prediccion y registra en historial."],
        ["confirmar_acciones_pendientes()", "funcion",
         "confirmar_acciones_pendientes(estado, indices) -> int. "
         "Confirma las acciones de modo asistido seleccionadas "
         "por el gestor. Retorna el numero ejecutadas."],
        ["calcular_resumen()", "funcion",
         "calcular_resumen(estado) -> dict. Metricas de regimen estable "
         "(ticks >= WARM_UP_TICKS): O_media, I_media, nivel_modal, "
         "RMSE pred., cumple_O_rango, cumple_I_nivel."],
        ["snapshot_hospital()", "funcion",
         "snapshot_hospital(estado) -> dict. Vista serializable del "
         "estado actual del hospital para renderizar el grid HTML. "
         "Incluye areas, camas, pacientes y metricas del indicador."],
    ]
    story.append(tabla(api_mot, [uw*0.28, uw*0.14, uw*0.58], st))
    story.append(Spacer(1, 0.3*cm))

    # --- 2.4 app.py
    story.append(Paragraph("2.4 app.py", st["h2"]))
    story.append(p(
        "Punto de entrada de la aplicacion Streamlit. Implementa el router "
        "de paginas y los componentes de UI. No contiene logica de dominio.", st))
    story.append(Spacer(1, 0.15*cm))
    api_app = [
        ["Funcion", "Descripcion"],
        ["main()",
         "Router principal. Lee st.session_state['pagina'] y despacha "
         "a pagina_config(), pagina_simular() o pagina_resultados()."],
        ["pagina_config()",
         "Pantalla de configuracion. Renderiza controles de infraestructura "
         "y parametros. Al pulsar Iniciar crea EstadoSimulacion y "
         "navega a pagina_simular()."],
        ["pagina_simular(modelo)",
         "Pantalla principal. Avanza un tick por rerun, renderiza el grid "
         "HTML interactivo, las graficas de evolucion temporal y el panel "
         "de modo asistido cuando hay acciones pendientes."],
        ["pagina_resultados()",
         "Pantalla de resumen final. Invoca pagina_resumen_inline()."],
        ["pagina_resumen_inline(sim)",
         "Renderiza metricas de regimen estable, verificacion CE-B y "
         "figura completa de O(t) e I(t) con los 200 ticks."],
        ["_html_grid(snap, tick_actual, ticks_total)",
         "Genera el HTML del grid de camas con panel JS de ficha de "
         "paciente (D_F6_003/007). Incluye leyenda, metricas, barra de "
         "progreso, areas con celdas de color y cola de espera."],
        ["_fig_evolucion(historial, mostrar_todo)",
         "Genera la figura Matplotlib de O(t) vs prediccion T+4 e I(t) "
         "con bandas de nivel. mostrar_todo=False muestra los ultimos "
         "60 ticks; True muestra la serie completa."],
        ["cargar_modelo()",
         "Cargado con @st.cache_resource. Deserializa modelo_final_f4b.pkl "
         "una sola vez por sesion de Streamlit."],
    ]
    story.append(tabla(api_app, [uw*0.32, uw*0.68], st))
    story.append(PageBreak())

    # =========================================================================
    # SECCION 3 — DEPENDENCIAS ENTRE MODULOS
    # =========================================================================
    story.append(Paragraph("3. Diagrama de Dependencias entre Modulos", st["h1"]))
    story.append(LineaAccento(uw))
    story.append(Spacer(1, 0.3*cm))
    story.append(p(
        "El grafo de dependencias es un DAG (grafo aciclico dirigido). "
        "Las flechas indican 'importa de':", st))
    story.append(Spacer(1, 0.2*cm))
    story.append(BloqueCodigo(
        "app.py\n"
        "  |-- motor_simulacion.py\n"
        "  |     |-- sistema_experto.py     (Paciente, Cama, Area, Piso,\n"
        "  |     |                            Hospital, SistemaExperto,\n"
        "  |     |                            calcular_indicador_I)\n"
        "  |     |-- generador_pacientes.py  (generar_llegadas_tick,\n"
        "  |     |                            LAMBDA_POR_ESCENARIO)\n"
        "  |     |-- numpy                   (RNG, arrays de features)\n"
        "  |     `-- [modelo_final_f4b.pkl]  (cargado via joblib)\n"
        "  |-- streamlit                    (UI, session_state)\n"
        "  |-- matplotlib                   (figuras O(t) e I(t))\n"
        "  `-- joblib                       (carga del modelo pkl)\n\n"
        "pruebas_integracion_f6.py\n"
        "  |-- motor_simulacion.py           (mismas dependencias)\n"
        "  `-- reportlab                     (PDF de resultados)\n\n"
        "generar_manual_usuario_f6.py\n"
        "  |-- reportlab                     (PDF del manual)\n"
        "  `-- matplotlib                    (figuras ilustrativas)\n\n"
        "generar_documentacion_tecnica_f6.py\n"
        "  `-- reportlab                     (este documento)"
    ))
    story.append(Spacer(1, 0.2*cm))
    story.append(p(
        "Ninguno de los modulos base (sistema_experto.py, "
        "generador_pacientes.py) importa modulos de F6. "
        "La dependencia es estrictamente descendente.", st))
    story.append(Spacer(1, 0.3*cm))

    dep_data = [
        ["Modulo", "Depende de", "No depende de"],
        ["sistema_experto.py",
         "Python stdlib, dataclasses, numpy",
         "Streamlit, matplotlib, sklearn, motor_simulacion, app"],
        ["generador_pacientes.py",
         "Python stdlib, numpy, sistema_experto (Paciente)",
         "Streamlit, matplotlib, sklearn, motor_simulacion, app"],
        ["motor_simulacion.py",
         "sistema_experto, generador_pacientes, numpy, joblib",
         "Streamlit, matplotlib, reportlab, app"],
        ["app.py",
         "motor_simulacion, streamlit, matplotlib, joblib",
         "reportlab, pruebas_integracion_f6"],
        ["modelo_final_f4b.pkl",
         "sklearn 1.8, joblib 1.5 (para deserializar)",
         "Cualquier modulo del proyecto"],
    ]
    story.append(tabla(dep_data, [uw*0.28, uw*0.40, uw*0.32], st))
    story.append(PageBreak())

    # =========================================================================
    # SECCION 4 — CONVENCIONES DE CODIGO
    # =========================================================================
    story.append(Paragraph("4. Convenciones de Codigo y Nomenclatura", st["h1"]))
    story.append(LineaAccento(uw))
    story.append(Spacer(1, 0.3*cm))

    conv_data = [
        ["Convencion", "Descripcion", "Ejemplo"],
        ["Nombres de variables en espanol",
         "Todos los nombres de variables, funciones y clases del dominio "
         "estan en espanol para consistencia con el modelo conceptual de F2.",
         "tiempo_estancia_esperado,\nprioridad_clinica,\ncalcular_indicador_I"],
        ["Nombres de modulos en espanol con guion bajo",
         "Los archivos .py del proyecto usan espanol sin tildes.",
         "motor_simulacion.py,\nsistema_experto.py,\ngenerador_pacientes.py"],
        ["IDs de decisiones D_Fx_NNN",
         "Cada decision de diseno relevante lleva un ID unico con el "
         "formato D_<fase>_<numero>. Se referencia en comentarios y "
         "en la documentacion formal.",
         "D_F4A_001, D_F5_008,\nD_F6_002"],
        ["IDs de riesgos R0N",
         "Los riesgos activos se rastrean con IDs R02, R03, R09 "
         "desde F1. Se mencionan en comentarios junto a la regla "
         "o codigo que los mitiga.",
         "# R02 — caso borde\n# R09 — sesgo demografico"],
        ["Semillas documentadas",
         "Todo uso de aleatoriedad incluye un comentario con el ID "
         "de la decision que fija la semilla.",
         "np.random.default_rng(seed)\n# D_F5_003 SEED=99"],
        ["Dataclasses inmutables para entidades",
         "Las entidades del dominio son dataclasses. Los campos de "
         "identidad (id_*) son strings UUID generados en __post_init__. "
         "La mutacion ocurre solo a traves de funciones atomicas.",
         "@dataclass\nclass Paciente:\n  id_paciente: str"],
        ["Type hints en API publica",
         "Todas las funciones publicas de motor_simulacion.py llevan "
         "anotaciones de tipo. Las funciones internas de sistema_experto.py "
         "siguen el mismo patron.",
         "def crear_estado(\n  escenario: str,\n  ...\n) -> EstadoSimulacion"],
        ["Prefijo _ para funciones internas",
         "Las funciones de uso interno dentro de un modulo llevan "
         "prefijo de guion bajo y no forman parte de la API publica.",
         "_procesar_altas(),\n_asignar_cama(),\n_html_grid()"],
    ]
    story.append(tabla(conv_data, [uw*0.26, uw*0.46, uw*0.28], st))
    story.append(PageBreak())

    # =========================================================================
    # SECCION 5 — DECISIONES DE DISENO CONSOLIDADAS
    # =========================================================================
    story.append(Paragraph(
        "5. Registro Consolidado de Decisiones de Diseno", st["h1"]))
    story.append(LineaAccento(uw))
    story.append(Spacer(1, 0.3*cm))
    story.append(p(
        "Se listan las decisiones de diseno con impacto arquitectonico o "
        "de reproducibilidad acumuladas a lo largo de las seis fases. "
        "Las marcadas con Rev.=F6 fueron tomadas en la fase actual.", st))
    story.append(Spacer(1, 0.2*cm))

    dec_data = [
        ["ID", "Decision", "Justificacion", "Rev."],
        # F2/F3
        ["D_F3_001",
         "SEED global = 42",
         "Semilla base del proyecto para generacion de datos "
         "y entrenamiento del modelo.",
         "—"],
        ["D008",
         "Hospital referencia: UCI=10, Urg=20, Hosp=40, Obs=15, Sala=10",
         "Configuracion base de 95 camas. Parametrizable en F6 "
         "via ConfigHospital (D_F6_008).",
         "F6"],
        # F4-A
        ["D_F4A_001",
         "Mutaciones solo en funciones atomicas",
         "Garantiza invariante cama<->paciente. Previene R02.",
         "—"],
        ["D_F4A_002",
         "TransicionInvalidaError para esperando->dado_de_alta",
         "F2 Tabla 5 declara esta transicion invalida. "
         "Detecta bugs de logica en tiempo de ejecucion.",
         "—"],
        ["D_F4A_003",
         "Orden tick: limpieza-altas-traslados-asignaciones-escalamiento",
         "Modela el ciclo de desinfeccion: camas liberadas en T "
         "no estan disponibles en el mismo T.",
         "F5"],
        ["D_F4A_005",
         "P1 sin UCI permanece en espera, no se reasigna",
         "Preserva componente C del indicador I. "
         "RSO-04 activa RES-04 al superar t_umbral_critico.",
         "F5"],
        ["D_F4A_006/007",
         "Camas temporales en Area.camas_temporales (lista separada)",
         "No alteran capacidad_total ni el calculo de O. "
         "Se destruyen al alta del paciente.",
         "F5"],
        # F4-B
        ["D_F4B_001",
         "Variable objetivo: O_{t+4}, no I",
         "Interpretabilidad clinica maxima. Evita multicolinealidad "
         "de predecir I con sus propios componentes como features.",
         "F5"],
        ["D_F4B_002",
         "Horizonte T+4 (1 hora)",
         "Minimo clinicamente accionable con bajo ruido dado lambda fija.",
         "F5"],
        ["D_F4B_010",
         "Parsimonia: RL si diff RMSE < 1 pp vs RF",
         "Diff = 0.14 pp -> Regresion Lineal ganadora por navaja de Occam.",
         "—"],
        # F5
        ["D_F5_003",
         "SEED evaluacion = 99",
         "Distinta a semillas de entrenamiento (42/123/456/789). "
         "Evita data leakage entre entrenamiento y evaluacion.",
         "—"],
        ["D_F5_005",
         "Gestor asistido confirma 100% en E3",
         "Escenario pesimista conservador para analisis de F6.",
         "F6"],
        ["D_F5_008",
         "Umbrales CE-B ajustados al equilibrio sintetico",
         "O_eq calculada via Ley de Little (L=lambda*W). "
         "Normal: [35,55]%, Alta demanda: [60,88]%, Crisis: [80,100]%.",
         "F6"],
        ["D_F5_009",
         "Riesgo R03 declarado MITIGADO en F5",
         "RMSE eval coherente con F4-B. 3 escenarios diferenciables. "
         "CE-B cumplido en los tres.",
         "—"],
        # F6
        ["D_F6_001",
         "Stack F6: streamlit 1.45.1",
         "Unica dependencia nueva en F6. API session_state estable.",
         "—"],
        ["D_F6_002",
         "Motor tick a tick con estado en EstadoSimulacion",
         "Permite que Streamlit re-renderice entre ticks "
         "sin reiniciar el estado del hospital.",
         "—"],
        ["D_F6_003",
         "Grid HTML con panel JS de ficha de paciente",
         "Click en cama -> panel lateral con datos del paciente "
         "sin salir del componente HTML.",
         "—"],
        ["D_F6_006",
         "Animacion via st.rerun() completo por tick",
         "Mas simple que st.empty(). Compatible con velocidad 1x-10x. "
         "Delay = 0.35/velocidad segundos.",
         "—"],
        ["D_F6_007",
         "Info paciente: panel lateral JS dentro del grid HTML",
         "Click en cama roja abre ficha; click en libre la cierra.",
         "—"],
        ["D_F6_008",
         "Pantalla de configuracion separada antes del simulador",
         "Permite configurar infraestructura hospitalaria antes "
         "de iniciar. Mas claro para usuarios sin conocimiento tecnico.",
         "—"],
        ["D_F6_009",
         "Configuracion de pisos/areas dinamica declarada trabajo futuro",
         "Requeriria refactorizar reglas RSO-02 del SE. "
         "Fuera del alcance del PMV (TF-01).",
         "—"],
        ["D_F6_010",
         "Tolerancias pruebas: O+/-2pp, I+/-3pp, RMSE+/-0.5pp",
         "Variabilidad admisible entre motor independiente (F5) "
         "y motor integrado (F6) con la misma semilla.",
         "—"],
        ["D_F6_012",
         "Pruebas de integracion usan hospital D008",
         "Comparabilidad directa con valores de referencia de F5.",
         "—"],
    ]
    story.append(tabla(dec_data,
                       [uw*0.14, uw*0.25, uw*0.49, uw*0.12], st))
    story.append(PageBreak())

    # =========================================================================
    # SECCION 6 — REPRODUCCION DEL ENTORNO
    # =========================================================================
    story.append(Paragraph(
        "6. Instrucciones para Reproducir el Entorno de Desarrollo", st["h1"]))
    story.append(LineaAccento(uw))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("6.1 Requisitos del sistema", st["h2"]))
    req_data = [
        ["Componente", "Version", "Notas"],
        ["Python",      "3.10+",     "Probado con Python 3.12"],
        ["pip / uv",    "reciente",  "uv pip fue el gestor usado en desarrollo"],
        ["SO",          "cualquiera",
         "Windows 10+, macOS 12+, Ubuntu 22.04+"],
        ["RAM",         "2 GB min.", "4 GB recomendados"],
        ["Disco",       "50 MB",     "Para los archivos del proyecto"],
    ]
    story.append(tabla(req_data, [uw*0.22, uw*0.18, uw*0.60], st))
    story.append(Spacer(1, 0.25*cm))

    story.append(Paragraph("6.2 Estructura de carpetas", st["h2"]))
    story.append(BloqueCodigo(
        "simulador_f6/\n"
        "|- app.py                          # Punto de entrada Streamlit\n"
        "|- motor_simulacion.py             # Motor tick a tick (F6)\n"
        "|- sistema_experto.py              # Motor del dominio (F4-A)\n"
        "|- generador_pacientes.py          # Generador sintetico (F3)\n"
        "|- modelo_final_f4b.pkl            # Modelo predictivo (F4-B)\n"
        "|- pruebas_integracion_f6.py       # Suite de pruebas (F6)\n"
        "|- generar_manual_usuario_f6.py    # Genera manual PDF (F6)\n"
        "`- generar_documentacion_tecnica_f6.py  # Este documento"
    ))
    story.append(Spacer(1, 0.25*cm))

    story.append(Paragraph(
        "6.3 Instalacion de dependencias (entorno limpio)", st["h2"]))
    story.append(BloqueCodigo(
        "# Opcion A: pip\n"
        "python -m venv venv\n"
        "source venv/bin/activate          # Windows: venv\\Scripts\\activate\n"
        "pip install streamlit==1.45.1 numpy==2.4 pandas==3.0 \\\n"
        "            matplotlib==3.10 seaborn==0.13 \\\n"
        "            scikit-learn==1.8 joblib==1.5 reportlab==4.4\n\n"
        "# Opcion B: uv pip (mas rapido)\n"
        "uv venv\n"
        "source .venv/bin/activate\n"
        "uv pip install streamlit==1.45.1 numpy==2.4 pandas==3.0 \\\n"
        "               matplotlib==3.10 seaborn==0.13 \\\n"
        "               scikit-learn==1.8 joblib==1.5 reportlab==4.4"
    ))
    story.append(Spacer(1, 0.25*cm))

    story.append(Paragraph("6.4 Ejecucion de la aplicacion", st["h2"]))
    story.append(BloqueCodigo(
        "# Desde la carpeta simulador_f6/\n"
        "streamlit run app.py\n\n"
        "# La app se abre en http://localhost:8501\n"
        "# Para cambiar el puerto:\n"
        "streamlit run app.py --server.port 8502"
    ))
    story.append(Spacer(1, 0.25*cm))

    story.append(Paragraph("6.5 Verificacion del entorno", st["h2"]))
    story.append(p(
        "Para verificar que todas las dependencias estan correctamente "
        "instaladas antes de lanzar la app, ejecuta:", st))
    story.append(Spacer(1, 0.1*cm))
    story.append(BloqueCodigo(
        "python -c \"\n"
        "import streamlit, numpy, pandas, matplotlib, sklearn, joblib\n"
        "import sistema_experto, generador_pacientes, motor_simulacion\n"
        "import joblib; joblib.load('modelo_final_f4b.pkl')\n"
        "print('Entorno OK')\n"
        "\""
    ))
    story.append(PageBreak())

    # =========================================================================
    # SECCION 7 — PRUEBAS
    # =========================================================================
    story.append(Paragraph(
        "7. Instrucciones para Ejecutar las Pruebas", st["h1"]))
    story.append(LineaAccento(uw))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("7.1 Pruebas unitarias del sistema experto (F4-A)",
                            st["h2"]))
    story.append(p(
        "La suite de 33 pruebas unitarias cubre cada regla de las 6 capas "
        "del sistema experto. No requiere pytest — el runner esta "
        "integrado en el propio modulo.", st))
    story.append(Spacer(1, 0.1*cm))
    story.append(BloqueCodigo(
        "# Desde la carpeta del proyecto\n"
        "python test_sistema_experto.py\n\n"
        "# Resultado esperado:\n"
        "# Total: 33 | Pasadas: 33 | Fallidas: 0"
    ))
    story.append(Spacer(1, 0.25*cm))

    story.append(Paragraph("7.2 Pruebas de integracion (F6)", st["h2"]))
    story.append(p(
        "Ejecuta los tres escenarios con SEED=99 sobre el prototipo "
        "integrado y compara contra valores de referencia de F5. "
        "Genera pruebas_integracion_f6.pdf con los resultados.", st))
    story.append(Spacer(1, 0.1*cm))
    story.append(BloqueCodigo(
        "python pruebas_integracion_f6.py\n\n"
        "# Resultado esperado:\n"
        "# RESULTADO GLOBAL: 18/18 checks PASS\n"
        "# PDF generado: pruebas_integracion_f6.pdf"
    ))
    story.append(Spacer(1, 0.25*cm))

    story.append(Paragraph("7.3 Verificacion de reproducibilidad", st["h2"]))
    story.append(p(
        "Para verificar que una simulacion es reproducible con la misma "
        "semilla, ejecuta el siguiente fragmento:", st))
    story.append(Spacer(1, 0.1*cm))
    story.append(BloqueCodigo(
        "import joblib\n"
        "from motor_simulacion import crear_estado, avanzar_tick, calcular_resumen\n\n"
        "modelo = joblib.load('modelo_final_f4b.pkl')\n"
        "resultados = []\n"
        "for _ in range(2):  # dos ejecuciones con la misma semilla\n"
        "    estado = crear_estado('normal', seed=99, ticks_total=200)\n"
        "    for _ in range(200):\n"
        "        avanzar_tick(estado, modelo)\n"
        "    r = calcular_resumen(estado)\n"
        "    resultados.append(round(r['O_media'], 3))\n\n"
        "assert resultados[0] == resultados[1], 'No reproducible!'\n"
        "print(f'Reproducible: O_media = {resultados[0]}% en ambas ejecuciones')"
    ))
    story.append(Spacer(1, 0.25*cm))

    story.append(Paragraph(
        "7.4 Regeneracion de documentos PDF de F6", st["h2"]))
    story.append(p(
        "Los tres documentos PDF de F6 pueden regenerarse en cualquier "
        "momento ejecutando los scripts correspondientes:", st))
    story.append(Spacer(1, 0.1*cm))
    story.append(BloqueCodigo(
        "python pruebas_integracion_f6.py        # Entregable 2\n"
        "python generar_manual_usuario_f6.py     # Entregable 3\n"
        "python generar_documentacion_tecnica_f6.py  # Entregable 4"
    ))
    story.append(PageBreak())

    # =========================================================================
    # SECCION 8 — LIMITACIONES Y TRABAJO FUTURO
    # =========================================================================
    story.append(Paragraph(
        "8. Limitaciones Conocidas y Trabajo Futuro", st["h1"]))
    story.append(LineaAccento(uw))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("8.1 Limitaciones del prototipo actual", st["h2"]))
    lim_data = [
        ["ID", "Limitacion", "Impacto", "Categoria"],
        ["LIM-01",
         "Datos 100% sinteticos",
         "Los resultados no pueden extrapolarse directamente a "
         "hospitales reales sin calibracion con datos historicos reales.",
         "Datos"],
        ["LIM-02",
         "Lambda fija por escenario durante toda la simulacion",
         "En un hospital real la tasa de llegadas varia por hora "
         "del dia, dia de la semana y estacionalidad.",
         "Modelo"],
        ["LIM-03",
         "Topologia de pisos y areas fija (D008)",
         "No es posible agregar pisos o areas nuevas sin refactorizar "
         "las reglas RSO-02 del sistema experto (TF-01).",
         "Arquitectura"],
        ["LIM-04",
         "Un solo usuario concurrente",
         "Streamlit en modo local no esta disenado para multiples "
         "usuarios simultaneos. Cada instancia tiene su propio estado.",
         "Despliegue"],
        ["LIM-05",
         "Modelo predictivo lineal (RMSE ~2.6 pp)",
         "Adecuado para el PMV academico. En produccion se explorariam "
         "modelos con mayor capacidad para capturar no linealidades.",
         "ML"],
        ["LIM-06",
         "Sin persistencia entre sesiones",
         "Al cerrar el navegador o refrescar la pagina, el estado "
         "de la simulacion se pierde. No hay guardado/carga de sesiones.",
         "Despliegue"],
        ["LIM-07",
         "Sin autenticacion ni control de acceso",
         "Apropiado para uso academico local. "
         "Un despliegue real requeriria autenticacion de usuarios.",
         "Seguridad"],
    ]
    story.append(tabla(lim_data,
                       [uw*0.10, uw*0.26, uw*0.44, uw*0.20], st))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("8.2 Trabajo futuro (TF)", st["h2"]))
    tf_data = [
        ["ID", "Mejora propuesta", "Prerequisito tecnico", "Prioridad"],
        ["TF-01",
         "Configuracion dinamica de pisos y areas",
         "Refactorizar reglas RSO-02 para que la cadena de traslados "
         "sea generica sobre cualquier topologia de areas.",
         "Media"],
        ["TF-02",
         "Lambda variable por hora del dia (patron diurno/nocturno)",
         "Extender generador_pacientes.py con una funcion de "
         "intensidad temporal lambda(t).",
         "Alta"],
        ["TF-03",
         "Calibracion con datos historicos reales anonimizados",
         "Acceso a datos de hospitales colombianos bajo Resolucion "
         "3100/2019. Ajuste de distribuciones en F3.",
         "Alta"],
        ["TF-04",
         "Guardado y carga de sesiones de simulacion",
         "Serializar EstadoSimulacion a JSON/pickle y restaurarlo "
         "desde la UI de Streamlit.",
         "Media"],
        ["TF-05",
         "Modelo predictivo no lineal (Random Forest o Gradient Boosting)",
         "Dataset mayor (>10.000 filas) con variabilidad de lambda. "
         "Ajuste de hiperparametros con CV.",
         "Baja"],
        ["TF-06",
         "Despliegue productivo multi-usuario",
         "Separar estado de sesion en base de datos (Redis/PostgreSQL). "
         "Autenticacion con OAuth. Contenedor Docker.",
         "Alta (v2.0)"],
        ["TF-07",
         "Integracion con HIS/EHR reales",
         "API de interoperabilidad HL7 FHIR. "
         "Cumplimiento de Resolucion 866/2021 (historia clinica digital).",
         "Alta (v2.0)"],
    ]
    story.append(tabla(tf_data,
                       [uw*0.10, uw*0.28, uw*0.44, uw*0.18], st))

    doc.build(story, onFirstPage=hf, onLaterPages=hf)
    print(f"PDF generado: {ruta_salida}")


def main():
    print("Generando Documentacion Tecnica F6...")
    ruta = os.path.join(BASE_DIR, "documentacion_tecnica_f6.pdf")
    construir_pdf(ruta)
    print("Listo.")

if __name__ == "__main__":
    main()
