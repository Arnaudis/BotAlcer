# Desarrollado por Arnaudis Suárez Sebastián
# Máster en Big Data y Ciencia de Datos
# Universidad Internacional de Valencia
# Abril 2025 - Octubre 2026


import streamlit as st
import os
from langchain_ollama import ChatOllama
from BotAlcer import inicializar_recursos_rag, rag_query
# ver donde tarda
import time
from datetime import datetime
import re
from streamlit_autorefresh import st_autorefresh
from PIL import Image, ImageDraw, ImageFont
import io
import base64




# --------------------------------------
# 1. Configuración de la web e Historial
# --------------------------------------

# Configuración Inicial de la Página Web
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ICON_PATH = os.path.join(BASE_DIR, "pictures", "icono.ico")

st.set_page_config(
    page_title="BotAlcer - Asistente ERC",
    page_icon=ICON_PATH,
    layout="centered"
)

# Inicializar historial de conversación
if "historial_conversacion" not in st.session_state:
    st.session_state.historial_conversacion = []

if "procesando" not in st.session_state:
    st.session_state.procesando = False

if "mensajes" not in st.session_state:
    saludo_inicial = (
        "¡Hola! Soy BotAlcer, tu asistente sobre la Enfermedad Renal Crónica (ERC) de ALCER.\n"
        "¿Estarías interesado en dejar tus datos de contacto para que se ponga en contacto contigo "
        "una trabajadora social de ALCER?"
    )

    st.session_state.mensajes = [
        {
            "rol": "Asistente",
            "texto": saludo_inicial
        }
    ]

# Controlar el estado inicial del contacto
if "contacto_estado" not in st.session_state:
    st.session_state.contacto_estado = "pendiente"

# Guardar los datos de contacto
if "nombre_contacto" not in st.session_state:
    st.session_state.nombre_contacto = ""

if "movil_contacto" not in st.session_state:
    st.session_state.movil_contacto = ""

# Guardar la fecha y hora de inicio de la conversación
if "inicio_conversacion" not in st.session_state:
    st.session_state.inicio_conversacion = datetime.now()

# Guardar la fecha y hora de la última interacción
if "ultima_interaccion" not in st.session_state:
    st.session_state.ultima_interaccion = st.session_state.inicio_conversacion

# Controlar si la conversación está cerrada por inactividad
if "conversacion_cerrada" not in st.session_state:
    st.session_state.conversacion_cerrada = False

# Guardar la ruta del archivo de la conversación
if "archivo_conversacion" not in st.session_state:
    st.session_state.archivo_conversacion = None

# Carpeta donde se guardarán las conversaciones
CONVERSACIONES_DIR = os.path.join(BASE_DIR, "conversaciones")
os.makedirs(CONVERSACIONES_DIR, exist_ok=True)


# Guardar toda la conversación en un archivo
def guardar_conversacion():

    # Actualizar la hora de la última interacción
    st.session_state.ultima_interaccion = datetime.now()

    texto_conversacion = ""

    # Recorrer todos los mensajes de la conversación
    for msg in st.session_state.mensajes:
        texto_conversacion += f"{msg['rol'].upper()}: {msg['texto']}\n\n"

    # Contar el número total de palabras de la conversación
    palabras = re.findall(
        r"\b[\wÁÉÍÓÚÜÑáéíóúüñ]+\b",
        texto_conversacion
    )

    total_palabras = len(palabras)

    # Obtener los datos de contacto
    nombre = st.session_state.nombre_contacto.strip()

    if nombre:
        nombre = nombre
    else:
        nombre = "No lo ha proporcionado"

    movil = st.session_state.movil_contacto.strip()

    if movil:
        movil = movil
    else:
        movil = "No lo ha proporcionado"

    # Crear un resumen sencillo de la conversación
    mensajes_usuario = [
        msg["texto"]
        for msg in st.session_state.mensajes
        if msg["rol"] == "Usuario"
    ]

    if mensajes_usuario:
        resumen = (
            "La persona ha realizado las siguientes consultas:\n"
            + "\n".join(
                f"- {mensaje}"
                for mensaje in mensajes_usuario
            )
        )
    else:
        resumen = "No se han realizado consultas."

    # Crear el nombre del archivo usando la fecha y hora de inicio
    inicio = st.session_state.inicio_conversacion

    nombre_archivo = (
        f"{inicio.strftime('%Y%m%d_%H%M%S')}_{total_palabras}.txt"
    )

    ruta_archivo = os.path.join(
        CONVERSACIONES_DIR,
        nombre_archivo
    )

    # Eliminar el archivo anterior de esta conversación
    # para mantener un único archivo actualizado
    archivo_anterior = st.session_state.archivo_conversacion

    if archivo_anterior and archivo_anterior != ruta_archivo:
        if os.path.exists(archivo_anterior):
            os.remove(archivo_anterior)

    # Crear la cabecera de la conversación
    cabecera = (
        "==================================================\n"
        "CONVERSACIÓN BOTALCER\n"
        "==================================================\n\n"
        f"Nombre: {nombre}\n"
        f"Nº de móvil: {movil}\n"
        f"Hora de inicio: {inicio.strftime('%d/%m/%Y %H:%M:%S')}\n"
        f"Hora de finalización: "
        f"{st.session_state.ultima_interaccion.strftime('%d/%m/%Y %H:%M:%S')}\n"
        f"Número de palabras: {total_palabras}\n\n"
        "RESUMEN:\n"
        f"{resumen}\n\n"
        "==================================================\n\n"
        "CONVERSACIÓN\n"
        "==================================================\n\n"
    )

    # Guardar la conversación completa
    with open(ruta_archivo, "w", encoding="utf-8") as archivo:
        archivo.write(cabecera)
        archivo.write(texto_conversacion)

    # Guardar la ruta actual en la sesión
    st.session_state.archivo_conversacion = ruta_archivo



# -----------------
# 2. CSS
# -----------------

# Fondo blanco a través de CSS inyectado (evita que el modo oscuro lo rompa)
st.markdown(
    """
    <style>

    /* --------------------------------------
       CONFIGURACIÓN GENERAL
       -------------------------------------- */

    .stApp {
        /* Color de fondo y color del texto */
        background-color: #004C42 !important;
        color: #2c3e50 !important;
    }

    /* Contenedor principal más ancho */
    .block-container {
        max-width: 1000px !important;
        width: 100% !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        box-sizing: border-box !important;
    }

    /* Título principal */
    h1 {
        font-size: 42px !important;
        white-space: normal !important;
        text-align: center !important;
        line-height: 1.2 !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        margin-top: 20px !important;
        margin-bottom: 0px !important;
    }

    h1, h2, h3, p, span {
        color: #ffffff !important;
    }

    [data-testid="stHeader"] {
        display: none !important; /* Esconder completamente la cabecera invisible */
    }

    /* Subimos el logo */
    .block-container {
        padding-top: 0.5rem !important;
        margin-top: -1rem !important;
    }

    /* Centrar el logo */
    [data-testid="stImage"] {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
    }

    [data-testid="stImage"] img {
        margin-left: auto !important;
        margin-right: auto !important;
        max-width: 100% !important;
        height: auto !important;
    }

    /* --------------------------------------
       LOGO
       -------------------------------------- */

    [data-testid="stImage"] img {
        max-width: 100% !important;
        height: auto !important;
    }

    /* --------------------------------------
       ENTRADA DE TEXTO DEL USUARIO
       -------------------------------------- */

    [data-testid="stChatInput"] {
        border: 2px solid #009837 !important;
        border-radius: 12px !important;
        background-color: #ffffff !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }

    [data-testid="stChatInput"] textarea {
        background-color: #ffffff !important;
        color: #1e3a8a !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }

    /* --------------------------------------
       RESPUESTA DEL CHATBOT
       -------------------------------------- */

    [data-testid="stChatMessage"] {
        max-width: 100% !important;
        box-sizing: border-box !important;
    }

    [data-testid="stChatMessage"] p {
        background-color: #ffffff !important;
        color: #1e3a8a !important;
        padding: 8px !important;
        margin: 0 !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
    }

    /* Texto de la respuesta del asistente */
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        background-color: #ffffff !important;
        color: #000000 !important;
        padding-top: 2px !important;
        padding-bottom: 2px !important;
        max-width: 100% !important;
        overflow-wrap: break-word !important;
        word-wrap: break-word !important;
    }

    /* --------------------------------------
       BOTONES
       -------------------------------------- */

    [data-testid="stButton"] button {
        width: 100% !important;
        min-height: 45px !important;
    }

    /* --------------------------------------
       FORMULARIO
       -------------------------------------- */

    [data-testid="stForm"] {
        width: 100% !important;
        box-sizing: border-box !important;
    }

    /* --------------------------------------
       TABLET
       -------------------------------------- */

    @media (max-width: 768px) {

        .block-container {
            max-width: 100% !important;
            padding-left: 1.25rem !important;
            padding-right: 1.25rem !important;
        }

        h1 {
            font-size: 34px !important;
            white-space: normal !important;
        }

        h3 {
            font-size: 20px !important;
        }

    }

    /* --------------------------------------
       MÓVIL
       -------------------------------------- */

    @media (max-width: 600px) {

        .block-container {
            max-width: 100% !important;
            width: 100% !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            padding-top: 1rem !important;
        }

        /* Título principal */
        h1 {
            font-size: 27px !important;
            line-height: 1.2 !important;
            white-space: normal !important;
            text-align: center !important;
            margin-top: 5px !important;
            margin-bottom: 8px !important;
        }

        /* Subtítulo */
        h3 {
            font-size: 18px !important;
            line-height: 1.25 !important;
            text-align: center !important;
        }

        /* Logo */
        [data-testid="stImage"] img {
            max-width: 75% !important;
            height: auto !important;
        }

        /* Mensajes del chatbot */
        [data-testid="stChatMessage"] {
            width: 100% !important;
            max-width: 100% !important;
            margin-left: 0 !important;
            margin-right: 0 !important;
        }

        [data-testid="stChatMessage"] p {
            font-size: 15px !important;
            line-height: 1.45 !important;
            padding: 8px !important;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
        }

        /* Entrada de texto */
        [data-testid="stChatInput"] {
            width: 100% !important;
            border-radius: 10px !important;
        }

        [data-testid="stChatInput"] textarea {
            font-size: 16px !important;
        }

        /* Botones */
        [data-testid="stButton"] button {
            width: 100% !important;
            min-height: 48px !important;
            font-size: 16px !important;
        }

        /* Espacio antes de los botones */
        [data-testid="stHorizontalBlock"] {
            width: 100% !important;
        }

    }

    /* --------------------------------------
       MÓVIL PEQUEÑO
       -------------------------------------- */

    @media (max-width: 400px) {

        .block-container {
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }

        h1 {
            font-size: 24px !important;
        }

        h3 {
            font-size: 17px !important;
        }

        [data-testid="stImage"] img {
            max-width: 70% !important;
        }

        [data-testid="stChatMessage"] p {
            font-size: 14px !important;
        }

    }

    </style>
    """,
    unsafe_allow_html=True
)

# Añado el logo centrado
# Construir ruta absoluta dinámica para la imagen
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "pictures", "logoAlcer.png")

# Contenedor centrado para el logo
if os.path.exists(LOGO_PATH):

    with open(LOGO_PATH, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <div style="
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
        ">
            <a href="https://arnaudis.es" target="_self">
                <img
                    src="data:image/png;base64,{logo_base64}"
                    width="250"
                    style="cursor: pointer;"
                >
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

else:
    st.error(f"No se encontró el logo en: {LOGO_PATH}")

#st.title("🏥 BotAlcer")
st.title("Asistente sobre la Enfermedad Renal Crónica (ERC)")
st.markdown(
    """
    <h3 style="
        text-align: center;
        text-size: 10px;
        margin-top: -25px;
        margin-bottom: 10px;
    ">
        Por ALCER Las Palmas y Arnaudis Suárez Sebastián
    </h3>
    """,
    unsafe_allow_html=True
)





# ----------------
# 3. Llamada a LLM
# ----------------

# Conexiones BackEnd (Memorizado para no conectarse en cada clic)
@st.cache_resource
def iniciar_componentes():
    # Inicializa Pinecone, Embeddings y verifica el índice en botalcer.py
    index, embeddings = inicializar_recursos_rag()

    # Inicializa el LLM
    ollama_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    llm = ChatOllama(model="qwen3:1.7b", base_url=ollama_url, reasoning=False, temperature=0.1, num_predict=120, num_ctx=2048,)
    return index, embeddings, llm

# Se ejecuta una sola vez al arrancar la app o cuando la caché vence
index, embeddings, llm = iniciar_componentes()

# Comprobar automáticamente la inactividad de la conversación
# Comprobar automáticamente la inactividad de la conversación 
if st.session_state.contacto_estado == "finalizado" and not st.session_state.conversacion_cerrada and not st.session_state.procesando: 
    st_autorefresh(interval=150000, key="control_inactividad") 

    tiempo_inactivo = (datetime.now() - st.session_state.ultima_interaccion).total_seconds()
    if tiempo_inactivo >= 180:
        st.session_state.mensajes.append({
            "rol": "Asistente",
            "texto": "Han pasado unos minutos sin actividad. Me despido por ahora. ¡Gracias por utilizar BotAlcer!"
        })

        st.session_state.conversacion_cerrada = True
        guardar_conversacion()
        st.rerun()




# -------------------------------------------------------
# 4. Gestión de entradas, salidas e historial en pantalla
# -------------------------------------------------------

# Renderizar todo el historial en pantalla
for msg in st.session_state.mensajes:
    if msg["rol"] == "Asistente":
        with st.chat_message(msg["rol"], avatar=ICON_PATH):
            st.write(msg["texto"])
    else:
        with st.chat_message(msg["rol"]):
            st.write(msg["texto"])


# -------------------------------------------------------
# 4.1. Solicitud de datos de contacto
# -------------------------------------------------------

if st.session_state.contacto_estado == "pendiente":

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Lo deseo", use_container_width=True):
            st.session_state.contacto_estado = "formulario"
            st.rerun()

    with col2:
        if st.button("No lo deseo", use_container_width=True):
            st.session_state.contacto_estado = "finalizado"

            st.session_state.mensajes.append({
                "rol": "Asistente",
                "texto": "Gracias. ¿En qué te puedo ayudar ahora?"
            })

            guardar_conversacion()

            st.rerun()


# -------------------------------------------------------
# 4.2. Formulario de datos de contacto
# -------------------------------------------------------

if st.session_state.contacto_estado == "formulario":

    st.markdown("### Datos de contacto")

    with st.form("formulario_contacto"):

        nombre = st.text_input(
            "Nombre",
            value=st.session_state.nombre_contacto
        )

        movil = st.text_input(
            "Nº de móvil",
            value=st.session_state.movil_contacto
        )

        enviar_datos = st.form_submit_button(
            "Enviar datos",
            use_container_width=True
        )

    if enviar_datos:

        if not nombre.strip() or not movil.strip():
            st.warning("Por favor, introduce tu nombre y tu nº de móvil.")

        else:
            # Guardar los datos de contacto
            st.session_state.nombre_contacto = nombre.strip()
            st.session_state.movil_contacto = movil.strip()

            st.session_state.contacto_estado = "finalizado"

            st.session_state.mensajes.append({
                "rol": "Asistente",
                "texto": f"Gracias {nombre.strip()} por facilitar tus datos. Contactaremos contigo lo más pronto posible. ¿En qué te puedo ayudar ahora mismo?"
            })

            guardar_conversacion()

            st.rerun()


# -------------------------------------------------------
# 4.3. Chatbot
# -------------------------------------------------------

# Función para crear un avatar con la inicial del nombre
def crear_avatar_inicial(nombre):
    inicial = nombre.strip()[0].upper() if nombre.strip() else "?"

    imagen = Image.new("RGB", (100, 100), "white")
    dibujo = ImageDraw.Draw(imagen)

    # Círculo
    dibujo.ellipse((0, 0, 100, 100), fill="#00665A")

    # Letra
    fuente = ImageFont.load_default(size=50)

    caja = dibujo.textbbox((0, 0), inicial, font=fuente)
    ancho = caja[2] - caja[0]
    alto = caja[3] - caja[1]

    x = (100 - ancho) / 2
    y = (100 - alto) / 2 - caja[1]

    dibujo.text((x, y), inicial, fill="white", font=fuente)

    buffer = io.BytesIO()
    imagen.save(buffer, format="PNG")
    buffer.seek(0)

    return buffer


if st.session_state.contacto_estado == "finalizado" and not st.session_state.conversacion_cerrada:

    # Entrada del usuario
    # Avatar del usuario...
    if st.session_state.nombre_contacto == "":
        avatar_usuario = "👤"
    else:
        avatar_usuario = crear_avatar_inicial(
            st.session_state.nombre_contacto
        )

    if query := st.chat_input("¿En qué te puedo ayudar hoy?"):
        # Mostrar la pregunta en pantalla
        with st.chat_message("Usuario", avatar=avatar_usuario):
            st.write(query)
        st.session_state.mensajes.append({"rol": "Usuario", "texto": query})
        
        # Proceso RAG (ahora SOLO tu lógica real)
        st.session_state.procesando = True

        with st.spinner("Pensando..."):
            answer = rag_query(query, llm, st.session_state.historial_conversacion,index,embeddings,k=1)
            st.session_state.historial_conversacion.append({"Usuario": query, "Asistente": answer})

        st.session_state.procesando = False

        # Mostrar la respuesta del Bot
        with st.chat_message("Asistente", avatar=ICON_PATH):
            st.write(answer)
        st.session_state.mensajes.append({"rol": "Asistente", "texto": answer})

        guardar_conversacion()

