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

if "mensajes" not in st.session_state:
    saludo_inicial = (
        "¡Hola! Soy BotAlcer, tu asistente sobre la Enfermedad Renal Crónica (ERC) de ALCER.\n"
        "¿Estarías interesado en dejar tus datos de contacto para que se ponga en contacto contigo "
        "una trabajadora social de ALCER?"
    )

    st.session_state.mensajes = [
        {
            "rol": "assistant",
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
        max-width: 800px !important;
        width: 100% !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        box-sizing: border-box !important;
    }

    /* Título principal */
    h1 {
        font-size: 32px !important;
        white-space: normal !important;
        text-align: center !important;
        line-height: 1.2 !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
    }

    h1, h2, h3, p, span {
        color: #ffffff !important;
    }

    [data-testid="stHeader"] {
        display: none !important; /* Esconder completamente la cabecera invisible */
    }

    /* Subimos el logo */
    .block-container {
        padding-top: 0rem !important;
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

        [data-testid="stHorizontalBlock"] {
            margin-top: -4em !important;
            margin-bottom: 0 !important;
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

        /* Ajuste del bloque del logo */
        [data-testid="stHorizontalBlock"] {
            margin-top: -3em !important;
            margin-bottom: 0 !important;
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
    unsafe_allow_html=True  # <-- ¡Muy importante para que el CSS funcione!
)

# Añado el logo centrado
# Construir ruta absoluta dinámica para la imagen
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "pictures", "logoAlcer.png")

# Contenedor centrado para el logo
st.markdown(
    """
    <div style="
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
    ">
    """,
    unsafe_allow_html=True
)

if os.path.exists(LOGO_PATH):
    st.image(LOGO_PATH, width=250)
else:
    st.error(f"No se encontró el logo en: {LOGO_PATH}")

st.markdown("</div>", unsafe_allow_html=True)


/* Subimos el título y el subtítulo */
h1 {
    margin-top: -25px !important;
    margin-bottom: 5px !important;
}

#st.title("🏥 BotAlcer")
st.title("Asistente sobre la Enfermedad Renal Crónica (ERC)")
st.markdown(
    """
    <h3 style="
        text-align: center;
        margin-top: -25px;
        margin-bottom: 10px;
    ">
        Por ALCER Las Palmas
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




# -------------------------------------------------------
# 4. Gestión de entradas, salidas e historial en pantalla
# -------------------------------------------------------

# Renderizar todo el historial en pantalla
for msg in st.session_state.mensajes:
    with st.chat_message(msg["rol"]):
        st.write(msg["texto"])


# -------------------------------------------------------
# 4.1. Solicitud de datos de contacto
# -------------------------------------------------------

if st.session_state.contacto_estado == "pendiente":

    st.markdown("<div style='height: 90px;'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Lo deseo", use_container_width=True):
            st.session_state.contacto_estado = "formulario"
            st.rerun()

    with col2:
        if st.button("No lo deseo", use_container_width=True):
            st.session_state.contacto_estado = "finalizado"

            st.session_state.mensajes.append({
                "rol": "assistant",
                "texto": "¿En qué te puedo ayudar hoy?"
            })

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
                "rol": "assistant",
                "texto": "Gracias. ¿En qué te puedo ayudar hoy?"
            })

            st.rerun()


# -------------------------------------------------------
# 4.3. Chatbot
# -------------------------------------------------------

if st.session_state.contacto_estado == "finalizado":

    # Entrada del usuario
    if query := st.chat_input("¿En qué te puedo ayudar hoy?"):
        # Mostrar la pregunta en pantalla
        with st.chat_message("user"):
            st.write(query)
        st.session_state.mensajes.append({"rol": "user", "texto": query})
        
        # Proceso RAG (ahora SOLO tu lógica real)
        with st.spinner("Pensando..."):
            answer = rag_query(query, llm, st.session_state.historial_conversacion,index,embeddings,k=1)
            st.session_state.historial_conversacion.append({"usuario": query, "asistente": answer})

        # Mostrar la respuesta del Bot
        with st.chat_message("assistant"):
            st.write(answer)
        st.session_state.mensajes.append({"rol": "assistant", "texto": answer})
