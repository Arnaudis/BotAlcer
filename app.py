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
st.set_page_config(page_title="BotAlcer - Asistente ERC", page_icon="🏥", layout="centered")

# Inicializar historial de conversación
if "historial_conversacion" not in st.session_state:
    st.session_state.historial_conversacion = []

if "mensajes" not in st.session_state:
    saludo_inicial = "¡Hola! Soy BotAlcer, tu asistente sobre la Enfermedad Renal Crónica (ERC) de ALCER. ¿En qué te puedo ayudar hoy?"
    st.session_state.mensajes = [{"rol": "assistant", "texto": saludo_inicial}]




# -----------------
# 2. CSS
# -----------------

# Fondo blanco a través de CSS inyectado (evita que el modo oscuro lo rompa)
st.markdown(
    """
    <style>

    /* =========================================================
       FONDO GENERAL DE LA APLICACIÓN
       ========================================================= */

    .stApp {
        /* Color de fondo y color del texto */
        background-color: #004C42 !important;
        color: #2c3e50 !important;
    }


    /* =========================================================
       CONTENEDOR PRINCIPAL
       ========================================================= */

    /* Contenedor principal más ancho */
    .block-container {
        max-width: 1000px !important;

        /* Dejamos espacio arriba para que la cabecera fija
           no tape el historial de conversación */
        padding-top: 280px !important;

        padding-left: 2rem !important;
        padding-right: 2rem !important;
        padding-bottom: 5rem !important;
    }


    /* =========================================================
       CABECERA FIJA DE BOTALCER
       ========================================================= */

    /* La cabecera queda fija mientras hacemos scroll */
    .st-key-cabecera_fija {
        position: fixed !important;

        top: 0 !important;
        left: 0 !important;

        width: 100vw !important;
        height: 260px !important;

        background-color: #004C42 !important;

        z-index: 999999 !important;

        padding-top: 15px !important;
        padding-bottom: 10px !important;

        box-sizing: border-box !important;

        /* Evita que aparezca una sombra o borde extraño */
        border: none !important;
        box-shadow: none !important;
    }


    /* El contenido de la cabecera permanece centrado */
    .st-key-cabecera_fija > div {
        max-width: 1000px !important;
        margin: 0 auto !important;
    }


    /* =========================================================
       TÍTULO PRINCIPAL
       ========================================================= */

    h1 {
        font-size: 42px !important;
        white-space: nowrap !important;
        text-align: center !important;
    }

    h1, h2, h3, p, span {
        color: #ffffff !important;
    }


    /* =========================================================
       OCULTAR CABECERA NATIVA DE STREAMLIT
       ========================================================= */

    [data-testid="stHeader"] {
        display: none !important;
        /* Esconder completamente la cabecera invisible */
    }


    /* =========================================================
       LOGOS / COLUMNAS DE LA CABECERA
       ========================================================= */

    [data-testid="stHorizontalBlock"] {
        margin-top: 0 !important;
        margin-bottom: 1rem !important;
    }


    /* =========================================================
       PERSONALIZAMOS LA ENTRADA DE TEXTO DEL USUARIO
       ========================================================= */

    [data-testid="stChatInput"] {
        border: 2px solid #009837 !important;
        border-radius: 12px !important;
        background-color: #ffffff !important;
    }

    [data-testid="stChatInput"] textarea {
        background-color: #ffffff !important;
        color: #1e3a8a !important;
    }


    /* =========================================================
       RESPUESTA DEL CHATBOT
       ========================================================= */

    [data-testid="stChatMessage"] p {
        background-color: #ffffff !important;
        color: #1e3a8a !important;
        padding: 2px !important;
    }


    /* Texto de la respuesta del asistente */
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        background-color: #ffffff !important;
        color: #000000 !important;
        padding-top: 2px !important;
        padding-bottom: 2px !important;
    }

    </style>
    """,
    unsafe_allow_html=True  # <-- ¡Muy importante para que el CSS funcione!
)


# Añado el logo centrado
# Construir ruta absoluta dinámica para la imagen
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "pictures", "logoAlcer.png")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # Al estar dentro de col2, st.image centrará el logo automáticamente en el medio de la web
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=380)
    else:
        st.error(f"No se encontró el logo en: {LOGO_PATH}")

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
    unsafe_allow_html=True)




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
