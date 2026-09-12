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

#if "mensajes" not in st.session_state:
#    saludo_inicial = "¡Hola! Soy BotAlcer, tu asistente sobre la Enfermedad Renal Crónica (ERC) de ALCER."
#    st.session_state.mensajes = [{"rol": "assistant", "texto": saludo_inicial}]

# Controlar el estado inicial del contacto
if "contacto_estado" not in st.session_state:
    st.session_state.contacto_estado = "pendiente"

# Controlar si ya se ha mostrado la pregunta de contacto
if "contacto_pregunta_mostrada" not in st.session_state:
    st.session_state.contacto_pregunta_mostrada = False
    
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
    .stApp {
    /* Color de fondo y color del texto */
        background-color: #004C42 !important;
        color: #2c3e50 !important;
    }

    /* Contenedor principal más ancho */
    .block-container {
        max-width: 1000px !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    /* Título principal */
    h1 {
        font-size: 42px !important;
        white-space: nowrap !important;
        text-align: center !important;
    }

    h1, h2, h3, p, span {
        color: #ffffff !important;
    }

    [data-testid="stHeader"] {
        display: none !important; /* Esconder completamente la cabecera invisible */
    }

    /* Subimos el logo y bajamos el título */
    [data-testid="stHorizontalBlock"] {
        margin-top: -6em !important;    /* Desplaza la imagen hacia arriba para absorber el vacío */
        margin-bottom: -1rem !important; /* Contrae el espacio vacío de la parte inferior de la imagen */
    }

    /* Personalizamos la entrada de texto del usuario */
    [data-testid="stChatInput"] {
        border: 2px solid #009837 !important;
        border-radius: 12px !important;
        background-color: #ffffff !important;
    }

    [data-testid="stChatInput"] textarea {
        background-color: #ffffff !important;
        color: #1e3a8a !important;
    }

    /* Respuesta del chatbot */
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
        st.image(LOGO_PATH, width=275)
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

# Mostrar la pregunta sobre los datos de contacto
if not st.session_state.contacto_pregunta_mostrada:
    st.session_state.mensajes.append({
        "rol": "assistant",
        "texto": "¡Hola! Soy BotAlcer, tu asistente sobre la Enfermedad Renal Crónica (ERC) de ALCER. ¿Estarías interesado en dejar tus datos de contacto para que se ponga en contacto contigo una trabajadora social de ALCER?"
    })
    st.session_state.contacto_pregunta_mostrada = True

# Renderizar todo el historial en pantalla
for msg in st.session_state.mensajes:
    with st.chat_message(msg["rol"]):
        st.write(msg["texto"])


# -------------------------------------------------------
# 4.1. Solicitud de datos de contacto
# -------------------------------------------------------

if st.session_state.contacto_estado == "pendiente":

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