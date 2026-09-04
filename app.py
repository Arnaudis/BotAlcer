import streamlit as st
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from BotAlcer import inicializar_recursos_rag, rag_query

load_dotenv()

# Configuración Inicial de la Página Web
st.set_page_config(page_title="BotAlcer - Asistente ERC", page_icon="🏥", layout="centered")

# Inicializar historial de conversación
if "historial_conversacion" not in st.session_state:
    st.session_state.historial_conversacion = []

if "mensajes" not in st.session_state:
    saludo_inicial = "¡Hola! Soy BotAlcer, tu asistente sobre la Enfermedad Renal Crónica. ¿En qué te puedo ayudar hoy?"
    st.session_state.mensajes = [{"rol": "assistant", "texto": saludo_inicial}]


# Fondo blanco a través de CSS inyectado (evita que el modo oscuro lo rompa)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff !important;
        color: #2c3e50 !important;
    }

    h1, h2, h3, p, span {
        color: #1e3a8a !important;
    }

    [data-testid="stHeader"] {
        display: none !important; /* Esconder completamente la cabecera invisible */
    }

    /* Subimos el logo y bajamos el título */
    [data-testid="stHorizontalBlock"] {
        margin-top: -5rem !important;    /* Desplaza la imagen hacia arriba para absorber el vacío */
        margin-bottom: 1rem !important; /* Contrae el espacio vacío de la parte inferior de la imagen */
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
    </style>
    """,
    unsafe_allow_html=True  # <-- ¡Muy importante para que el CSS funcione!
)

# Añado el logo centrado
# Construir ruta absoluta dinámica para la imagen
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "pictures", "logo.png")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # Al estar dentro de col2, st.image centrará el logo automáticamente en el medio de la web
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=200)
    else:
        st.error(f"No se encontró el logo en: {LOGO_PATH}")

st.title("🏥 BotAlcer")
st.subheader("Asistente experto en Enfermedad Renal Crónica")


# Conexiones BackEnd (Memorizado para no conectarse en cada clic)
@st.cache_resource
def iniciar_componentes():
    # Inicializa Pinecone, Embeddings y verifica el índice en botalcer.py
    index, embeddings = inicializar_recursos_rag()

    # Inicializa el LLM
    ollama_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    llm = OllamaLLM(model="qwen2.5:3b", temperature=0.2, base_url=ollama_url)

    return index, embeddings, llm

# Se ejecuta una sola vez al arrancar la app o cuando la caché vence
index, embeddings, llm = iniciar_componentes()


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
        answer = rag_query(query, llm, st.session_state.historial_conversacion,index,embeddings,)
        st.session_state.historial_conversacion.append({"usuario": query, "asistente": answer})

    # Mostrar la respuesta del Bot
    with st.chat_message("assistant"):
        st.write(answer)
    st.session_state.mensajes.append({"rol": "assistant", "texto": answer})
