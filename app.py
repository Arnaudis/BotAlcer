import streamlit as st
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import OllamaEmbeddings
from BotAlcer import rag_query   # <-- Importo tu lógica real

load_dotenv()

# Configuración Inicial de la Página Web
st.set_page_config(page_title="BotAlcer - Asistente ERC", page_icon="🏥", layout="centered")

# Inicializar historial de conversación
if "historial_conversacion" not in st.session_state:
    st.session_state.historial_conversacion = []

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
        border: 5px solid #009837 !important;
        border-radius: 15px !important;
        background-color: #ffffff !important;
    }

    [data-testid="stChatInput"] textarea {
        background-color: #009837 !important;
        color: #000000 !important;
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
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("botalcer-mistral")
    
    # IMPORTANTE para Docker: Si Ollama corre fuera del contenedor (ej. en tu PC), 
    # se suele configurar OLLAMA_HOST en las variables de entorno.
    ollama_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_url)
    llm = OllamaLLM(model="mistral", temperature=0.2, base_url=ollama_url)
    return index, embeddings, llm

index, embeddings, llm = iniciar_componentes()

# Inicializar historial si no existe
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []


# Saludo inicial
if len(st.session_state.mensajes) == 0:
    saludo_inicial = "¡Hola! Soy BotAlcer, tu asistente sobre la Enfermedad Renal Crónica. ¿En qué te puedo ayudar hoy?"
    with st.chat_message("assistant"):
        st.write(saludo_inicial)
    st.session_state.mensajes.append({"rol": "assistant", "texto": saludo_inicial})


# Gestionamos el historial
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
        answer = rag_query(query, llm, st.session_state.historial_conversacion)
        st.session_state.historial_conversacion.append({"usuario": query, "asistente": answer})

    # Mostrar la respuesta del Bot
    with st.chat_message("assistant"):
        st.write(answer)
    st.session_state.mensajes.append({"rol": "assistant", "texto": answer})
