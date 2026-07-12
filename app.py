import streamlit as st
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import OllamaEmbeddings

load_dotenv()

# Configuración Inicial de la Página Web
st.set_page_config(page_title="BotAlcer - Asistente ERC", page_icon="🏥", layout="centered")

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

    # Subimos el logo y bajamos el título
    [data-testid="stHorizontalBlock"] {
        margin-top: -5rem !important;    /* Desplaza la imagen hacia arriba para absorber el vacío */
        margin-bottom: 1rem !important; /* Contrae el espacio vacío de la parte inferior de la imagen */
    }

    # Personalizamos la entrada de texto del usuario
    [data-testid="stChatInput"] {
        border: 5px solid # !important;
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
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # Al estar dentro de col2, st.image centrará el logo automáticamente en el medio de la web
    st.image("Pictures/LogoWeb.png", width=200)

st.title("🏥 BotAlcer")
st.subheader("Asistente experto en Enfermedad Renal Crónica")

# Inicializar historial en la sesión web si no existe
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

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

# Mostrar el historial de la conversación en la web
for msg in st.session_state.mensajes:
    with st.chat_message(msg["rol"]):
        st.write(msg["texto"])

# Entrada del usuario
if query := st.chat_input("¿En qué te puedo ayudar hoy?"):
    # Mostrar la pregunta en pantalla
    with st.chat_message("user"):
        st.write(query)
    st.session_state.mensajes.append({"rol": "user", "texto": query})
    
    # Proceso RAG
    with st.spinner("Pensando..."):
        qvec = embeddings.embed_query(query)
        res = index.query(vector=qvec, top_k=4, include_metadata=True)
        
        matches = [m for m in res["matches"] if m["score"] > 0.5] if res["matches"] else []
        
        if not matches:
            answer = "No encontré información adecuada en los documentos para responder a tu pregunta."
        else:
            context = "\n\n".join(m["metadata"]["text"] for m in matches[:4])
            
            # Construir el hilo del historial estructurado para el Prompt
            history_text = ""
            for m in st.session_state.mensajes[:-1]: # Excluyendo la última pregunta recién añadida
                rol_tag = "Usuario" if m["rol"] == "user" else "Asistente"
                history_text += f"{rol_tag}: {m['texto']}\n"

            prompt = f"""
            <sistema>
            Eres BotAlcer, un asistente experto en Enfermedad Renal Crónica. Responde basándote EXCLUSIVAMENTE en el contexto RAG y el historial proporcionado.
            </sistema>
            <contexto>{context}</contexto>
            <historial>{history_text}</historial>
            <usuario>{query}</usuario>
            <asistente>
            """
            answer = llm.invoke(prompt)
            
    # Mostrar la respuesta del Bot
    with st.chat_message("assistant"):
        st.write(answer)
    st.session_state.mensajes.append({"rol": "assistant", "texto": answer})