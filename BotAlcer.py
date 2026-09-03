# Desarrollado por Arnaudis Suárez Sebastián
# Máster en Big Data y Ciencia de Datos
# Universidad Internacional de Valencia
# Abril 2025 - Octubre 2026



# -----------------
# 1. Importaciones
# -----------------

import os
import warnings
from getpass import getpass
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import streamlit as st

# La libería PyPDFLoader genera un DeprecationWarning y queremos que no aparezca.
warnings.filterwarnings("ignore", category=DeprecationWarning)



# -----------------
# 2. API de Pinecone
# -----------------

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    PINECONE_API_KEY = getpass("Introduce tu Pinecone API Key: ")
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY



# -------------------------------------------
# 3. Embeddings en nomic-embed-text (Ollama)
# -------------------------------------------

ollama_url = os.getenv("OLLAMA_HOST")
# Mistral es el LLM

embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://172.17.0.1:11434")
# En local ponemos 127.0.0.1
# embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://127.0.0.1:11434")



# ---------------------------
# 4. Preparación de Pinecone
# ---------------------------

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "botalcer-mistral"

# Se conecta directamente al índice para evitar consultas adicionales a la API
index = pc.Index(index_name)



# -----------------------------------------------
# 5. Carga de Información y Subida a Pinecone
# -----------------------------------------------

# Si fueramos a cargar varios PDFs...
"""
PDF_PATH_1 = "1_Certificado de discapacidad.pdf"  # <-- cámbialo
PDF_PATH_2 = "2_Gestion de dialisis.pdf"
PDF_PATH_3 = "3_Grado de dependencia.pdf"
PDF_PATH_4 = "4_Incapacidad permanente.pdf"
PDF_PATH_5 = "5_Otras actividades.pdf"
PDF_PATH_6 = "6_Pensiones No Contributivas.pdf"
loader_1 = PyPDFLoader(PDF_PATH_1)
loader_2 = PyPDFLoader(PDF_PATH_2)
loader_3 = PyPDFLoader(PDF_PATH_3)
loader_4 = PyPDFLoader(PDF_PATH_4)
loader_5 = PyPDFLoader(PDF_PATH_5)
loader_6 = PyPDFLoader(PDF_PATH_6)
raw_docs_1 = loader_1.load()
raw_docs_2 = loader_2.load()
raw_docs_3 = loader_3.load()
raw_docs_4 = loader_4.load()
raw_docs_5 = loader_5.load()
raw_docs_6 = loader_6.load()
raw_docs = raw_docs_1 + raw_docs_2 + raw_docs_3 + raw_docs_4 + raw_docs_5 + raw_docs_6
"""

# Subir a Pinecone solo si el índice está vacío.
if index.describe_index_stats()["total_vector_count"] == 0:
    print("El índice está vacío. Cargando PDF y subiendo documentos...")
    
    # PDF unificado con la información de los servicios
    PDF_PATH = "0_Informacion_Servicios.pdf"
    loader = PyPDFLoader(PDF_PATH)
    raw_docs = loader.load()
    
    # El chunk es la partición del texto en trozos más pequeñas. Hacemos que cada trozo tenga 1000 caracteres, 
    # con un solapamiento de 200 caracteres entre ellos, que es el chunk_overlap. Esto ayuda a mantener el contexto cuando se dividen los documentos.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.split_documents(raw_docs)
    print("Chunks generados:", len(docs))
    
    # Extraer todos los textos de una vez para acelerar los embeddings
    texts = [d.page_content for d in docs]
    vecs = embeddings.embed_documents(texts)
    
    vectors = []
    for i, (d, vec) in enumerate(zip(docs, vecs)):
        vectors.append({
            # Un id que será único por chunk.
            "id": f"{PDF_PATH}_page_{d.metadata.get('page', 0)}_chunk_{i}",
            "values": vec,
            "metadata": {
                # Texto del chunk, número de página (si está disponible) y fuente del documento.
                "text": d.page_content,
                "page": d.metadata.get("page", None),
                "source": d.metadata.get("source", PDF_PATH)
            }
        })
    index.upsert(vectors=vectors)
    print("Documentos subidos a Pinecone:", len(vectors))
else:
    print("El índice de Pinecone ya contiene datos. Omitiendo lectura de PDF y subida.")



# ----------------------
# 6. Gestión del Modelo
# ----------------------

# Si la temperaturas es 0.0, las respuestas que obtendremos serán siempre las mismas para la misma pregunta.
# Con la temperatura a 0.2 para que de respuestas casi idénticas
# Si queremos respuestas más variadas, podemos subir la temperatura a 0.6 o 0.8, pero cuidado con respuestas incoherentes.

llm = OllamaLLM(model="llama3.2:1b", temperature=0.2, base_url="http://172.17.0.1:11434")
# llm = OllamaLLM(model="llama3.2:1b", temperature=0.2, base_url="http://127.0.0.1:11434")



# --------------------------------
# 7. Memoria para la conversación
# --------------------------------

# En Streamlit inicializamos el historial dentro de st.session_state para que persista
if "historial_conversacion" not in st.session_state:
    st.session_state.historial_conversacion = []

# Referencia directa a la memoria cargada
historial_conversacion = st.session_state.historial_conversacion



# -------------------------------------
# 8. Función RAG que incorpora memoria
# -------------------------------------

# Plantilla estructurada utilizando los roles nativos del modelo
system_template = """Eres BotAlcer, un asistente experto en Enfermedad Renal Crónica (ERC) y en los
servicios ofrecidos por la asociación ALCER. Tu misión es responder de forma
clara, precisa y útil, basándote EXCLUSIVAMENTE en:
1) El contexto recuperado del RAG.
2) El historial resumido de la conversación.

Reglas estrictas:
- Si la información NO aparece en el contexto, dilo explícitamente.
- No inventes datos, no completes información ausente.
- No generalices si el documento no lo respalda.
- Mantén un tono empático, profesional y en español.
- Resume cuando sea necesario, pero sin perder precisión.
- Si el usuario hace una pregunta fuera del contexto, indícalo y ofrece reformularla.
- Si el usuario pide opinión, aclara que no puedes opinar y responde con datos del contexto.
- Si el usuario pide algo que no está en el documento, dilo claramente.
Tu objetivo es ser útil, exacto y seguro.

Información recuperada del documento (RAG):
{context}

Resumen del historial de la conversación:
{history}"""

human_template = "Pregunta del usuario:\n{query}"

prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(human_template)
])


def rag_query(query, k=4):
    # Primeramente vamos a realizar unos pasos previos de normalización y filtro de las entradas del usuario.
    # Normalizar la entrada convirtiendo a minúsculas y quitar espacios sobrantes
    q_norm = query.strip().lower()
    # Elimino signos de interrogación y exclamación
    for char in ["!", "¡", "?", "¿", ".", ",", ";", ":"]:
        q_norm = q_norm.replace(char, "")
    q_norm = q_norm.strip()
    # Eliminar saltos de línea y caracteres invisibles
    q_norm = q_norm.replace("\n", "").replace("\r", "").replace("\t", "")
    
    # Comprobar si el mensaje es ÚNICAMENTE un saludo o empieza por uno
    saludos = ["hola", "buenas", "buenas tardes", "buenas noches", "buenos dias", "saludos", "que tal"]
    if any(q_norm.startswith(saludo) for saludo in saludos):
        respuesta = "¡Hola! Soy BotAlcer, tu asistente sobre la Enfermedad Renal Crónica. ¿En qué te puedo ayudar hoy?"
        st.session_state.historial_conversacion.append({"usuario": query, "asistente": respuesta})
        return respuesta

    # Hay tratar qué responder ante peticiones del usuario relacionadas con salir del chatbot.
    palabras_salida = ["salir", "como salgo", "adios", "chao", "cancelar"]
    if any(q_norm.startswith(salida) for salida in palabras_salida):
        respuesta = "BotAlcer se despide de ti. ¡Hasta pronto!"
        st.session_state.historial_conversacion.append({"usuario": query, "asistente": respuesta})
        return respuesta

    # Tenemos que dar respuesta al usuario que se siente agradecido.
    agradecimientos = ["gracias", "muchas gracias", "ok gracias", "perfecto gracias"]
    if any(q_norm.startswith(agradecimiento) for agradecimiento in agradecimientos):
        respuesta = "¡De nada! Estoy siempre a disposición para cualquier duda que tengas sobre la Enfermedad Renal Crónica o ALCER."
        st.session_state.historial_conversacion.append({"usuario": query, "asistente": respuesta})
        return respuesta

    # Generar embedding de la consulta del usuario
    qvec = embeddings.embed_query(query)
    res = index.query(vector=qvec, top_k=k, include_metadata=True)

    Mensaje="O tu pregunta no está bien formulada o no encontré información adecuada sobre tu pregunta para poder responderte."
    # Comprobar si hay coincidencias
    if not res.get("matches"):
        st.session_state.historial_conversacion.append({"usuario": query, "asistente": Mensaje})
        return Mensaje
    
    # Filtrar por similitud mínima de 0.35 y ordenar descendentemente por score
    matches = [m for m in res["matches"] if m["score"] > 0.35]
    matches = sorted(matches, key=lambda x: x["score"], reverse=True)[:k]
    
    if not matches:
        st.session_state.historial_conversacion.append({"usuario": query, "asistente": Mensaje})
        return Mensaje
    
    # Construir el contexto concatenando los chunks recuperados
    context = "\n\n".join(m["metadata"].get("text", "") for m in matches)
    
    # Construir el historial resumido
    history_text = ""
    for i in st.session_state.historial_conversacion[-4:]:
        history_text += f"Usuario: {i['usuario']}\nAsistente: {i['asistente']}\n\n"
    
    if not history_text.strip():
        history_text = "Sin historial previo."

    # Formatear el prompt usando la estructura de mensajes de LangChain
    messages = prompt_template.format_messages(
        context=context,
        history=history_text,
        query=query
    )

    # Respuesta por Streaming (Escribe palabra a palabra en tiempo real)
    print("\nBotAlcer:")
    full_response = ""
    for chunk in llm.stream(messages):
        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
        print(content, end="", flush=True)
        full_response += content
    print("\n")
    
    # Guardar el turno en la memoria
    st.session_state.historial_conversacion.append({"usuario": query, "asistente": full_response})
    return full_response



# -------------------
# 9. Ejemplo de uso
# -------------------

# Solo para realizar preguntas por prompt, sin interacción continua ni memoria.
"""
if __name__ == "__main__":
    pregunta = "¿Qué discapacidad mínima me otorgan tras el reconocimiento?"
    respuesta = rag_query(pregunta)
    print("\n=== RESPUESTA RAG ===\n")
    print(respuesta)
    pregunta = "¿Dónde presento la solicitud de reconocimiento?"
    respuesta = rag_query(pregunta)
    print("\n=== RESPUESTA RAG ===\n")
    print(respuesta)
    pregunta = "¿Me puedo dializar fuera de mi casa?"
    respuesta = rag_query(pregunta)
    print("\n=== RESPUESTA RAG ===\n")
    print(respuesta)
    pregunta = "¿Cuánto cobraría con una incapacidad permanente?"
    respuesta = rag_query(pregunta)
    print("\n=== RESPUESTA RAG ===\n")
    print(respuesta)
"""



# ----------------------------------------------------
# 10. Chat interactivo con opción SALIR para concluir
# ----------------------------------------------------

# En local...
# if __name__ == "__main__":
#     print("\n¡Bienvenido a BotAlcer, tu asistente personal sobre la Enfermedad Renal Crónica!")
#     while True:
#         pregunta = input("¿En qué te puedo ayudar?   ")
#         # Opción para terminar la ejecución
#         if pregunta.strip().upper() == "SALIR":
#             print("BotAlcer se despide de ti. ¡Hasta pronto!")
#             break
#         respuesta = rag_query(pregunta)
#         #BotAlcer va respondiendo en tiempo real. Si quisiéramos que respondiese todo de golpe, descomentaríamos la siguiente línea.
#         #print("\nBotAlcer:\n", respuesta, "\n")


# En Streamlit...
st.title("BotAlcer")
st.write("¡Bienvenido a BotAlcer, tu asistente personal sobre la Enfermedad Renal Crónica!")

# Inicializamos el historial de la conversación
if "historial_conversacion" not in st.session_state:
    st.session_state.historial_conversacion = []

# Renderizar el historial acumulado previamente
for mensaje in st.session_state.historial_conversacion:
    with st.chat_message("user"):
        st.markdown(mensaje["usuario"])
    with st.chat_message("assistant"):
        st.markdown(mensaje["asistente"])

# Guardamos y procesamos la nueva pregunta que haga el usuario
if prompt := st.chat_input("¿En qué te puedo ayudar?"):
    # Muestra inmediatamente el mensaje del usuario en pantalla
    with st.chat_message("user"):
        st.markdown(prompt)

    # Genera la respuesta (la función rag_query ya la guarda en st.session_state)
    respuesta = rag_query(prompt)

    # Muestra inmediatamente la respuesta en pantalla
    with st.chat_message("assistant"):
        st.markdown(respuesta)
