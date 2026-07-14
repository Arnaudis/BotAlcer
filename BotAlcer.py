# Desarrollado por Arnaudis Suárez Sebastián
# Máster en Big Data y Ciencia de Datos
# Universidad Internacional de Valencia
# Abril 2025 - Octubre 2026



# -----------------
# 1. Importaciones
# -----------------

import os
from getpass import getpass
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
# La libería PyPDFLoader genera un DeprecationWarning y queremos que no aparezca.
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.llms import Ollama



# -----------------
# 2. API de Pinecone
# -----------------

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    PINECONE_API_KEY = getpass("Introduce tu Pinecone API Key: ")
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY



# ------------------------
# 3. Carga de Información
# ------------------------

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
# Pero los hemos unifiacado...
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



# -------------------------------------------
# 4. Embeddings en nomic-embed-text (Ollama)
# -------------------------------------------

# Mistral es el LLM
embeddings = OllamaEmbeddings(model="nomic-embed-text")



# ---------------------------
# 5. Preparación de Pinecone
# ---------------------------

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "botalcer-mistral"
existing_indexes = pc.list_indexes().names()
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=768,  # Dimensión del embedding.
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)



# -----------------------------------
# 6. Subida de documentos a Pinecone
# -----------------------------------

vectors = []
for i, d in enumerate(docs):
    # El embed_query es para consultas, el embed_documents es para documentos. 
    # Aunque en este caso, como solo tenemos un texto, podríamos usar cualquiera de los dos, pero lo correcto es usar embed_documents.
    # vec = embeddings.embed_query(d.page_content)
    vec = embeddings.embed_documents([d.page_content])[0]
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



# ----------------------
# 7. Gestión del Modelo
# ----------------------

# Si la temperaturas es 0.0, las respuestas que obtendremos serán siempre las mismas para la misma pregunta.
# Con la temperatura a 0.2 para que de respuestas casi idénticas
# Si queremos respuestas más variadas, podemos subir la temperatura a 0.6 o 0.8, pero cuidado con respuestas incoherentes.
llm = OllamaLLM(model="mistral", temperature=0.2)



# --------------------------------
# 8. Memoria para la conversación
# --------------------------------

# Esta lista, inicialmente vacía, se usará para almacenar la conversacion entre usuario y asistente
historial_conversacion = []



# -------------------------------------
# 9. Función RAG que incorpora memoria
# -------------------------------------

def rag_query(query, k=4):
    # k es el número de resultados que queremos recuperar de Pinecone. Muy alto, podemos obtener resultados irrelevantes, pero si es muy bajo perder información útil.
    # Para Mistral 7B, mejor 4.
    # embed_query es lo correcto para consultas del usuario.
    qvec = embeddings.embed_query(query)
    res = index.query(
        vector=qvec,
        top_k=k,
        include_metadata=True
    )
    # Si no hay coincidencias, devolvemos mensaje claro.
    if not res["matches"]:
        return "O tu pregunta no está bien formulada o no encontré información adecuada sobre tu pregunta para poder responderte."
    # Filtrar por similitud mínima. 
    # Si es muy alto podemos quedarnos sin resultados aunque si se baja mucho el valor, se pueden obtener resultados sin relación con la pregunta.
    matches = [m for m in res["matches"] if m["score"] > 0.5]
    # Si después del filtrado no queda nada, devolvemos un mensaje claro
    if not matches:
        return "O tu pregunta no está bien formulada o no encontré información adecuada en el documento para poder responderte."
    # Limitar a los k mejores
    matches = matches[:k]
    # Construimos el contexto concatenando los chunks recuperados.
    context = "\n\n".join(m["metadata"]["text"] for m in matches)
    # Obtener historial resumido
    history_text = ""
    for i in historial_conversacion:
        history_text += f"Usuario: {i['usuario']}\nAsistente: {i['asistente']}\n\n"
    # Prompt mejorado: incluye memoria + contexto RAG. Además es flexible pues permite conversación continua y razonamiento.
    # Además incluimos etiquetas para especificar los diferentes roles (sistema, usuario, asistente) y el modelo entiende mejor el formato de la conversación.
    prompt = f"""
    <sistema>
    Eres BotAlcer, un asistente experto en Enfermedad Renal Crónica (ERC) y en los
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
    - Si el usuario hace una pregunta fuera del contexto, indícalo y ofrece
    reformularla.
    - Si el usuario pide opinión, aclara que no puedes opinar y responde con datos
    del contexto.
    - Si el usuario pide algo que no está en el documento, dilo claramente.
    Tu objetivo es ser útil, exacto y seguro.
    </sistema>
    <contexto>
    Información recuperada del documento (RAG):
    {context}
    </contexto>
    <historial>
    Resumen del historial de la conversación:
    {history_text}
    </historial>
    <usuario>
    Pregunta del usuario:
    {query}
    </usuario>
    <asistente>
    Genera la mejor respuesta posible siguiendo todas las reglas anteriores.
    </asistente>
    """
    # Obtenemos la respuesta del modelo
    answer = llm.invoke(prompt)
    # Guardamos el turno completo (Pregunta y Respuesta) al mismo tiempo
    historial_conversacion.append({"usuario": query, "asistente": answer})
    return answer



# -------------------
# 10. Ejemplo de uso
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
# 11. Chat interactivo con opción SALIR para concluir
# ----------------------------------------------------

if __name__ == "__main__":
    print("\n¡Bienvenido a BotAlcer, tu asistente personal sobre la Enfermedad Renal Crónica!")
    while True:
        pregunta = input("¿En qué te puedo ayudar?   ")
        # Opción para terminar la ejecución
        if pregunta.strip().upper() == "SALIR":
            print("BotAlcer se despide de ti. ¡Hasta pronto!")
            break
        respuesta = rag_query(pregunta)
        print("\nBotAlcer:\n", respuesta, "\n")
