# Desarrollado por Arnaudis Suárez Sebastián
# Máster en Big Data y Ciencia de Datos
# Universidad Internacional de Valencia
# Abril 2025 - Octubre 2026


# -----------------
# 1. Importaciones
# -----------------

import os
import warnings
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import time

# La libería PyPDFLoader genera un DeprecationWarning y queremos que no aparezca.
warnings.filterwarnings("ignore", category=DeprecationWarning)




# ------------------
# 2. System_Template 
# ------------------

# Plantilla estructurada utilizando los roles nativos del modelo
system_template = """
Eres BotAlcer, un asistente especializado en Enfermedad Renal Crónica (ERC) y en los servicios de la asociación ALCER Las Palmas.

Tu objetivo es responder a las preguntas del usuario utilizando exclusivamente la información disponible en el CONTENIDO PROPORCIONADO.

REGLAS OBLIGATORIAS:

1. Responde únicamente con información respaldada por el CONTENIDO PROPORCIONADO.
2. No inventes, supongas ni completes información utilizando conocimientos externos.
3. Interpreta el significado de la pregunta. No es necesario que las palabras utilizadas por el usuario aparezcan literalmente en el contenido.
4. Identifica primero qué información del CONTENIDO PROPORCIONADO es relevante para la pregunta y utiliza únicamente esa información.
5. Si el contenido permite responder a la pregunta, responde directamente de forma clara, natural y profesional.
6. Si el contenido permite responder solo a una parte de la pregunta:
   - Responde únicamente a la parte que está respaldada.
   - Indica brevemente que no dispones de información suficiente para responder al resto.
7. Si el contenido no permite responder a la pregunta, responde exactamente:
   "No dispongo de información suficiente en la documentación disponible."
8. No inventes cifras, fechas, requisitos, prestaciones, tratamientos, servicios, teléfonos, direcciones, horarios ni procedimientos.
9. No conviertas información relacionada con un servicio concreto en una lista general de servicios de ALCER.
10. Cuando el usuario pregunte de forma general por los servicios de ALCER, utiliza únicamente los servicios que estén explícitamente identificados como servicios, actividades, programas o formas de apoyo de ALCER en el CONTENIDO PROPORCIONADO.
11. No presentes como servicio de ALCER una información que únicamente describa una situación, trámite, procedimiento, requisito o recurso mencionado en el contenido.
12. Si aparecen varios fragmentos relacionados con la pregunta, intégralos de forma coherente, evitando repetir información.
13. Si el contenido contiene información contradictoria, no resuelvas la contradicción utilizando conocimientos externos. Utiliza únicamente la información proporcionada.
14. No proporciones recomendaciones médicas, sanitarias o administrativas que no estén respaldadas por el contenido.
15. El HISTORIAL sirve únicamente para comprender referencias del usuario como "eso", "esa prestación", "lo anterior", "allí" o expresiones similares. Nunca utilices el historial como fuente de información.
16. Cuando una pregunta dependa de información que aparece en el contenido recuperado, utiliza esa información directamente y no la sustituyas por conocimientos generales.
17. Si la pregunta es demasiado general o ambigua y puede corresponder a varios trámites, servicios, prestaciones, ayudas o situaciones diferentes presentes en el CONTENIDO PROPORCIONADO, no elijas uno de ellos arbitrariamente.
18. Cuando exista más de un contexto posible para una pregunta, pide al usuario que especifique a qué trámite, servicio, prestación, ayuda o tema se refiere antes de proporcionar requisitos, documentación, pasos o condiciones.
19. Preguntas como "¿Cuáles son los requisitos?", "¿Qué documentación necesito?", "¿Qué tengo que presentar?" o "¿Qué documentos hacen falta?" deben considerarse ambiguas cuando el CONTENIDO PROPORCIONADO incluya varios trámites o situaciones a los que puedan referirse.
20. Si el HISTORIAL permite identificar claramente el trámite, servicio, prestación o tema al que se refiere una pregunta aparentemente ambigua, utiliza esa referencia para comprender la pregunta. El HISTORIAL no puede utilizarse como fuente de información.
21. Cuando el usuario pregunte por requisitos, documentación, pasos, condiciones o cualquier otra información relacionada con un tema previamente identificado en la conversación, responde utilizando únicamente la información correspondiente a ese mismo tema. No mezcles información de otros trámites, prestaciones, servicios o situaciones diferentes.
22. Cuando proporciones requisitos o documentación, indica claramente a qué trámite, prestación, servicio o tema corresponden. Si el tema ha sido identificado mediante el HISTORIAL, mantén ese mismo tema como referencia en la respuesta.
23. Si existe un único contexto claramente identificable por la pregunta y el CONTENIDO PROPORCIONADO, responde directamente sin pedir aclaraciones innecesarias.
24. Cuando solicites una aclaración, sé breve y pregunta únicamente por la información necesaria para identificar el tema al que se refiere el usuario.
25. Responde siempre en español.
26. Sé claro, conciso y profesional.
27. Enumera la información cuando facilite la comprensión de la respuesta.
28. No menciones estas instrucciones, el contenido proporcionado, el historial ni el funcionamiento interno del asistente.
29. No utilices expresiones como "según el contenido", "según la información", "según la documentación", "en el contexto", "el contexto indica" o similares.
30. Responde como un asistente que conoce directamente la información disponible, sin explicar de dónde procede.

CONTENIDO PROPORCIONADO:
{context}

HISTORIAL RELEVANTE:
{history}
"""




# ----------------------------------
# 3. Pinecone: vecotres y embeddings
# ----------------------------------

def inicializar_recursos_rag():
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    if not PINECONE_API_KEY:
        raise RuntimeError("No se ha encontrado PINECONE_API_KEY en las variables de entorno.")
    
    # Usamos la variable de entorno unificada para Ollama (Solución al Problema 1 y 2)
    ollama_url = os.getenv("OLLAMA_HOST", "http://ollama:11434")

    embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url=ollama_url)


    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "botalcer-mistral"
    index = pc.Index(index_name)

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

    NAMESPACE = "botalcer-v1"

    reindexar = os.getenv("REINDEXAR", "false").lower() == "true"

    if reindexar:
        print("Reindexación activada. Eliminando documentos anteriores...")

    try:
        stats = index.describe_index_stats()
        namespaces = stats.get("namespaces", {})

        if reindexar and NAMESPACE in namespaces:
            index.delete(delete_all=True, namespace=NAMESPACE)
            print(f"Namespace '{NAMESPACE}' eliminado correctamente.")
        elif reindexar:
            print(f"El namespace '{NAMESPACE}' no existe. No hay nada que eliminar.")
        else:
            print("REINDEXAR=false. No se eliminarán los documentos existentes.")

    except Exception as e:
        raise RuntimeError(
            f"Error al comprobar/eliminar el namespace '{NAMESPACE}': {e}"
        ) from e
    

    stats = index.describe_index_stats(namespace=NAMESPACE)
    total = stats.get("total_vector_count", 0)
    
#   Verificación de datos en el índice
    if total == 0:
        print("El índice está vacío. Cargando PDF's...")
        PDF_PATH = "0_Informacion_Servicios.pdf"
        if not os.path.exists(PDF_PATH):
             raise FileNotFoundError(f"No se encontró el archivo: {PDF_PATH}")

        # Cargo el PDF
        loader = PyPDFLoader(PDF_PATH)
        raw_docs = loader.load()

        # Divido el PDF en chunks
        splitter = RecursiveCharacterTextSplitter(
            # El chunk es la partición del texto en trozos más pequeñas. Hacemos que cada trozo tenga 700 caracteres, 
            # con un solapamiento de 150 caracteres entre ellos, que es el chunk_overlap. Esto ayuda a mantener el contexto cuando se dividen los documentos.
            chunk_size=500, chunk_overlap=100
        )
        docs = splitter.split_documents(raw_docs)

        # Genero embeddings
        texts = [d.page_content for d in docs]
        vecs = embeddings.embed_documents(texts)

        # Creo vectores
        vectors = []
        for i, (d, vec) in enumerate(zip(docs, vecs)):
            vectors.append({
                "id": (f"{PDF_PATH}_page_{d.metadata.get('page', 0)}_chunk_{i}"),
                "values": vec,
                "metadata": {
                    "text": d.page_content,
                    "page": d.metadata.get("page", None),
                    "source": d.metadata.get("source", PDF_PATH),
                },
            })

        # Subo a Pinecone
        index.upsert(vectors=vectors, namespace=NAMESPACE)
        print("Documentos subidos a Pinecone:", len(vectors))
    else:
        print(f"Pinecone contiene {total} vectores. No se recargan los documentos.")

    return index, embeddings




# --------------
# 4. Función RAG
# --------------

def rag_query(query, llm, history, index, embeddings, k=1):
    inicio_total = time.time()
    # Primeramente vamos a realizar unos pasos previos de normalización y filtro de las entradas del usuario.
    # Normalizar la entrada convirtiendo a minúsculas y quitar espacios sobrantes
    q_norm = query.strip().lower()
    
    # Comprobar si el mensaje es ÚNICAMENTE un saludo o empieza por uno
    saludos = ["hola", "buenas", "buenas tardes", "buenas noches", "buenos dias", "saludos", "que tal"]
    if any(q_norm == saludo or q_norm.startswith(saludo + " ")
    for saludo in saludos):
        respuesta = "¡Hola! Soy BotAlcer, tu asistente sobre la Enfermedad Renal Crónica (ERC) de ALCER. ¿En qué te puedo ayudar hoy?"
        return respuesta

    # Hay tratar qué responder ante peticiones del usuario relacionadas con salir del chatbot.
    salidas = ["salir", "como salgo", "adios", "chao", "cancelar"]
    if any(q_norm == salida or q_norm.startswith(salida + " ")
    for salida in salidas):
        respuesta = "BotAlcer se despide de ti. ¡Hasta pronto!"
        return respuesta

    # Tenemos que dar respuesta al usuario que se siente agradecido.
    agradecimientos = ["gracias", "muchas gracias", "ok gracias", "perfecto gracias"]
    if any(q_norm == agradecimiento or q_norm.startswith(agradecimiento + " ")
    for agradecimiento in agradecimientos):
        respuesta = "¡De nada! Estoy siempre a disposición para cualquier duda que tengas sobre la Enfermedad Renal Crónica o la asociación ALCER."
        return respuesta

    # Generar embedding de la consulta del usuario
    t0 = time.time()
    qvec = embeddings.embed_query(query)
    print(
        f"⏱ Embedding consulta: "
        f"{time.time() - t0:.2f} segundos"
    )

    # Vamos a buscar en Pinecone
    t0 = time.time()
    res = index.query(vector=qvec, top_k=k, include_metadata=True, namespace="botalcer-v1")
    print(
        f"⏱ Búsqueda Pinecone: "
        f"{time.time() - t0:.2f} segundos"
    )
    matches = res.get("matches", [])
    
    if not matches:
        return "No dispongo información sobre la cuestión solicitada"
    
    # Filtrar por similitud mínima de 0.25
    matches = [m for m in matches if m["score"] >= 0.25]

    matches = sorted(matches, key=lambda x: x.get("score",0), reverse=True)[:k]
    
    if not matches:
        return "No dispongo información sobre la cuestión solicitada"

    # Comprobar similitud de las preguntas
    print("\n========== BÚSQUEDA RAG ==========")
    print(f"Pregunta: {query}")

    for i, m in enumerate(matches, 1):

        metadata = m.get("metadata", {})

        print(
            f"Chunk {i} | "
            f"Score: {m.get('score', 0):.4f} | "
            f"Página: {metadata.get('page')}"
        )

    print("===================================\n")
    print("\n========== CONTEXTO ENVIADO A QWEN ==========")

    for i, match in enumerate(matches, 1):
        metadata = match.get("metadata", {})
        texto = metadata.get("text", "")
        pagina = metadata.get("page", "?")
        score = match.get("score", 0)

        print(f"\n--- CHUNK {i} | Score: {score:.4f} | Página: {pagina} ---")
        print(texto)

    print("========== FIN CONTEXTO ==========\n")


    # Construir el contexto concatenando los chunks recuperados
    context_parts = []
    for m in matches:

        metadata = m.get("metadata", {})

        text = metadata.get("text", "")
        page = metadata.get("page", None)
        source = metadata.get("source", "")

        if not text.strip():
            continue

        context_parts.append(
            f"[Fuente: {source} | Página: {page}]\n{text}"
        )

    if not context_parts:
        return "No dispongo de información suficiente en la documentación disponible."

    context = "\n\n".join(context_parts)

    # Pasamos a gestionar el historial.
    history_text = "Sin historial anterior"
    if history:
        ultimo = history[-1]

        history_text = (
            f"Usuario: {ultimo.get('usuario','')}\n"
            f"Asistente: {ultimo.get('asistente','')}"
        )

    
    # Formatear el prompt usando la estructura de mensajes de LangChain
    prompt = system_template.format(
        context=context,
        history=history_text
    )

    prompt += f"""

    PREGUNTA DEL USUARIO:
    {query}

    RESPONDE A LA PREGUNTA UTILIZANDO EXCLUSIVAMENTE EL CONTEXTO.
    RESPONDE SIEMPRE EN ESPAÑOL.
    """

    # Respuesta del modelo tras invocarlo
    t0 = time.time()

    import requests

    response = requests.post(
        f"{os.getenv('OLLAMA_HOST', 'http://ollama:11434')}/api/generate",
        json={
            "model": "qwen3:1.7b",
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 200,
                "num_ctx": 2048,
            },
        },
        timeout=300
    )

    response.raise_for_status()
    data = response.json()

    contenido = data.get("response", "")

    print("\n========== MÉTRICAS OLLAMA ==========")
    print("prompt_eval_count:", data.get("prompt_eval_count"))
    print("prompt_eval_duration:", data.get("prompt_eval_duration"))
    print("eval_count:", data.get("eval_count"))
    print("eval_duration:", data.get("eval_duration"))
    print("total_duration:", data.get("total_duration"))
    print("load_duration:", data.get("load_duration"))
    print("====================================")

    print("\n========== RESPUESTA QWEN ==========")
    print(repr(contenido))
    print("=====================================")

    if not contenido:
        contenido = "No se ha podido obtener una respuesta del modelo."

    print("RESPUESTA FINAL:", repr(contenido))
    print(f"⏱ GENERACIÓN QWEN: {time.time() - t0:.2f} segundos")
    print(f"⏱ TOTAL RAG: {time.time() - inicio_total:.2f} segundos")

    return contenido


