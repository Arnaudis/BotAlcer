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
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
import time
# Librería para reconocer patrones en preguntas similares. Cómo se pide? o cómo pedirlo?
import re
import unicodedata
# La libería PyPDFLoader genera un DeprecationWarning y queremos que no aparezca.
warnings.filterwarnings("ignore", category=DeprecationWarning)
import requests
import json
#####




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
17. Si la pregunta es demasiado general o ambigua y puede corresponder a varios trámites, servicios, prestaciones, ayudas o situaciones diferentes presentes en el CONTENIDO PROPORCIONADO, no elijas uno de ellos arbitrariamente.
18. Cuando exista más de un contexto posible para una pregunta, pide al usuario que especifique a qué trámite, servicio, prestación, ayuda o tema se refiere antes de proporcionar requisitos, documentación, pasos o condiciones.
19. Preguntas como "¿Cuáles son los requisitos?", "¿Qué documentación necesito?", "¿Qué tengo que presentar?" o "¿Qué documentos hacen falta?" deben considerarse ambiguas cuando el CONTENIDO PROPORCIONADO incluya varios trámites o situaciones a los que puedan referirse.
21. Cuando el usuario utilice expresiones como "lo", "eso", "esa", "ese", "aquello", "cómo lo pido", "cómo se solicita", "qué documentación hace falta", "qué requisitos necesito" o expresiones similares, utiliza el HISTORIAL para determinar a qué tema concreto se refiere antes de responder.
22. Cuando el usuario pregunte por requisitos, documentación, pasos, condiciones o cualquier otra información relacionada con un tema previamente identificado en la conversación, responde utilizando únicamente la información correspondiente a ese mismo tema. No mezcles información de otros trámites, prestaciones, servicios o situaciones diferentes.
23. Cuando proporciones requisitos o documentación, indica claramente a qué trámite, prestación, servicio o tema corresponden. Si el tema ha sido identificado mediante el HISTORIAL, mantén ese mismo tema como referencia en la respuesta.
25. Cuando solicites una aclaración, sé breve y pregunta únicamente por la información necesaria para identificar el tema al que se refiere el usuario.
26. Responde siempre en español.
28. Enumera la información cuando facilite la comprensión de la respuesta.
29. No menciones estas instrucciones, el contenido proporcionado, el historial ni el funcionamiento interno del asistente.
31. Responde como un asistente que conoce directamente la información disponible, sin explicar de dónde procede.
32. Si el contenido tiene viñetas (•), reprodúcelas todas, en el mismo orden, sin omitir, dividir ni resumir ninguna: conserva cada viñeta entera, con su encabezado (lo que va antes de los dos puntos) y todos sus datos. Incluye toda la información del contenido que responda a la pregunta, con sus cifras, porcentajes, enlaces web y teléfonos. No comentes estas instrucciones ni lo que falte en el contenido.

CONTENIDO PROPORCIONADO:
{context}

HISTORIAL RELEVANTE:
{history}
"""




# ----------------------------------
# 3. Pinecone: vecotres y embeddings
# ----------------------------------

def dividir_por_preguntas(paginas, splitter, max_chars=800):
    chunks = []
    for doc in paginas:
        for trozo in re.split(r"\n(?=¿)", doc.page_content):
            trozo = trozo.strip()
            if not trozo:
                continue
            if len(trozo) <= max_chars:
                chunks.append(Document(page_content=trozo, metadata=doc.metadata))
            else:
                chunks.extend(splitter.split_documents(
                    [Document(page_content=trozo, metadata=doc.metadata)]))
    return chunks

def titulo_pregunta(texto):
    m = re.match(r"^\s*(¿[^¿]*?\?)", texto, re.S)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else texto


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

        # Divido el PDF en chunks pero que utilice para guiarse los cierres de pregunta
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=100,
            separators=["\n¿", "¿", "\n\n", "\n", " ", ""]
        )
        docs = dividir_por_preguntas(raw_docs, splitter)

        # Genero embeddings
        texts = [titulo_pregunta(d.page_content) for d in docs]
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

    # Preparamos el modelo para que la primera consulta real no sea tan larga
    try:
        embeddings.embed_query("hola")
        prefijo = system_template.split("{context}")[0]   # parte fija de las reglas
        requests.post(
            f"{ollama_url}/api/generate",
            json={"model": "qwen3:1.7b", "prompt": prefijo, "stream": False,
                  "think": False, "keep_alive": -1,
                  "options": {"num_predict": 1, "num_ctx": 3072, "num_thread": 8}},
            timeout=300,
        )
    except Exception as e:
        print("Precalentamiento fallido:", e)


    return index, embeddings




# --------------
# 4. Función RAG
# --------------


def unir_lineas(t):
    """Une las líneas que el PDF cortó por el ancho de página.
    Une cuando la línea siguiente empieza en minúscula o dígito (continuación),
    y sin espacio si la anterior acaba en '-' o '/' (URL partidas)."""
    t = re.sub(r"(?<=[-/])\n(?=[a-záéíóúüñ0-9])", "", t)
    t = re.sub(r"\n(?=[a-záéíóúüñ0-9])(?!\d+[.)]\s)(?!o\s+[A-ZÁÉÍÓÚÑ])", " ", t)
    return re.sub(r"[ \t]{2,}", " ", t)


RE_TITULO = re.compile(r"^\s*(¿[^¿]*?\?)\s*\((\d) de \d\)\s*", re.S)

# Preparo una función para unir las respuestas a las preguntas que tienen dos mitades y se recuperan en dos chunks separados.
def preparar_contexto(textos):
    """Une las mitades '(1 de 2)' / '(2 de 2)' de una misma pregunta, en orden,
    quitando la marca y el título repetido."""
    grupos, orden = {}, []
    for pos, t in enumerate(textos):
        m = RE_TITULO.match(t)
        if m:
            titulo = re.sub(r"\s+", " ", m.group(1)).strip()
            parte, cuerpo = int(m.group(2)), t[m.end():].strip()
        else:
            titulo, parte, cuerpo = None, 0, t.strip()
        clave = re.sub(r"\s+", "", titulo).lower() if titulo else f"_{pos}"
        if clave not in grupos:
            grupos[clave] = []
            orden.append(clave)
        grupos[clave].append((parte, titulo, cuerpo))
    salida = []
    for clave in orden:
        g = sorted(grupos[clave], key=lambda x: x[0])
        titulo = g[0][1]
        cuerpo = unir_lineas("\n".join(c for _, _, c in g))
        salida.append(f"{titulo}\n{cuerpo}" if titulo else cuerpo)
    return "\n\n".join(salida)


def clave_pregunta(texto):
    m = re.match(r"^\s*(¿[^¿]*?\?)", texto, re.S)
    return re.sub(r"\s+", "", m.group(1)).lower() if m else None


PALABRAS_VACIAS = {
    "cuáles", "cuales", "cuál", "cual", "son", "es", "hay", "los", "las", "el", "la", "lo","un", "una", "unos", "unas", "de", "del", "al", 
    "a", "en", "y", "o", "que", "qué", "para", "por", "con", "se", "su", "sus", "me", "mi", "puedo", "puede", "pueden", "debo", "hace", "falta", 
    "necesito", "necesita", "necesarios", "necesaria", "requisitos", "requisito", "generales", "documentación", "documentacion", "documentos", 
    "pasos", "cómo", "como", "dónde", "donde", "cuándo", "cuando", "obtener", "solicitar", "pedir", "tramitar", "conseguir", "hacer", "tengo", 
    "tener", "quiero", "saber", "hola", "información", "informacion", "sobre",
}

def extraer_tema(pregunta):
    palabras = re.findall(r"[\wáéíóúüñ]+", pregunta.lower())
    return " ".join(p for p in palabras if p not in PALABRAS_VACIAS)

# Tipos de dato que el usuario puede pedir y palabras que el contexto debe contener para poder darlo
DATOS_PEDIDOS = [
    (r"\b(horarios?|a que hora)\b",
     {"horario", "horarios", "hora", "horas", "abierto", "abierta", "abre", "cierra",
      "lunes", "martes", "miercoles", "jueves", "viernes", "manana", "tarde"}),
    (r"\b(precio|precios|coste|costes|importe|cuanto (cuesta|cuestan|vale|valen|pago|cobran))\b",
     {"coste", "costes", "cuesta", "cuestan", "precio", "precios", "cuota", "cuantia", "importe",
      "euros", "€", "gratis", "gratuito", "gratuita", "gratuitos", "gratuitas",
      "pagar", "abonar", "abona", "abonan"}),
]

def dato_ausente(q_norm, contexto):
    """True si el usuario pide un tipo de dato (horario, precio...) y el contexto no lo contiene."""
    ctx = "".join(c for c in unicodedata.normalize("NFD", contexto.lower()) if unicodedata.category(c) != "Mn")
    palabras = set(re.findall(r"\w+", ctx))
    for patron, evidencias in DATOS_PEDIDOS:
        if re.search(patron, q_norm):
            tiene = bool(palabras & evidencias) or ("€" in evidencias and "€" in ctx)
            if not tiene:
                return True
    return False

def _sin_tildes(t):
    return "".join(c for c in unicodedata.normalize("NFD", t.lower()) if unicodedata.category(c) != "Mn")


# Para resolver problemas a la hora de "entender" las preguntas de los usuarios.
# Algunas del tipo "¿de qué me puedes/podemos hablar/conversar?" pueden dar complicaciones.
PATRONES_TEMAS = [
    r"(entonces )?(de|sobre) que (me )?(puedes|puede|podemos|puedo|se puede) hablar( conmigo| contigo)?",
    r"(entonces )?(de |sobre )?que temas (me )?(puedes |puede |podemos |puedo |se pueden )?(hablar|tratar|informar|hablas|tratas|conoces|dominas|sabes|conversas)",
    r"(entonces )?que (me )?(puedes|puede|podemos|puedo) (informar|decir|contar|hacer|consultar|preguntar|conversar|tratar|conocer|saber|dominar)",
    r"(entonces )?en que (me )?(puedes|puede|podemos) ayudar",
    r"(entonces )?(que sabes( hacer)?|para que sirves|que haces|que informas|que conoces|que dominas)",
    r"(entonces )?(sobre )?que (te )?(puedo|podemos) (preguntar|consultar|informar|decir|contar|hablar|tratar|conversar|conocer|saber|dominar)",
]
TITULO_TEMAS = "¿De qué temas puede hablar BotAlcer y en qué me puede ayudar?"

# Sinónimos que el usuario usa y el documento no (el PDF dice "solicitar")
SINONIMOS = {
    "pedir": "solicitar", "pide": "solicita", "pido": "solicito", "piden": "solicitan",
    "pedirlo": "solicitarlo", "pedirla": "solicitarla",
    "pedirlos": "solicitarlos", "pedirlas": "solicitarlas",
}
# Quita el saludo para quedarme con la pregunta.
SALUDOS = ["hola", "buenas", "buenas tardes", "buenas noches", "buenos dias", "saludos", "que tal"]
def limpiar_entrada(texto):
    t = texto.strip()
    s = _sin_tildes(t)
    for saludo in sorted(SALUDOS, key=len, reverse=True):
        m = re.match(r"^\W*" + re.escape(saludo) + r"\b[\s,.;:!¡]*", s)
        if m and m.end() < len(t):
            t, s = t[m.end():].strip(), s[m.end():]
            break
    m = re.match(r"^\W*(quiero|quisiera|necesito|me gustaria)\s+(saber|conocer|informacion|mas informacion)\s*(sobre|acerca de)?\s*", s)
    if m and m.end() < len(t):
        t = t[m.end():].strip()
    # Sustituir sinónimos por la palabra que usa el documento
    t = re.sub(r"\b(" + "|".join(SINONIMOS) + r")\b",
               lambda mo: SINONIMOS[mo.group(0).lower()], t, flags=re.I)
    return t


VACIAS_NORM = {_sin_tildes(p) for p in PALABRAS_VACIAS}

def _raiz(p):
    return p[:-1] if len(p) > 3 and p.endswith("s") else p

def palabras_clave(t):
    t = re.sub(r"[^\w\s]", " ", _sin_tildes(t))
    return {_raiz(p) for p in t.split() if len(p) > 2 and p not in VACIAS_NORM}

def cubre_titulo(q_norm, titulo):
    """True si la consulta tiene >= 2 palabras clave y todas están en el título de la pregunta."""
    claves = palabras_clave(q_norm)
    return len(claves) >= 2 and claves <= palabras_clave(titulo)

def cobertura(consulta, titulo):
    """Fracción de las palabras de la consulta que aparecen en el título de la pregunta."""
    qt = {_raiz(p) for p in re.sub(r"[^\w\s]", " ", _sin_tildes(consulta)).split()}
    tt = {_raiz(p) for p in re.sub(r"[^\w\s]", " ", _sin_tildes(titulo)).split()}
    return len(qt & tt) / len(qt) if qt else 0.0


def rag_query(query, llm, history, index, embeddings, k=2, on_token=None):
    inicio_total = time.time()
    # Primeramente vamos a realizar unos pasos previos de normalización y filtro de las entradas del usuario.
    # Normalizar la entrada convirtiendo a minúsculas y quitar espacios sobrantes. Quitaremos tildes y signos desde el principio.
    q_norm = re.sub(r"[^\w\s]", "", "".join(c for c in unicodedata.normalize("NFD", query.strip().lower()) if unicodedata.category(c) != "Mn"))
  
    # Comprobar si el mensaje es ÚNICAMENTE un saludo o empieza por uno
    saludos = ["hola", "buenas", "buenas tardes", "buenas noches", "buenos dias", "saludos", "que tal"]
    if any(q_norm == saludo or q_norm.startswith(saludo + " ") for saludo in saludos) and len(q_norm.split()) <= 3:
        respuesta = "¡Hola! Soy BotAlcer, tu asistente sobre la Enfermedad Renal Crónica (ERC) de ALCER. ¿En qué te puedo ayudar hoy?"
        return respuesta

    # Quitar saludo y sacar lo importante: "hola, quiero información sobre la cuota" -> "la cuota"
    query = limpiar_entrada(query)
    q_norm = re.sub(r"[^\w\s]", "", _sin_tildes(query.strip()))


    # Hay que reastrear por posibles preguntas que presenten ambigüedad solicitudes, documentación o requisitos.
    patrones_ambiguos = [
        r"^como se (pide|piden|solicita|solicitan|consigue|consiguen|obtiene|obtienen|tramita|tramitan|hace|hacen)$",
        r"^como (puedo|se puede) (pedir|solicitar|conseguir|obtener|tramitar|hacer)(lo|la|los|las)?$",
        r"^que tengo que hacer$",
        r"^que tengo que hacer para solicitarlo$",
        r"^que tengo que hacer para pedirlo$",
        r"^que tengo que hacer para conseguirlo$",
        r"^que tengo que hacer para obtenerlo$",
        r"^que pasos tengo que dar$",
        r"^que pasos tengo que seguir$",
        r"^que pasos hay que seguir$",
        r"^que requisitos hay$",
        r"^cuales son los requisitos$",
        r"^que documentacion necesito$",
        r"^que documentos necesito$",
        r"^que papeles necesito$",
        r"^que tengo que presentar$",
        r"^que documentos hacen falta$",
        r"^que papeles hacen falta$",
        r"^que hace falta para solicitarlo$",
        r"^que hace falta para pedirlo$"
    ]

    # Quitamos las tildes
    q_norm_sin_tildes = ''.join(
        c for c in unicodedata.normalize("NFD", q_norm)
        if unicodedata.category(c) != "Mn"
    )

    
    es_seguimiento_corto = q_norm_sin_tildes.startswith("y ") and len(q_norm_sin_tildes.split()) <= 5
    
    # Comprobar si existe un tema claramente identificado en la conversación anterior.
    if any(re.fullmatch(patron, q_norm_sin_tildes) for patron in patrones_ambiguos) or es_seguimiento_corto:
        # Si no existe historial, no podemos saber a qué trámite o servicio se refiere.
        if not history:
            respuesta = "¿Qué trámite, prestación, ayuda o servicio quieres consultar?"
            return respuesta


    # Hay tratar qué responder ante peticiones del usuario relacionadas con salir del chatbot.
    salidas = ["salir", "como salgo", "adios", "chao", "cancelar"]
    if any(q_norm == salida or q_norm.startswith(salida + " ") for salida in salidas) and len(q_norm.split()) <= 2:
        respuesta = "BotAlcer se despide de ti. ¡Hasta pronto!"
        return respuesta

    # Tenemos que dar respuesta al usuario que se siente agradecido.
    agradecimientos = ["gracias", "muchas gracias", "ok gracias", "perfecto gracias"]
    if any(q_norm == agradecimiento or q_norm.startswith(agradecimiento + " ") for agradecimiento in agradecimientos) and len(q_norm.split()) <= 3:
        respuesta = "¡De nada! Estoy siempre a disposición para cualquier duda que tengas sobre la Enfermedad Renal Crónica o la asociación ALCER."
        return respuesta

    # Consejo médico es para pregunta sobre qué tomar, dosis, etc.
    consejo_medico = (
        re.search(r"\b(que|cual)\b.*\b(medicacion|medicamento|medicamentos|pastilla|pastillas|farmaco|farmacos)\b.*\b(tomar|recomiendas|recomienda)\b", q_norm)
        or re.search(r"\b(dosis|cuanto)\b.*\b(tomar|medicacion|medicamento)\b", q_norm)
        or re.search(r"\bque\b.*\b(tomar|tomo)\b.*\b(tension|dolor|fiebre|azucar|colesterol)\b", q_norm)
        or re.search(r"\bcuant[oa]s?\b.*\b(tomar|tomo)\b", q_norm)
    )
    if consejo_medico:
        return "No puedo ofrecer recomendaciones médicas. Consulta con tu médico o nefrólogo."


    # Existe un tema identificado anteriormente? Hay mantener los temas de conversación para posible preguntas dependientes del contexto
    consulta_rag = query


    # Y si hacemos preguntas con referencia a pregunta anterior...
    referencias = ["lo", "eso", "esa", "ese", "esto", "aquello"]

    def depende(qn):
        return (
            any(re.fullmatch(p, qn) for p in patrones_ambiguos)
            or any(r in qn.split() for r in referencias)
            or (qn.startswith("y ") and len(qn.split()) <= 5)   # "¿Y por invalidez?", "¿Y en Lanzarote?"
        )

    depende_del_historial = depende(q_norm)
    if history and depende_del_historial:
        # Buscar la última pregunta del usuario que no sea ella misma un seguimiento
        tema = ""
        pregunta_previa = ""
        for intercambio in reversed(history):
            previo = limpiar_entrada(intercambio.get("usuario", "").strip())
            previo_norm = re.sub(r"[^\w\s]", "", "".join(
                c for c in unicodedata.normalize("NFD", previo.lower())
                if unicodedata.category(c) != "Mn"))

            es_seguimiento = depende(previo_norm)
            if previo and not es_seguimiento:
                tema = extraer_tema(previo)
                pregunta_previa = previo
                break
        if tema:
            if q_norm.startswith("y "):
                # "¿Y por invalidez?" -> "<pregunta anterior completa> por invalidez"
                complemento = re.sub(r"^[\s¿]*[yY]\s+", "", query).strip(" ¿?")
                consulta_rag = f"{pregunta_previa} {complemento}"
            else:
                consulta_rag = f"{pregunta_previa} {query}"

    # Preguntas del tipo "¿de qué me puedes hablar?": ir directamente al fragmento de temas
    if any(re.fullmatch(p, q_norm_sin_tildes) for p in PATRONES_TEMAS):
        consulta_rag = TITULO_TEMAS


    print("\n========== CONSULTA RAG CONTEXTUALIZADA ==========")
    print(consulta_rag)
    print("===================================================")

    # Generar embedding de la consulta del usuario, pero usando el historial.
    t0 = time.time()
    qvec = embeddings.embed_query(consulta_rag)
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
        return "No dispongo de información suficiente en la documentación disponible."
    
    
    # Ordeno los resultados y quiero una similitud mínima
    UMBRAL_MINIMO = 0.75
    matches = sorted(matches, key=lambda m: m.get("score", 0) + 0.1 * cobertura(consulta_rag, titulo_pregunta(m.get("metadata", {}).get("text", ""))),
        reverse=True,)[:k]
    for m in matches:
        t = m.get("metadata", {}).get("text", "")
        print(f"  candidato {m['score']:.4f} | {titulo_pregunta(t)}")

    # Rescate: consulta corta ("avericum telde") cuyas palabras clave están todas en el título de la mejor pregunta
    UMBRAL_RESCATE = 0.60
    rescatada = (
        bool(matches)
        and UMBRAL_RESCATE <= matches[0]["score"] < UMBRAL_MINIMO
        and cubre_titulo(q_norm, titulo_pregunta(matches[0].get("metadata", {}).get("text", "")))
    )
    if rescatada:
        print(f"Aceptada por coincidencia de palabras clave: {matches[0]['score']:.4f}")

    if not matches or (matches[0]["score"] < UMBRAL_MINIMO and not rescatada):
        if matches:
            texto_mejor = matches[0].get("metadata", {}).get("text", "")
            print(f"Rechazada por umbral: {matches[0]['score']:.4f} < {UMBRAL_MINIMO} "
                  f"| mejor pregunta: {titulo_pregunta(texto_mejor)}")
        return "No dispongo de información suficiente en la documentación disponible."
    

    if len(matches) > 1:
        c0 = clave_pregunta(matches[0].get("metadata", {}).get("text", ""))
        c1 = clave_pregunta(matches[1].get("metadata", {}).get("text", ""))
        if not (c0 is not None and c0 == c1):
            matches = matches[:1]

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

        # Para añadir el chunk recuperado a la lista context_parts, que formará el context que se inserta en el prompt.
        #context_parts.append(f"[Fuente: {source} | Página: {page}]\n{text}")
        context_parts.append(text)

    if not context_parts:
        return "No dispongo de información suficiente en la documentación disponible."

    #context = "\n\n".join(context_parts)
    context = preparar_contexto(context_parts)
    print(context)

    if dato_ausente(q_norm, context):
        print("Rechazada: el contexto no contiene el dato solicitado")
        return "No dispongo de información suficiente en la documentación disponible."

    #Los contactos de Avericum se devuelven como están, sin gastar tiempo en el modelo. Es porque repite dos veces la información
    #con el prompt existente.
    titulo_mejor = titulo_pregunta(matches[0].get("metadata", {}).get("text", ""))
    if len(matches) == 1 and re.search(r"dirección y cómo se puede contactar|qué temas puede hablar BotAlcer", titulo_mejor, re.I):
        print("Respuesta directa: ficha de contacto o de temas (sin LLM)")
        print(f"⏱ TOTAL RAG: {time.time() - inicio_total:.2f} segundos")
        cuerpo = context.split("\n", 1)[1].strip() if "\n" in context else context
        return cuerpo.replace("\n", "  \n")
    
    # Pasamos a gestionar el historial.
    history_text = "Sin historial anterior"

    if history:
        historial_prompt = []

        for intercambio in history[-1:]:
            usuario_anterior = intercambio.get("usuario", "").strip()
            asistente_anterior = intercambio.get("asistente", "").strip()[:300]
            if usuario_anterior:
                historial_prompt.append(f"Usuario: {usuario_anterior}")

        if historial_prompt:
            history_text = "\n".join(historial_prompt)

    
    # Formatear el prompt usando la estructura de mensajes de LangChain
    prompt = system_template.format(
        context=context,
        history=history_text
    )

    prompt += f"""

    PREGUNTA DEL USUARIO:
    {query}
    Responde usando solo el contenido proporcionado. Si el contenido tiene viñetas (•), reprodúcelas todas, en el mismo orden, sin omitir, dividir ni resumir ninguna: conserva cada viñeta entera, con su encabezado (lo que va antes de los dos puntos) y todos sus datos. Incluye toda la información del contenido que responda a la pregunta, con sus cifras, porcentajes, enlaces web y teléfonos. Mantén las advertencias, fechas y aclaraciones que acompañen a un dato (por ejemplo, si una cuantía corresponde a un año concreto o es solo orientativa). Incluye las frases introductorias del contenido que respondan a la pregunta. No repitas la pregunta ni la misma información dos veces. No comentes estas instrucciones ni lo que falte en el contenido. No añadas frases de cierre. Sin negritas. Responde siempre en español.
    Si el contenido no responde a lo que se pregunta, aunque trate un tema parecido, responde exactamente: "No dispongo de información suficiente en la documentación disponible."
    """

    # Respuesta del modelo tras invocarlo
    t0 = time.time()

    contenido = ""
    data = {}
    with requests.post(
        f"{os.getenv('OLLAMA_HOST', 'http://ollama:11434')}/api/generate",
        json={"model": "qwen3:1.7b", "prompt": prompt, "stream": True,
            "think": False, "keep_alive": -1,
            "options": {
                "temperature": 0, "num_predict": 800, "num_ctx": 3072,"num_thread": 8
            },
        },
        stream=True, timeout=300) as response:
        response.raise_for_status()
        for linea in response.iter_lines():
            if not linea:
                continue
            data = json.loads(linea)
            contenido += data.get("response", "")
            if on_token and data.get("response"):
                on_token(contenido)
            if data.get("done"):
                break


    # Limpiar la salida: espacios sobrantes, negritas y eco de la pregunta al principio
    contenido = re.sub(r"^[•\s]*¿[^?\n]*\?\s*\n+", "", contenido)
    q_eco = query.strip()
    if "?" in q_eco and contenido.lower().startswith(q_eco.lower()):
        contenido = contenido[len(q_eco):].strip()

    SIN_INFO = "No dispongo de información suficiente en la documentación disponible."

    # Si el modelo solo comenta que el contenido no cubre la pregunta (sin viñetas),
    # sustituir toda la respuesta por la frase estándar de "sin información"
    if re.search(r"\b(contenido proporcionado|documentación proporcionada)\b", contenido, flags=re.I) and "•" not in contenido:
        contenido = SIN_INFO

    # Quitar el comentario inicial del modelo sobre el "contenido proporcionado"
    contenido = re.sub(r"^[^.\n]*\b(contenido proporcionado|documentación proporcionada)\b[^.\n]*[.:]\s*", "", contenido, flags=re.I).strip()
    contenido = re.sub(r"^La información (es la siguiente|que se proporciona es la siguiente)\s*[:.]?\s*", "", contenido, flags=re.I).strip()

    # Quitar la frase final en la que el modelo comenta el "contenido proporcionado"
    contenido = re.sub(r"\s*[^.\n]*\b(contenido proporcionado|documentación proporcionada)\b[^.\n]*\.?\s*$", "", contenido, flags=re.I).strip()

    # Como el modelo Qwen3:1.7b añade, a veces info, pq ignora la orden "no comentes lo que falte".
    # Hay que eliminar las líneas donde el modelo va a sur aire y comenta lo que NO contiene el texto
    if contenido != SIN_INFO:
        META = re.compile(r"((este )?contenido (disponible|proporcionado|no incluye)|documentaci[oó]n (disponible|proporcionada)"                          r"|no se (proporcionan?|mencionan?|incluyen?|especifican?|indica)"
                          r"|el contenido no|sin mencionar)", re.I)
        lineas = [l for l in contenido.split("\n") if not META.search(l)]
        contenido = "\n".join(lineas).strip() or SIN_INFO

    # Quitar el "No dispongo..." añadido detrás de una respuesta que sí tiene contenido
    if contenido != SIN_INFO and contenido.endswith(SIN_INFO):
        contenido = contenido[:-len(SIN_INFO)].strip()



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
