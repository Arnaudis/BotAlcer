import sys

print("--- 1. Verificando Entorno ---")
print(sys.executable) # Esto te asegura que estás usando el entorno correcto

print("\n--- 2. Probando Importaciones ---")
try:
    from langchain_core import __version__ as core_version
    from langchain_ollama import OllamaLLM
    
    print(f"✅ ¡Éxito! langchain-core versión: {core_version}")
    print("✅ ¡Éxito! OllamaLLM se importó correctamente desde langchain_ollama.")
except ImportError as e:
    print(f"❌ Error de importación: {e}")

print("\n--- 3. Probando Conexión con Ollama ---")
try:
    # Nota: Asegúrate de tener la aplicación Ollama abierta en tu PC
    # Puedes cambiar "llama3" por el modelo que tengas descargado (ej. "mistral", "gemma2", etc.)
    llm = OllamaLLM(model="mistral") 
    
    print("Enviando ping a Ollama...")
    respuesta = llm.invoke("Dime el origen del nombre Arnaudis")
    print(f"🤖 Respuesta del modelo: {respuesta.strip()}")
except Exception as e:
    print(f"❌ No se pudo conectar con Ollama: {e}")
    print("👉 Revisa si la aplicación Ollama está corriendo en segundo plano.")