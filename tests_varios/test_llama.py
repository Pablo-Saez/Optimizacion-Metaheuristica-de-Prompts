import requests

# Función para generar prompts iniciales y procesar contenido
def generar_prompts_iniciales(cantidad, modelo="llama3.1", temperatura=1.0):
    prompts_iniciales = []
    for i in range(1, cantidad + 1):
        print(f"Haciendo consulta #{i} al modelo...")
        payload = {
            "model": modelo,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are generating prompts for a user to create realistic tweets about a disaster similar to the coronavirus. "
                        "Your response must be plain text only. Avoid using any JSON, lists, or structured formats in your response. "
                        
                )
                },
                {"role": "user", "content": "Generate a prompt to make a tweet about coronavirus"}
            ],
            "format": "json",
            "stream": False,
            "options": {
                "temperature": temperatura
            }
        }

        try:
            response = requests.post(
                "http://localhost:11434/api/chat",
                json=payload
            )

            if response.status_code == 200:
                resultado = response.json()
                contenido_raw = resultado.get("message", {}).get("content", "")
                if contenido_raw:
                    print(f"Respuesta cruda #{i}: {contenido_raw.strip()}")
                    # Extraer y limpiar los prompts de la respuesta
                    prompts = extraer_prompts_limpios(contenido_raw)
                    prompts_iniciales.extend(prompts)  # Agregar prompts a la población inicial
            else:
                print(f"Error en la solicitud #{i}: {response.status_code} - {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"Error de conexión en la consulta #{i}: {e}")
    
    return prompts_iniciales

# Función para limpiar y extraer prompts de las respuestas del modelo
def extraer_prompts_limpios(respuesta):
    try:
        # Intentar procesar la respuesta como JSON si es válido
        respuesta_json = eval(respuesta)
        if isinstance(respuesta_json, list):  # Si es una lista, devolverla directamente
            return [prompt.strip() for prompt in respuesta_json if isinstance(prompt, str)]
        elif isinstance(respuesta_json, dict):  # Si es un dict, devolver los valores
            return [value.strip() for value in respuesta_json.values() if isinstance(value, str)]
    except:
        # Si no es JSON válido, procesar líneas directamente
        return [line.strip() for line in respuesta.split("\n") if line.strip() and not line.startswith("{")]

# Generar la población inicial con 10 solicitudes
poblacion_inicial = generar_prompts_iniciales(5)
print("\nPoblación inicial generada:")
for i, prompt in enumerate(poblacion_inicial, 1):
    print(f"{i}. {prompt}")