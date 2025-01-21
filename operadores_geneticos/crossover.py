import random
import spacy

# Cargar el modelo preentrenado de spaCy
nlp = spacy.load("en_core_web_sm")

def crossover(individuo1, individuo2):
    """
    Realiza un crossover con cambios mínimos en el prompt del mejor individuo.
    Sustituye dos palabras clave del mejor individuo con palabras del segundo individuo.
    
    :param individuo1: Diccionario del primer individuo.
    :param individuo2: Diccionario del segundo individuo.
    :return: Nuevo individuo generado por crossover.
    """
    # Seleccionar al mejor individuo basado en fitness
    mejor_individuo, segundo_individuo = (
        (individuo1, individuo2) if individuo1["fitness"] >= individuo2["fitness"] else (individuo2, individuo1)
    )

    # Extraer Role y Task
    role_mejor, task_mejor = dividir_prompt(mejor_individuo["prompt"])
    _, task_segundo = dividir_prompt(segundo_individuo["prompt"])

    # Procesar los Tasks con spaCy
    doc_mejor = nlp(task_mejor)
    doc_segundo = nlp(task_segundo)

    # Extraer palabras clave
    palabras_clave_mejor = [token.text for token in doc_mejor if token.pos_ in ["NOUN", "VERB"]]
    palabras_clave_segundo = [token.text for token in doc_segundo if token.pos_ in ["NOUN", "VERB"]]

    if len(palabras_clave_mejor) < 2 or len(palabras_clave_segundo) == 0:
        # Si no hay suficientes palabras clave, retorna el mejor individuo sin cambios
        return crear_nuevo_individuo(mejor_individuo, mejor_individuo["prompt"], "Sin cambios")

    # Seleccionar dos palabras clave al azar del mejor individuo (el que tieene mejor fitness)
    palabras_a_cambiar = random.sample(palabras_clave_mejor, min(2, len(palabras_clave_mejor)))

    # Seleccionar dos palabras al azar del segundo individuo para sustituir
    palabras_nuevas = random.sample(palabras_clave_segundo, min(len(palabras_a_cambiar), len(palabras_clave_segundo)))

    # Reemplazar palabras en el task del mejor individuo
    nuevo_task = task_mejor
    for palabra_antigua, palabra_nueva in zip(palabras_a_cambiar, palabras_nuevas):
        nuevo_task = nuevo_task.replace(palabra_antigua, palabra_nueva, 1)

    # Crear el nuevo prompt
    nuevo_prompt = f"Role: {role_mejor}. Task: {nuevo_task}"

    # Validar y retornar el nuevo individuo
    # La idea de este crossover es que el largo del prompt se mantenga, es decir el largo del mejor individuo padre. Esto hace que los cambios
    # que se realizan sean mas controlado y se pueda ver el impacto que tiene dentro de la metaheuristica.
    return crear_nuevo_individuo(mejor_individuo, nuevo_prompt, "Crossover con cambios mínimos")


# === Utilidades ===

def crear_nuevo_individuo(individuo, nuevo_prompt, estrategia):
    """
    Crea un nuevo individuo a partir de un prompt generado por un operador de crossover.
    """
    if not validar_prompt(nuevo_prompt):
        nuevo_prompt = "Role: a regular person. Task: Discuss important health topics during lockdown."
    nuevo_individuo = individuo.copy()
    nuevo_individuo["prompt"] = nuevo_prompt
    nuevo_individuo["data_generada"] = ""
    nuevo_individuo["fitness"] = 0
    print(f"\nCrossover realizado: Estrategia: {estrategia}, Prompt generado: {nuevo_prompt}")
    return nuevo_individuo

def dividir_prompt(prompt):
    """
    Divide un prompt en su Role y Task.
    """
    try:
        role_part, task_part = prompt.split(". Task: ", 1)
        role = role_part.replace("Role: ", "").strip()
        task = task_part.strip()
        return role, task
    except ValueError:
        return "Unknown", "This is why..."

def validar_prompt(prompt):
    """
    Verifica si un prompt cumple con la estructura esperada.
    """
    return prompt.startswith("Role: ") and ". Task: " in prompt







# Ejemplo de uso
if __name__ == "__main__":
    individuo_a = {
        "texto_referencia": "La pandemia cambió la forma en que trabajamos.",
        "prompt_inicial": "El teletrabajo se convirtió en una necesidad durante la pandemia.",
        "evaluaciones": [{"data_generada": "...", "fitness": 0.9}]
    }

    individuo_b = {
        "texto_referencia": "La pandemia cambió la forma en que trabajamos.",
        "prompt_inicial": "Las reuniones virtuales reemplazaron a las físicas en la era del COVID-19.",
        "evaluaciones": [{"data_generada": "...", "fitness": 0.85}]
    }

    nuevo_individuo = crossover(individuo_a, individuo_b)
    print("\nNuevo individuo generado:")
    print(nuevo_individuo)

    
    
    ###codigo viejo
#     import random

# def crossover(prompt1, prompt2, strategy="random", split_token=" "):
#     """
#     Realiza crossover entre dos prompts para generar uno nuevo.

#     :param prompt1: El primer prompt.
#     :param prompt2: El segundo prompt.
#     :param strategy: Estrategia de crossover. Opciones: "random", "half".
#     :param split_token: Token para dividir los prompts (por defecto, espacio).
#     :return: Un nuevo prompt generado por crossover.
#     """
#     # Dividir los prompts en palabras o frases
#     parts1 = prompt1.split(split_token)
#     parts2 = prompt2.split(split_token)

#     # Estrategia de crossover
#     if strategy == "random":
#         # Toma palabras/frases aleatorias de ambos prompts
#         new_prompt = [
#             random.choice([word1, word2]) for word1, word2 in zip(parts1, parts2)
#         ]
#     elif strategy == "half":
#         # Usa la primera mitad de prompt1 y la segunda mitad de prompt2
#         split_point1 = len(parts1) // 2
#         split_point2 = len(parts2) // 2
#         new_prompt = parts1[:split_point1] + parts2[split_point2:]
#     else:
#         raise ValueError("Estrategia no reconocida. Usa 'random' o 'half'.")

#     # Combinar las palabras o frases en un único string
#     return split_token.join(new_prompt)


# # Ejemplo de uso
# if __name__ == "__main__":
#     prompt_a = "La inteligencia artificial está revolucionando la tecnología."
#     prompt_b = "El aprendizaje automático es clave para el futuro de la humanidad."
    
#     nuevo_prompt = crossover(prompt_a, prompt_b, strategy="random")
#     print("Prompt generado (estrategia random):", nuevo_prompt)
    
#     nuevo_prompt_half = crossover(prompt_a, prompt_b, strategy="half")
#     print("Prompt generado (estrategia half):", nuevo_prompt_half)