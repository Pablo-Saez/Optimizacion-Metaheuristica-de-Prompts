import json

def analizar_progreso(archivo="generaciones.json"):
    """
    Analiza el archivo de progreso de generaciones para determinar si hubo mejoras en promedio.
    
    :param archivo: Nombre del archivo JSON que contiene el progreso de las generaciones.
    :return: None
    """
    try:
        # Cargar el archivo de progreso
        with open(archivo, "r") as file:
            progreso = json.load(file)

        # Variables para el análisis
        generacion_anterior = None
        resumen_generaciones = []

        print("\n=== Análisis de Progreso ===\n")

        for gen_data in progreso:
            generacion = gen_data["generacion"]
            individuos = gen_data["individuos"]

            # Calcular métricas
            fitness_promedio = sum(ind["fitness"] for ind in individuos) / len(individuos)
            fitness_maximo = max(ind["fitness"] for ind in individuos)
            fitness_minimo = min(ind["fitness"] for ind in individuos)

            resumen_generaciones.append({
                "generacion": generacion,
                "fitness_promedio": fitness_promedio,
                "fitness_maximo": fitness_maximo,
                "fitness_minimo": fitness_minimo,
            })

            # Mostrar resultados de la generación actual
            print(f"Generación {generacion}:")
            print(f"  - Fitness Promedio: {fitness_promedio:.4f}")
            print(f"  - Fitness Máximo: {fitness_maximo:.4f}")
            print(f"  - Fitness Mínimo: {fitness_minimo:.4f}")

            # Comparar con la generación anterior
            if generacion_anterior:
                mejora_promedio = fitness_promedio - generacion_anterior["fitness_promedio"]
                mejora_maxima = fitness_maximo - generacion_anterior["fitness_maximo"]
                print(f"  - Cambio Promedio: {'+' if mejora_promedio > 0 else ''}{mejora_promedio:.4f}")
                print(f"  - Cambio Máximo: {'+' if mejora_maxima > 0 else ''}{mejora_maxima:.4f}")
            else:
                print("  - Primera generación, no hay cambios para comparar.")

            generacion_anterior = {
                "fitness_promedio": fitness_promedio,
                "fitness_maximo": fitness_maximo
            }

        print("\n=== Resumen General ===\n")
        print("Generación | Fitness Promedio | Fitness Máximo | Fitness Mínimo")
        for resumen in resumen_generaciones:
            print(f"{resumen['generacion']:>10} | {resumen['fitness_promedio']:.4f} | {resumen['fitness_maximo']:.4f} | {resumen['fitness_minimo']:.4f}")

    except (IOError, json.JSONDecodeError) as e:
        print(f"Error al leer o analizar el archivo {archivo}: {e}")

# Llamar a la función
analizar_progreso("borrar.json")