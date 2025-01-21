import json
import matplotlib.pyplot as plt


### Este codigo es para extraer los datos generados en el intervalo correcto de fitness (bert), el cual estipule que fuera entre 0.4 y 0.7
def cargar_tweets_generados(archivo_json):
    """
    Carga tweets generados desde un archivo JSON.
    """
    with open(archivo_json, 'r') as file:
        return json.load(file)

def filtrar_tweets_por_fitness(tweets, min_fitness=0.4, max_fitness=0.7):
    """
    Filtra los tweets generados por un rango de fitness.
    """
    return [tweet for tweet in tweets if min_fitness <= tweet["fitness"] <= max_fitness]

def guardar_json(datos, nombre_archivo):
    """
    Guarda los datos en un archivo JSON.
    """
    with open(nombre_archivo, 'w') as file:
        json.dump(datos, file, indent=4)
    print(f"Archivo guardado como: {nombre_archivo}")

def graficar_fitness_prompts(tweets, titulo, nombre_archivo):
    """
    Genera un gráfico de barras con el fitness en el eje Y y los prompts numerados en el eje X.
    """
    fitness = [tweet["fitness"] for tweet in tweets]
    labels = [f"Prompt {i+1}" for i in range(len(tweets))]
    
    # Crear gráfico
    plt.figure(figsize=(10, 6))
    plt.bar(labels, fitness, color='skyblue')
    plt.title(titulo)
    plt.xlabel("Prompts")
    plt.ylabel("Fitness")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Guardar gráfico
    plt.savefig(nombre_archivo)
    print(f"Gráfico guardado como: {nombre_archivo}")
    plt.close()

if __name__ == "__main__":
    # Nombre del archivo JSON con los datos originales
    archivo_json = "tweets.json"
    
    # Cargar los tweets generados
    tweets_generados = cargar_tweets_generados(archivo_json)
    
    # Crear gráficos antes del filtrado
    graficar_fitness_prompts(tweets_generados, 
                             "Distribución de fitness antes del filtrado", 
                             "fitness_antes.png")
    
    # Filtrar tweets con fitness entre 0.4 y 0.7
    tweets_filtrados = filtrar_tweets_por_fitness(tweets_generados)
    
    # Guardar los tweets filtrados en un nuevo archivo JSON
    guardar_json(tweets_filtrados, "tweets_filtrados.json")
    
    # Crear gráficos después del filtrado
    graficar_fitness_prompts(tweets_filtrados, 
                             "Distribución de fitness después del filtrado", 
                             "fitness_despues.png")