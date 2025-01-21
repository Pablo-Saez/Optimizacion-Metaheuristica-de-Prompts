import json
import matplotlib.pyplot as plt
import numpy as np

def graficar_metricas_por_prompt(archivo_entrada, nombre_grafico):
    """
    Genera un gráfico de líneas con puntos para las métricas fitness, BLEU y TF-IDF por cada prompt y lo guarda.
    """
    with open(archivo_entrada, 'r') as file:
        tweets = json.load(file)

    # Extraer datos
    prompts = [f"Prompt {i+1}" for i in range(len(tweets))]
    fitness = [tweet["fitness"] for tweet in tweets]
    bleu_scores = [tweet["bleu_score"] for tweet in tweets]
    tfidf_scores = [tweet["tfidf_score"] for tweet in tweets]

    # Crear índices para agrupar datos
    x_indices = np.arange(1, len(prompts) + 1)

    # Crear el gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(x_indices, fitness, label="Fitness", color="blue", marker='o', linestyle='-', linewidth=2)
    plt.plot(x_indices, bleu_scores, label="BLEU", color="green", marker='o', linestyle='-', linewidth=2)
    plt.plot(x_indices, tfidf_scores, label="TF-IDF", color="red", marker='o', linestyle='-', linewidth=2)

    # Configuración del gráfico
    plt.title("Comparación de métricas por Prompt", fontsize=16)
    plt.xlabel("Prompts", fontsize=14)
    plt.ylabel("Valor de la Métrica", fontsize=14)
    plt.xticks(x_indices, prompts, rotation=45, fontsize=10)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12, loc="upper left")  # Leyenda en la esquina superior izquierda
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Guardar el gráfico
    plt.tight_layout()
    plt.savefig(nombre_grafico)
    print(f"Gráfico guardado como: {nombre_grafico}")
    plt.close()

if __name__ == "__main__":
    # Archivo con los datos evaluados
    archivo_entrada = "tweets_comparados.json"
    nombre_grafico = "metricas_por_prompt.png"  # Nombre del archivo del gráfico

    # Generar gráfico
    graficar_metricas_por_prompt(archivo_entrada, nombre_grafico)