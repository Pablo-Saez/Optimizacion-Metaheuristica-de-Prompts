import json
import random
from collections import Counter
import matplotlib.pyplot as plt

# Keywords a analizar (actualizado)
keywords = {"corona", "coronavirus", "covid", "#covid", "covid19", "#covid19", 
            "quarantine", "pneumonia", "wuhan", "virus", "cough", "fever", "lockdown",
            "pandemic", "isolation", "loneliness", "safety", "travel", "performer", "shows",
            "india", "well-being", "chaos", "stay safe", "social distancing", "campus", 
            "college", "freshman", "online classes", "friends", "connection", "cancelled plans", 
            "zoom", "memories", "events"}

# Función para calcular la frecuencia de palabras clave
def calcular_frecuencia_keywords(textos, keywords):
    """
    Calcula la frecuencia de palabras clave en un conjunto de textos.
    Filtra palabras con frecuencia cero.
    """
    todas_las_palabras = " ".join(textos).lower().split()
    conteo = Counter(todas_las_palabras)
    return {palabra: conteo[palabra] for palabra in keywords if conteo[palabra] > 0}

# Función para graficar la frecuencia de palabras clave
def graficar_frecuencia_keywords(frecuencias, titulo, nombre_archivo):
    """
    Genera un gráfico de barras para la frecuencia de palabras clave.
    Muestra solo las 10 palabras más frecuentes, ordenadas de mayor a menor.
    """
    if not frecuencias:
        print(f"No hay palabras clave con coincidencias en {titulo}. Gráfico no generado.")
        return

    # Ordenar las palabras clave por frecuencia y seleccionar las 10 más frecuentes
    frecuencias_ordenadas = dict(sorted(frecuencias.items(), key=lambda x: x[1], reverse=True)[:10])
    
    etiquetas, valores = zip(*frecuencias_ordenadas.items())
    plt.figure(figsize=(12, 8))
    colores = ['red' if palabra in keywords else 'blue' for palabra in etiquetas]  # Resaltar palabras clave
    plt.bar(etiquetas, valores, color=colores)
    plt.title(titulo)
    plt.xlabel("Palabras clave")
    plt.ylabel("Frecuencia")
    plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()
    plt.savefig(nombre_archivo)
    plt.close()
    print(f"Gráfico guardado como {nombre_archivo}")

# Cargar datos
with open("dataset.txt", "r", encoding="utf-8") as f:
    dataset1 = [line.strip() for line in f.readlines()]

with open("tweets_filtrados.json", "r", encoding="utf-8") as f:
    dataset2 = json.load(f)

# Extraer data generada
dataset2_generated = [entry["data_generada"] for entry in dataset2]

# Seleccionar 50 datos aleatorios de cada conjunto
random_txt_samples = random.sample(dataset1, min(200, len(dataset1)))
random_json_samples = random.sample(dataset2_generated, min(200, len(dataset2_generated)))

# Calcular frecuencias de palabras clave
frecuencias_txt = calcular_frecuencia_keywords(random_txt_samples, keywords)
frecuencias_json = calcular_frecuencia_keywords(random_json_samples, keywords)

# Graficar frecuencias de palabras clave
graficar_frecuencia_keywords(frecuencias_txt, "Frecuencia de Palabras Clave Dataset de referencia (200 Muestras)", 
                             "frecuencia_keywords_txt.png")
graficar_frecuencia_keywords(frecuencias_json, "Frecuencia de Palabras Clave Dataset generado (200 Muestras)", 
                             "frecuencia_keywords_json.png")