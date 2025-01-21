import json
import nltk
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

# Descargar recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Función para cargar y preprocesar tweets de referencia
def cargar_tweets_referencia(archivo_txt):
    with open(archivo_txt, 'r') as file:
        tweets = file.readlines()
    return [tweet.strip().lower() for tweet in tweets]

# Función para cargar y preprocesar tweets generados
def cargar_tweets_generados(archivo_json):
    with open(archivo_json, 'r') as file:
        data = json.load(file)
    return [entry['data_generada'].strip().lower() for entry in data]

# Función para calcular frecuencias de palabras
def calcular_frecuencias(tweets):
    all_words = ' '.join(tweets)
    tokens = nltk.word_tokenize(all_words)
    stopwords = set(nltk.corpus.stopwords.words('english') + ['.', ',', '-', '&', '’', "'s", '?', '!'])

    filtered_tokens = [word for word in tokens if word not in stopwords]
    return Counter(filtered_tokens)

def graficar_frecuencias_normalizadas(frec_ref, frec_gen, top_n=10):
    # Normalizar frecuencias
    total_ref = sum(frec_ref.values())
    total_gen = sum(frec_gen.values())
    ref_normalizado = {word: count / total_ref for word, count in frec_ref.items()}
    gen_normalizado = {word: count / total_gen for word, count in frec_gen.items()}

    # Obtener las palabras más frecuentes comunes en ambos
    palabras_comunes = set(ref_normalizado.keys()).union(set(gen_normalizado.keys()))
    frec_ref_filtrado = {word: ref_normalizado.get(word, 0) for word in palabras_comunes}
    frec_gen_filtrado = {word: gen_normalizado.get(word, 0) for word in palabras_comunes}

    # Seleccionar top_n palabras comunes
    palabras_top = sorted(frec_ref_filtrado.keys(), key=lambda x: frec_ref_filtrado[x] + frec_gen_filtrado[x], reverse=True)[:top_n]

    # Crear DataFrame para el gráfico
    data = {
        "Palabra": palabras_top,
        "Frecuencia_Referencia": [frec_ref_filtrado[word] for word in palabras_top],
        "Frecuencia_Generados": [frec_gen_filtrado[word] for word in palabras_top]
    }
    df = pd.DataFrame(data)

    # Graficar barras agrupadas
    x = range(len(palabras_top))
    plt.figure(figsize=(12, 6))
    plt.bar(x, df["Frecuencia_Referencia"], width=0.4, label="Referencia", align="center")
    plt.bar([p + 0.4 for p in x], df["Frecuencia_Generados"], width=0.4, label="Generados", align="center")
    plt.xticks([p + 0.2 for p in x], df["Palabra"], rotation=45)
    plt.title("Comparación Normalizada de Frecuencias de Palabras")
    plt.xlabel("Palabras")
    plt.ylabel("Frecuencia Normalizada")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Guardar el gráfico
    plt.savefig("comparacion_frecuencias_normalizadas.png")
    print("Gráfico guardado como 'comparacion_frecuencias_normalizadas.png'")


# Importante este codigo funciona con 2 archivos, el de textos de referencia seleccionas un subset de textos para comparar y los textos que generamos, le puedes dar el json final extraera el campo de data_generada para comparar
archivo_txt_referencia = 'tweets_referencia.txt'  
archivo_json_generados = 'tweets_generados.json'

# Cargar datasets
tweets_referencia = cargar_tweets_referencia(archivo_txt_referencia)
tweets_generados = cargar_tweets_generados(archivo_json_generados)

# Calcular frecuencias
frecuencias_referencia = calcular_frecuencias(tweets_referencia)
frecuencias_generados = calcular_frecuencias(tweets_generados)

# Graficar comparación
graficar_frecuencias_normalizadas(frecuencias_referencia, frecuencias_generados, top_n=10)