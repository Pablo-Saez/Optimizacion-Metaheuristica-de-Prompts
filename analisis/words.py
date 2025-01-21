import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import nltk

# Descargar recursos necesarios de nltk
nltk.download('punkt')
nltk.download('stopwords')

def cargar_datos_json(archivo_json):
    """
    Cargar datos desde un archivo JSON.
    """
    with open(archivo_json, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def analizar_palabras_clave(df, columna='data_generada'):
    """
    Analiza palabras clave de la columna especificada en el DataFrame.
    """
    # Tokenizar las palabras
    all_words = ' '.join(df[columna]).lower()
    tokens = nltk.word_tokenize(all_words)

    # Filtrar palabras con apóstrofes, palabras específicas y signos de exclamación/pregunta
    tokens = [word for word in tokens if word not in ["'", "ca", "!", "?"]]

    # Contar frecuencias
    word_counts = Counter(tokens)

    # Eliminar palabras irrelevantes (stopwords básicas)
    stopwords = set(nltk.corpus.stopwords.words('english') + ['.', ',', '-', '&', '’', "'s", "!", "?"])
    filtered_counts = {word: count for word, count in word_counts.items() if word not in stopwords}

    return filtered_counts

def graficar_palabras_clave(word_counts, top_n=10):
    """
    Genera un gráfico de las palabras más frecuentes y lo guarda como imagen.
    """
    top_words = Counter(word_counts).most_common(top_n)
    words, counts = zip(*top_words)

    plt.figure(figsize=(10, 6))
    plt.bar(words, counts)
    plt.title('Palabras más frecuentes')
    plt.xlabel('Palabras')
    plt.ylabel('Frecuencia')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Guardar el gráfico como imagen
    plt.savefig('frecuencia_palabras_actualizado.png')
    print("Gráfico guardado como 'frecuencia_palabras_actualizado.png'")

if __name__ == "__main__":
    # Archivo JSON con los datos
    archivo_json = 'data.json'  # Aqui por lo general creaba un archivo a mano, copiaba la ultima generación y la guardaba para poder analizarla.

    # Cargar datos desde el archivo
    df = cargar_datos_json(archivo_json)

    # Analizar palabras clave
    word_counts = analizar_palabras_clave(df, columna='data_generada')

    # Graficar palabras clave
    graficar_palabras_clave(word_counts)