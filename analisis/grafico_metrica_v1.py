import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from evaluate import load

# Cargar la métrica BLEU de Hugging Face
bleu_metric = load("bleu")

def calcular_bleu(referencia, generado):
    """
    Calcula el BLEU entre un texto de referencia y un texto generado.
    """
    resultado = bleu_metric.compute(predictions=[generado], references=[[referencia]])
    return resultado["bleu"]

def calcular_tfidf_similitud(referencia, generado):
    """
    Calcula la similitud TF-IDF entre un texto de referencia y un texto generado.
    """
    vectorizador = TfidfVectorizer()
    tfidf_matrix = vectorizador.fit_transform([referencia, generado])
    similitud = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return similitud[0][0]

def evaluar_tweets_con_referencia(archivo_entrada, archivo_salida, texto_referencia):
    """
    Evalúa los tweets generados en comparación con un único texto de referencia.
    Guarda los resultados con las métricas BLEU y TF-IDF.
    """
    with open(archivo_entrada, 'r') as file:
        tweets = json.load(file)

    resultados = []
    total_bleu = 0
    total_tfidf = 0

    for tweet in tweets:
        generado = tweet["data_generada"]

        # Calcular métricas
        bleu_score = calcular_bleu(texto_referencia, generado)
        tfidf_score = calcular_tfidf_similitud(texto_referencia, generado)
        promedio = (bleu_score + tfidf_score) / 2

        # Acumular métricas
        total_bleu += bleu_score
        total_tfidf += tfidf_score

        # Guardar resultados
        tweet_resultado = {
            "texto_referencia": texto_referencia,
            "data_generada": generado,
            "fitness": tweet["fitness"],
            "bleu_score": bleu_score,
            "tfidf_score": tfidf_score,
            "promedio_metrica": promedio
        }
        resultados.append(tweet_resultado)

    # Calcular promedios
    num_tweets = len(tweets)
    promedio_bleu = total_bleu / num_tweets if num_tweets > 0 else 0
    promedio_tfidf = total_tfidf / num_tweets if num_tweets > 0 else 0

    print(f"Promedio BLEU: {promedio_bleu:.4f}")
    print(f"Promedio TF-IDF: {promedio_tfidf:.4f}")

    # Guardar resultados en un nuevo archivo JSON
    with open(archivo_salida, 'w') as file:
        json.dump(resultados, file, indent=4)
    print(f"Resultados guardados en: {archivo_salida}")

if __name__ == "__main__":
    # Configura el texto de referencia
    texto_referencia = "This quarantine has kicked my depression up a couple notches thanks to my work and routine being void now."

    # Archivos de entrada y salida
    archivo_entrada = "tweets_filtrados.json"
    archivo_salida = "tweets_comparados.json"

    # Evaluar tweets
    evaluar_tweets_con_referencia(archivo_entrada, archivo_salida, texto_referencia)