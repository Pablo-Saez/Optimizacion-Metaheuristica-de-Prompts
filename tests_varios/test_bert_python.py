from bert_score import score

def calcular_bert_score(texto1, texto2):
    """
    Calcula el BERTScore entre dos textos.
    
    :param texto1: Primer texto (predicción o respuesta generada).
    :param texto2: Segundo texto (texto de referencia).
    :return: Diccionario con precisión, recall y F1.
    """
    print("\nCalculando métricas BERTScore...")
    P, R, F1 = score([texto1], [texto2], lang="en", verbose=False)

    # Extraer las métricas de las listas
    precision = P.item()
    recall = R.item()
    f1 = F1.item()

    # Imprimir los resultados
    print(f"Texto 1: {texto1}")
    print(f"Texto 2: {texto2}")
    print(f"Precisión: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

if __name__ == "__main__":
    # Textos definidos directamente en el código
    texto1 = "hello how are you"
    texto2 = "das"



    # Calcular y mostrar el BERTScore
    resultados = calcular_bert_score(texto1, texto2)

    print("\nResultados:")
    print(f"Precisión: {resultados['precision']:.4f}")
    print(f"Recall: {resultados['recall']:.4f}")
    print(f"F1 Score: {resultados['f1']:.4f}")