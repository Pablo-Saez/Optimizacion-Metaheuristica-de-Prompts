from bert_score import score

import numpy as np
from evaluate import load
from bert_score import score

#Las primeras 2 funciones son paradigmas que tenia en implementaciones anteriores, pero no se usan en la versión final. 

# def calcular_metricas_bert(respuestas, textos_referencia):
#     """
#     Calcula las métricas BERTScore entre las respuestas generadas y los textos de referencia.
    
#     :param respuestas: Lista de respuestas generadas por el LLM.
#     :param textos_referencia: Lista de textos reales del dataset.
#     :return: Lista de diccionarios con las métricas de precisión, recall y F1 para cada par respuesta-texto.
#     """
#     if len(respuestas) != len(textos_referencia):
#         raise ValueError("Las respuestas y los textos de referencia no tienen el mismo tamaño.")

#     print("\nCalculando métricas BERTScore...")
#     P, R, F1 = score(respuestas, textos_referencia, lang="en", verbose=True)

#     # Construir la lista de métricas individuales
#     metricas_individuales = []
#     for i, (p, r, f1) in enumerate(zip(P.tolist(), R.tolist(), F1.tolist()), 1):
#         metricas_individuales.append({
#             "respuesta": respuestas[i - 1],
#             "texto_referencia": textos_referencia[i - 1],
#             "precision": p,
#             "recall": r,
#             "f1": f1
#         })

#     return metricas_individuales

    
# def calcular_bert_all_vs_all(respuestas, textos_referencia):
#     """
#     Calcula las métricas BERTScore comparando cada texto generado con todos los textos reales seleccionados.
#     Retorna un promedio de precisión, recall y F1 por texto generado.

#     :param respuestas: Lista de respuestas generadas por el LLM.
#     :param textos_referencia: Lista de textos reales del dataset.
#     :return: Diccionario con las métricas promedio para cada respuesta generada.
#     """
#     if not respuestas or not textos_referencia:
#         raise ValueError("Las respuestas y los textos de referencia no pueden estar vacíos.")

#     print("\nCalculando métricas BERTScore (All vs All)...")

#     # Listas para almacenar las métricas promedio por cada texto generado
#     promedios = []

#     for i, respuesta in enumerate(respuestas, 1):
#         print(f"\nComparando la respuesta {i} con todos los textos reales...")
        
#         # Comparar la respuesta actual con todos los textos reales
#         P, R, F1 = score([respuesta] * len(textos_referencia), textos_referencia, lang="en", verbose=False)
        
#         # Calcular promedios para esta respuesta
#         precision_promedio = P.mean().item()
#         recall_promedio = R.mean().item()
#         f1_promedio = F1.mean().item()

#         # Mostrar resultados para esta respuesta
#         # print(f"  Precision Promedio: {precision_promedio:.4f}")
#         # print(f"  Recall Promedio: {recall_promedio:.4f}")
#         # print(f"  F1 Promedio: {f1_promedio:.4f}")

#         # Guardar los promedios para esta respuesta
#         promedios.append({
#             "respuesta": respuesta,
#             "precision_promedio": precision_promedio,
#             "recall_promedio": recall_promedio,
#             "f1_promedio": f1_promedio
#         })

#     # Mostrar promedios generales
#     # print("\n=== Promedios Generales ===")
#     precision_global = np.mean([p["precision_promedio"] for p in promedios])
#     recall_global = np.mean([p["recall_promedio"] for p in promedios])
#     f1_global = np.mean([p["f1_promedio"] for p in promedios])

#     # print(f"Precision Global Promedio: {precision_global:.4f}")
#     # print(f"Recall Global Promedio: {recall_global:.4f}")
#     # print(f"F1 Global Promedio: {f1_global:.4f}")

#     return {
#         "promedios_por_respuesta": promedios,
#         "precision_global": precision_global,
#         "recall_global": recall_global,
#         "f1_global": f1_global
#     }
    


def calcular_metricas_bert_huggingface(individuo):
    """
    Calcula las métricas BERTScore entre la última data generada y el texto de referencia
    utilizando el modelo 'bert-base-uncased' de Hugging Face.

    :param individuo: Diccionario que contiene `evaluaciones` y `texto_referencia`.
    :return: Diccionario con las métricas de precisión, recall y F1.
    """
   
    

    # Tomar la última evaluación generada
    data_generada = individuo["data_generada"]
    texto_referencia = individuo["texto_referencia"]
    
    print(f'Evaluando : {data_generada} vs {texto_referencia} !!!!!!!!!!')

    print("\nCalculando métricas BERTScore (HuggingFace)...")

    # Cargar la métrica BERTScore desde Hugging Face
    bertscore = load("bertscore")

    # Calcular métricas
    resultados = bertscore.compute(
        predictions=[data_generada],
        references=[texto_referencia],
        model_type="bert-base-uncased",
        lang="en"
    )

    # Construir el diccionario de métricas
    metricas = {
        "precision": resultados["precision"][0],
        "recall": resultados["recall"][0],
        "f1": resultados["f1"][0]
    }

    print("\nResultados BERTScore:")
    print(f"Precision: {metricas['precision']:.4f}")
    print(f"Recall: {metricas['recall']:.4f}")
    print(f"F1: {metricas['f1']:.4f}")

    return metricas