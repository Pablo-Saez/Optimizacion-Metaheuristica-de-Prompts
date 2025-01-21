from transformers import PegasusForConditionalGeneration, PegasusTokenizer

def mutacion(individuo, num_return_sequences=1, num_beams=10, temperature=0.7):
    """
    Aplica mutación al prompt de un individuo utilizando Pegasus para generar un parafraseo.

    :param individuo: Diccionario que representa al individuo.
    :param num_return_sequences: Número de parafraseos a generar.
    :param num_beams: Número de beams utilizados en el modelo Pegasus.
    :param temperature: Controla la aleatoriedad del parafraseo.
    :return: Diccionario del individuo con el prompt mutado.
    """
    try:
        # Cargar el modelo y el tokenizer de Pegasus
        model_name = "tuner007/pegasus_paraphrase"
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        model = PegasusForConditionalGeneration.from_pretrained(model_name)

        # Extraer el prompt del individuo
        prompt = individuo["prompt"]

        # Tokenizar el texto de entrada
        tokens = tokenizer([prompt], truncation=True, padding="longest", return_tensors="pt")

        # Generar los parafraseos utilizando el modelo Pegasus
        paraphrase_ids = model.generate(
            **tokens,
            max_length=60,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            temperature=temperature
        )

        # Decodificar la primera secuencia generada
        paraphrased_prompt = tokenizer.decode(paraphrase_ids[0], skip_special_tokens=True)

        # Imprimir el parafraseo generado (opcional, para depuración)
        print("\n=== Mutación ===")
        print(f"Prompt original: {prompt}")
        print(f"Prompt mutado: {paraphrased_prompt}")

        # Crear una copia del individuo y actualizar el prompt
        individuo_mutado = individuo.copy()
        individuo_mutado["prompt"] = paraphrased_prompt

        # Reiniciar los valores generados y el fitness
        individuo_mutado["data_generada"] = ""
        individuo_mutado["fitness"] = 0

        return individuo_mutado

    except Exception as e:
        print(f"Error durante la mutación por parafraseo: {e}")
        return individuo  # Retornar el individuo original en caso de error



# Ejemplo de uso
if __name__ == "__main__":
    individuo = {
        "texto_referencia": "if i dont die of corona ill die of distance learning",
        "prompt_inicial": "Write a tweet that humorously expresses the frustration of being stuck between the struggles of surviving a pandemic and navigating online education.",
        "evaluaciones": [{"data_generada": "...", "fitness": 0.85}]
    }

    individuo_mutado = mutar(individuo)
    print("\nIndividuo después de la mutación:")
    print(individuo_mutado)




###codigo viejo:

# def mutar_parafraseo(prompt, num_return_sequences=3, num_beams=10, temperature=0.7):
#     """
#     Genera parafraseos de un prompt utilizando el modelo Pegasus.
    
#     :param prompt: El texto original del prompt a parafrasear.
#     :param num_return_sequences: Número de parafraseos a generar.
#     :param num_beams: Número de beams utilizados en el modelo Pegasus.
#     :param temperature: Controla la aleatoriedad del parafraseo.
#     :return: Lista de prompts parafraseados.
#     """
#     try:
#         # Verificar si SentencePiece está instalado
#         try:
#             import sentencepiece
#         except ImportError:
#             raise ImportError(
                
#             )

#         # Cargar el modelo y el tokenizer
#         model_name = "tuner007/pegasus_paraphrase"
#         tokenizer = PegasusTokenizer.from_pretrained(model_name)
#         model = PegasusForConditionalGeneration.from_pretrained(model_name)

#         # Tokenizar el texto de entrada
#         tokens = tokenizer([prompt], truncation=True, padding="longest", return_tensors="pt")
        
#         # Generar los parafraseos utilizando el modelo Pegasus
#         paraphrase_ids = model.generate(
#             **tokens,
#             max_length=60,
#             num_beams=num_beams,
#             num_return_sequences=num_return_sequences,
#             temperature=temperature
#         )
        
#         # Decodificar las secuencias generadas y retornarlas
#         paraphrases = tokenizer.batch_decode(paraphrase_ids, skip_special_tokens=True)

#         # Imprimir el prompt original y los parafraseos generados
#         print("\n=== Parafraseo ===")
#         print(f"Prompt original: {prompt}")
#         for i, parafraseo in enumerate(paraphrases, 1):
#             print(f"Parafraseo {i}: {parafraseo}")

#         return paraphrases

#     except ImportError as e:
#         print(f"Error crítico: {e}")
#         return []
#     except Exception as e:
#         print(f"Error durante la mutación por parafraseo: {e}")
#         return []