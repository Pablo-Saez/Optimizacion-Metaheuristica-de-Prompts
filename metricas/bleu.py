
import nltk
from nltk.translate.bleu_score import sentence_bleu
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

nltk.download('punkt')

def calculate_bleu(reference, generated):
    """
    Calcula la métrica BLEU para un texto de referencia y un texto generado.
    """
    reference_tokens = [nltk.word_tokenize(reference.lower())]
    generated_tokens = nltk.word_tokenize(generated.lower())
    bleu_score = sentence_bleu(reference_tokens, generated_tokens)
    return bleu_score

def calculate_tfidf(reference, generated):
    """
    Calcula la similitud TF-IDF entre un texto de referencia y un texto generado.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([reference, generated])
    similarity = (tfidf_matrix * tfidf_matrix.T).toarray()[0, 1]
    return similarity


