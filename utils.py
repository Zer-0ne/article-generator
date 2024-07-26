import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import cohere

co = cohere.Client("T7E5YdqYVduosUnRrTAGvimDFbrSXFSdUOmk3nHA")

def get_similarity(target, candidates):
    candidates = np.array(candidates)
    target = np.expand_dims(np.array(target), axis=0)
    similarity_scores = cosine_similarity(target, candidates)
    similarity_scores = np.squeeze(similarity_scores).tolist()
    similarity_scores = list(enumerate(similarity_scores))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    return similarity_scores

def classify_text(texts, examples):
    classifications = co.classify(inputs=texts, examples=examples)
    return [c.prediction for c in classifications.classifications]

def extract_tags(article):
    prompt = f"""Given an article, extract a list of tags containing keywords of that article.

    Article: {article}

    Tags:"""
    response = co.generate(
        model='command-r',
        prompt=prompt
    )
    return response.generations[0].text
