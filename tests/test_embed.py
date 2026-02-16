import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from embed_profiles import embed_texts


def test_embed_texts_returns_correct_shape():
    texts = ["Machine learning and AI research", "Climate change policy analysis"]
    embeddings = embed_texts(texts)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] == 384


def test_embed_texts_similar_texts_closer():
    texts = [
        "Machine learning and deep neural networks",
        "Artificial intelligence and deep learning",
        "Medieval French poetry and literature",
    ]
    embeddings = embed_texts(texts)
    sims = cosine_similarity(embeddings)
    # ML topics should be closer to each other than to poetry
    assert sims[0, 1] > sims[0, 2]
