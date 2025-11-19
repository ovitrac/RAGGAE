# tests/test_core_embeddings.py
import numpy as np
from RAGGAE.core.embeddings import STBiEncoder

def test_stbiencoder_basic():
    enc = STBiEncoder("intfloat/multilingual-e5-small", prefix_mode="e5")
    E = enc.embed_texts(["Hello", "Bonjour"])
    q = enc.embed_query("Salut")
    assert E.shape[0] == 2
    assert E.shape[1] == q.shape[0]
    # normalized embeddings ≈ unit length
    norms = np.linalg.norm(E, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)
