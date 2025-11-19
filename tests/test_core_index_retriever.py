# tests/test_core_index_retriever.py
from RAGGAE.core.embeddings import STBiEncoder
from RAGGAE.core.retriever import HybridRetriever

def test_hybrid_retriever_search():
    texts = [
        "MLOps with MLflow on Kubernetes (K8s)",
        "ISO 27001 certification is mandatory",
        "GitOps with ArgoCD and Helm charts",
    ]
    enc = STBiEncoder("intfloat/multilingual-e5-small", prefix_mode="e5")
    retr = HybridRetriever.build(enc, texts, metadatas=[{"page":1},{"page":2},{"page":3}])
    hits = retr.search("MLflow sur K8s", k=2)
    assert len(hits) == 2
    # top hit should be about MLflow
    assert "mlflow" in hits[0].text.lower()
