import os

from langchain_milvus import Milvus

from .embeddings import embeddings

MILVUS_URI = os.getenv("MILVUS_URI", "localhost:19530")


def build_vs(collection_name: str) -> Milvus:
    return Milvus(
        embedding_function=embeddings,
        collection_name=collection_name,
        connection_args={"uri": MILVUS_URI},
        auto_id=True,                 
        index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE"},
        search_params={"metric_type": "COSINE"},
        drop_old=False,               
    )





