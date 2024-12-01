import json
import numpy as np
import os

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search


SYMMETRIC_EMBEDDER = None
SYMMETRIC_CORPUS = None

ASYMMETRIC_EMBEDDER = None
ASYMMETRIC_CORPUS = None

CORPUS = None


def init():
    global SYMMETRIC_EMBEDDER, ASYMMETRIC_EMBEDDER, SYMMETRIC_CORPUS, ASYMMETRIC_CORPUS, CORPUS
    print("loading symmetric model")
    SYMMETRIC_EMBEDDER = SentenceTransformer('/root/.cache/all-MiniLM-L6-v2')
    print("loading asymmetric model")
    ASYMMETRIC_EMBEDDER = SentenceTransformer('/root/.cache/msmarco-distilbert-dot-v5')
    print("done")

    corpus = []

    with open("./templates.mdtemplates.json") as file:
        data = json.loads(file.read())
        CORPUS = [d["combinedField"] for d in data]

    print("computing symmetric embeddings")
    file = "./embeddings/symm_embeddings.npy"
    if not os.path.isfile(file):
        SYMMETRIC_CORPUS = SYMMETRIC_EMBEDDER.encode(corpus)
        np.save(file, np.array(SYMMETRIC_CORPUS))
    else:
        SYMMETRIC_CORPUS = np.load(file)


    print("computing asymmetric embeddings")
    file = "./embeddings/asymm_embeddings.npy"
    if not os.path.isfile(file):
        ASYMMETRIC_CORPUS = ASYMMETRIC_EMBEDDER.encode(corpus)
        np.save(file, np.array(ASYMMETRIC_CORPUS))
    else:
        ASYMMETRIC_CORPUS = np.load(file)
    print("done")


def symmetric_search(query: str):
    if not SYMMETRIC_EMBEDDER:
        init()

    query_embeddings = SYMMETRIC_EMBEDDER.encode(query)
    search = semantic_search(query_embeddings, SYMMETRIC_CORPUS)

    result = []
    for query in search:
        for res in query:
            result.append((res["score"], CORPUS[res["corpus_id"]]))
    return result

def asymmetric_search(query: str):
    if not ASYMMETRIC_EMBEDDER:
        init()

    query_embeddings = ASYMMETRIC_EMBEDDER.encode(query)
    search = semantic_search(query_embeddings, ASYMMETRIC_CORPUS)

    result = []
    for query in search:
        for res in query:
            result.append((res["score"], CORPUS[res["corpus_id"]]))
    return result