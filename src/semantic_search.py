import json
import numpy as np
import os

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search

print("loading symmetric model")
symm_embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("loading asymmetric model")
asymm_embedder = SentenceTransformer("msmarco-distilbert-dot-v5")
print("done")

corpus = []

with open("./templates.mdtemplates.json") as file:
    data = json.loads(file.read())
    corpus = [d["combinedField"] for d in data]

print("computing symmetric embeddings")
file = "./embeddings/symm_embeddings.npy"
if not os.path.isfile(file):
    symm_corpus_embeddings = symm_embedder.encode(corpus)
    np.save(file, np.array(symm_corpus_embeddings))
else:
    symm_corpus_embeddings = np.load(file)


print("computing asymmetric embeddings")
file = "./embeddings/asymm_embeddings.npy"
if not os.path.isfile(file):
    asymm_corpus_embeddings = asymm_embedder.encode(corpus)
    np.save(file, np.array(asymm_corpus_embeddings))
else:
    asymm_corpus_embeddings = np.load(file)

print("done")


def symmetric_search(query: str):
    query_embeddings = symm_embedder.encode(query)
    search = semantic_search(query_embeddings, symm_corpus_embeddings)

    result = []
    for query in search:
        for res in query:
            result.append((res["score"], corpus[res["corpus_id"]]))
    return result

def asymmetric_search(query: str):
    query_embeddings = asymm_embedder.encode(query)
    search = semantic_search(query_embeddings, asymm_corpus_embeddings)

    result = []
    for query in search:
        for res in query:
            result.append((res["score"], corpus[res["corpus_id"]]))#
    return result