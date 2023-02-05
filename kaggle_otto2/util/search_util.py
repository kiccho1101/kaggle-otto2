import gc

import faiss
import numpy as np
import torch
from sklearn.preprocessing import normalize


class SearchUtil:
    @staticmethod
    def ann_search(embeddings, query_embeddings, n_neighbors=50):
        index = faiss.IndexFlatIP(embeddings.shape[1])
        if getattr(faiss, "index_cpu_to_gpu", None) is not None:
            index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)

        embeddings = normalize(embeddings)
        embeddings = embeddings.astype(np.float32)
        query_embeddings = normalize(query_embeddings)
        query_embeddings = query_embeddings.astype(np.float32)

        index.add(embeddings)
        distances, indices = index.search(query_embeddings, n_neighbors)

        del index
        torch.cuda.empty_cache()
        gc.collect()

        return distances, indices
