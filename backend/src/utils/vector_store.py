import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

class VectorStore:
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.content_mapping = {}

    def add_content(self, content_list):
        embeddings = self.embedding_model.encode(content_list)
        self.index.add(np.array(embeddings).astype('float32'))
        
        for idx, content in enumerate(content_list):
            self.content_mapping[idx] = content

    def search(self, query, k=5):
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(
            np.array(query_embedding).astype('float32'), k
        )
        return [(self.content_mapping[idx], dist) for idx, dist in zip(indices[0], distances[0])]