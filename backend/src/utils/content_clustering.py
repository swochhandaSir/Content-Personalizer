from sklearn.cluster import DBSCAN
import numpy as np

class ContentClusterer:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.clusterer = DBSCAN(eps=0.3, min_samples=2)
        
    def cluster_similar_content(self, content_list):
        embeddings = np.array([
            self.embedding_model.generate_content_embedding(content)
            for content in content_list
        ])
        
        clusters = self.clusterer.fit_predict(embeddings)
        return clusters