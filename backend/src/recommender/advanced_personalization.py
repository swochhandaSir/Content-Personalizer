from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import numpy as np
from typing import List, Dict, Tuple

class AdvancedContentPersonalizer:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        # Initialize base models
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Initialize additional pipelines
        self.classifier = pipeline("zero-shot-classification")
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        
    def generate_content_embedding(self, content: str) -> np.ndarray:
        # Generate embeddings for content
        inputs = self.tokenizer(content, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()
    
    def calculate_content_similarity(self, content1: str, content2: str) -> float:
        # Calculate cosine similarity between two pieces of content
        emb1 = self.generate_content_embedding(content1)
        emb2 = self.generate_content_embedding(content2)
        return np.dot(emb1, emb2.T) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    def classify_content(self, content: str, labels: List[str]) -> Dict[str, float]:
        # Classify content into predefined categories
        result = self.classifier(content, labels)
        return dict(zip(result['labels'], result['scores']))
    
    def analyze_sentiment(self, content: str) -> Dict[str, float]:
        # Analyze the sentiment of the content
        result = self.sentiment_analyzer(content)[0]
        return {'label': result['label'], 'score': result['score']}
    
    def find_similar_content(self, query: str, content_list: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        # Find most similar content items to a query
        query_embedding = self.generate_content_embedding(query)
        similarities = []
        
        for content in content_list:
            content_embedding = self.generate_content_embedding(content)
            similarity = np.dot(query_embedding, content_embedding.T) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(content_embedding)
            )
            similarities.append((content, float(similarity)))
        
        # Sort by similarity score and return top_k
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]