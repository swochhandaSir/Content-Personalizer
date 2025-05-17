import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # Only show errors
from models.rag_llm_model import RAGLLMModel
from recommender.personalization import ContentPersonalizer

def main():
    # Initialize the vLLM-based personalizer
    rag_model = RAGLLMModel()
    personalizer = ContentPersonalizer(rag_model)
    
    # Example usage with movie content
    prompt = """Based on user preferences for action movies with complex plots,
    recommend similar movies from: The Dark Knight, Inception, The Matrix"""
    
    response = rag_model.generate_response(prompt)
    print("Movie Recommendations:")
    print(response)

if __name__ == "__main__":
    main()
    test_personalization()