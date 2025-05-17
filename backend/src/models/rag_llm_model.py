from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import torch
import requests
import json

class RAGLLMModel:
    def __init__(self, model_name="mistralai/Mistral-7B-v0.1"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.embeddings = HuggingFaceEmbeddings()
        self.vector_store = FAISS.from_texts([], self.embeddings)
        
    def add_to_knowledge_base(self, texts):
        self.vector_store.add_texts(texts)
        
    def generate_response(self, query, user_context, k=3):
        # Retrieve relevant context
        relevant_docs = self.vector_store.similarity_search(query, k=k)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # Construct prompt with user context and retrieved information
        prompt = f"""Based on the user's profile and preferences:
{user_context}

And considering this relevant information:
{context}

Query: {query}
Response:"""
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_length=200,
            temperature=0.7,
            top_p=0.9
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class RAGLLMModel:
    def __init__(self, server_url="http://localhost:8000"):
        self.server_url = server_url
        
    def generate_response(self, prompt, max_tokens=512, temperature=0.5):
        response = requests.post(
            f"{self.server_url}/v1/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": "souldrr/Llama-3.2-1B-fine-tune-300-movies-50-review",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        )
        return response.json()["choices"][0]["text"]