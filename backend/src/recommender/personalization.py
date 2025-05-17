from models.rag_llm_model import RAGLLMModel

class ContentPersonalizer:
    def __init__(self):
        self.rag_model = RAGLLMModel()
        self.user_profiles = {}
        
    def update_user_profile(self, user_id, interaction):
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = []
        self.user_profiles[user_id].append(interaction)
        
    def get_user_context(self, user_id):
        if user_id not in self.user_profiles:
            return "New user with no preference history."
        
        interactions = self.user_profiles[user_id]
        return f"User has shown interest in: {', '.join(i['content_type'] for i in interactions)}"
        
    def get_personalized_recommendations(self, user_id, query, num_recommendations=3):
        user_context = self.get_user_context(user_id)
        response = self.rag_model.generate_response(query, user_context)
        return response