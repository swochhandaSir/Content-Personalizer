class DynamicUserProfile:
    def __init__(self):
        self.interests = {}
        self.content_history = []
        self.interaction_weights = {}
        
    def update_profile(self, content, interaction_type, timestamp):
        # Extract content features using NLP
        features = self.extract_content_features(content)
        
        # Update interest weights based on interaction
        for feature in features:
            current_weight = self.interests.get(feature, 0)
            time_decay = self.calculate_time_decay(timestamp)
            interaction_weight = self.get_interaction_weight(interaction_type)
            
            self.interests[feature] = current_weight * time_decay + interaction_weight
            
    def get_personalized_score(self, content):
        features = self.extract_content_features(content)
        return sum(self.interests.get(feature, 0) for feature in features)