from transformers import pipeline

class ContentAnalyzer:
    def __init__(self):
        self.classifier = pipeline("zero-shot-classification")
        self.summarizer = pipeline("summarization")
        
    def extract_topics(self, content, candidate_topics):
        results = self.classifier(
            content,
            candidate_topics,
            multi_label=True
        )
        return dict(zip(results['labels'], results['scores']))
        
    def generate_summary(self, content):
        summary = self.summarizer(content, max_length=130, min_length=30)
        return summary[0]['summary_text']