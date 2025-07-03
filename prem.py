from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
import json

app = Flask(__name__)
CORS(app)


class ERPChatbot:
    def __init__(self, knowledge_path='knowledge_graph.json'):
        self.vectorizer = TfidfVectorizer()
        self.knowledge_graph = self.load_knowledge_graph(knowledge_path)
        self._prepare_embeddings()

    def load_knowledge_graph(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def _prepare_embeddings(self):
        all_patterns = []
        for intent in self.knowledge_graph['intents']:
            all_patterns.extend(intent['patterns'])
            for sub in intent.get('subintents', []):
                all_patterns.extend(sub['patterns'])

        self.vectorizer.fit(all_patterns)

        for intent in self.knowledge_graph['intents']:
            intent['patterns_embeddings'] = self.vectorizer.transform(intent['patterns'])
            for sub in intent.get('subintents', []):
                sub['patterns_embeddings'] = self.vectorizer.transform(sub['patterns'])

    def analyze_sentiment(self, text):
        lower = text.lower()
        if any(word in lower for word in ['good', 'great', 'excellent', 'happy']):
            return {'label': 'POSITIVE', 'score': 0.9}
        elif any(word in lower for word in ['bad', 'terrible', 'sad', 'angry']):
            return {'label': 'NEGATIVE', 'score': 0.9}
        return {'label': 'NEUTRAL', 'score': 0.6}

    @lru_cache(maxsize=100)
    def vectorize_query(self, text):
        return self.vectorizer.transform([text.lower()])

    def classify_intent(self, text):
        normalized_text = text.lower().strip()
        text_embedding = self.vectorize_query(normalized_text)

        # 1. Exact match with subintent
        for intent in self.knowledge_graph['intents']:
            for sub in intent.get('subintents', []):
                if normalized_text in [p.lower() for p in sub['patterns']]:
                    return {"intent": intent['intent'], "subintent": sub['subintent'], "confidence": 1.0}

        # 2. Exact match with intent
        for intent in self.knowledge_graph['intents']:
            if normalized_text in [p.lower() for p in intent['patterns']]:
                return {"intent": intent['intent'], "subintent": None, "confidence": 1.0}

        # 3. Similarity-based fallback
        best_intent = None
        best_subintent = None
        highest_intent_sim = 0
        highest_subintent_sim = 0

        for intent in self.knowledge_graph['intents']:
            intent_sim = cosine_similarity(text_embedding, intent['patterns_embeddings']).max()
            if intent_sim > highest_intent_sim:
                highest_intent_sim = intent_sim
                best_intent = intent

            for sub in intent.get('subintents', []):
                sub_sim = cosine_similarity(text_embedding, sub['patterns_embeddings']).max()
                if sub_sim > highest_subintent_sim:
                    highest_subintent_sim = sub_sim
                    best_subintent = {'intent': intent['intent'], 'subintent': sub['subintent']}

        if highest_subintent_sim >= 0.65 and highest_subintent_sim > highest_intent_sim:
            return {
                "intent": best_subintent['intent'],
                "subintent": best_subintent['subintent'],
                "confidence": float(highest_subintent_sim)
            }

        if best_intent:
            return {
                "intent": best_intent['intent'],
                "subintent": None,
                "confidence": float(highest_intent_sim)
            }

        return {"intent": None, "subintent": None, "confidence": 0.0}


# Initialize chatbot once
chatbot = ERPChatbot()


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        message = data.get('message', '').strip()

        if not message:
            return jsonify({"error": "Message is required."}), 400

        sentiment = chatbot.analyze_sentiment(message)
        intent_data = chatbot.classify_intent(message)

        return jsonify({
            "intent": intent_data['intent'],
            "subintent": intent_data.get('subintent'),
            "confidence": intent_data['confidence'],
            "sentiment": sentiment['label'],
            "sentiment_score": sentiment['score']
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
