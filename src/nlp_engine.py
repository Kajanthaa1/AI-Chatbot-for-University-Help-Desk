import json
import os
import random
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

# Download required NLTK data securely
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')


class ChatBotEngine:
    def __init__(self, intents_path):
        self.intents_path = intents_path
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(tokenizer=self.tokenize_and_lemmatize, stop_words='english', token_pattern=None)
        self.model = LogisticRegression(random_state=42, max_iter=200)
        
        self.intents_data = []
        self.classes = []
        self.X_train_raw = []
        self.y_train = []
        
        self.load_data()
        self.train_model()

    def tokenize_and_lemmatize(self, text):
        # Tokenize and lemmatize text
        tokens = nltk.word_tokenize(text)
        return [self.lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalnum()]

    def load_data(self):
        if not os.path.exists(self.intents_path):
            raise FileNotFoundError(f"Intents file not found at {self.intents_path}")
            
        with open(self.intents_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.intents_data = data['intents']
            
        for intent in self.intents_data:
            tag = intent['tag']
            if tag not in self.classes:
                self.classes.append(tag)
            
            for pattern in intent['patterns']:
                self.X_train_raw.append(pattern)
                self.y_train.append(tag)

    def train_model(self):
        if len(self.X_train_raw) == 0:
            return
            
        # Fit vectorizer and transform text patterns
        X_train_tfidf = self.vectorizer.fit_transform(self.X_train_raw)
        
        # Train classifier
        self.model.fit(X_train_tfidf, self.y_train)

    def get_response_for_intent(self, intent_tag):
        for intent in self.intents_data:
            if intent['tag'] == intent_tag:
                return random.choice(intent['responses'])
        return "I'm not sure how to respond to that."

    def predict_intent(self, text):
        # Transform the user text
        X_test_tfidf = self.vectorizer.transform([text])
        
        # Get probability distribution
        probas = self.model.predict_proba(X_test_tfidf)[0]
        
        # Get highest probability and corresponding intent
        max_idx = np.argmax(probas)
        confidence = probas[max_idx]
        predicted_tag = self.model.classes_[max_idx]
        
        # If confidence is too low, we return a fallback response
        if confidence < 0.25:
            return {
                "intent": "unknown",
                "confidence": float(confidence),
                "response": "I didn't quite understand that. Could you rephrase your question?"
            }
            
        response = self.get_response_for_intent(predicted_tag)
        
        return {
            "intent": predicted_tag,
            "confidence": float(confidence),
            "response": response
        }

if __name__ == "__main__":
    # Test the engine locally
    engine = ChatBotEngine("data/intents.json")
    test_queries = ["When is the exam?", "Exam when?", "what subjects are offered?"]
    for q in test_queries:
        res = engine.predict_intent(q)
        print(f"Q: {q}")
        print(f"Intent: {res['intent']} ({res['confidence']:.2%})")
        print(f"A: {res['response']}\n")
