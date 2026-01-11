import json
import random
import pickle
import numpy as np
import nltk



from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("omw-1.4")

lemmatizer = WordNetLemmatizer()

# Load intents
with open("intents.json", "r") as f:
    data = json.load(f)

patterns = []
tags = []

for intent in data["intents"]:
    tag = intent["tag"]
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(tag)

# Preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalnum()]
    return " ".join(tokens)

patterns_processed = [preprocess_text(p) for p in patterns]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns_processed)

# Train classifier
model = LogisticRegression(max_iter=200)
model.fit(X, tags)

# Save vectorizer + model
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(model, open("model.pkl", "wb"))

print("Training complete!")
print("Saved model.pkl and vectorizer.pkl")
