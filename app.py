import json
import random
import pickle
import streamlit as st
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Load intents
with open("intents.json", "r") as f:
    intents = json.load(f)

# Load model + vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalnum()]
    return " ".join(tokens)

def get_response(user_input):
    cleaned = preprocess_text(user_input)
    X = vectorizer.transform([cleaned])
    tag = model.predict(X)[0]

    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Sorry, I didn't understand 😕"

# -------- Streamlit UI --------
st.set_page_config(page_title="AI Chatbot using NLP", page_icon="🤖")

st.title("🤖 AI Chatbot using NLP")
st.write("Type something and press Enter.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_text = st.chat_input("Ask me something...")

if user_text:
    # show user message
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # bot response
    bot_reply = get_response(user_text)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
