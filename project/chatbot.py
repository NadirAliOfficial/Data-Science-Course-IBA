import streamlit as st
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------
# 1. SAMPLE DATA
# ---------------------------------------------------
# A tiny dataset of mental-health-related statements
# and a simple sentiment label: 'negative', 'neutral', or 'positive'
data_texts = [
    "I feel terrible. My anxiety is so high.",
    "Life is okay, I'm doing alright I guess.",
    "I'm feeling wonderful and full of energy!",
    "I'm so depressed, I can't stop crying.",
    "Nothing is good or bad, just neutral I suppose.",
    "I am content, nothing special.",
    "I'm extremely happy today!",
    "I hate everything about my life.",
    "My mood is not good, but not bad either.",
    "This is the best day I've had all year."
]
data_labels = [
    "negative",
    "neutral",
    "positive",
    "negative",
    "neutral",
    "neutral",
    "positive",
    "negative",
    "neutral",
    "positive"
]

# ---------------------------------------------------
# 2. TRAIN A SIMPLE CLASSIFIER
# ---------------------------------------------------
# We'll do everything in-memory (no pipeline from huggingface).
# Step (a): Vectorize the texts with TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data_texts)

# Step (b): Train logistic regression
clf = LogisticRegression()
clf.fit(X, data_labels)

# We have a minimal dictionary of "canned" responses:
responses = {
    "negative": [
        "I'm sorry to hear that you're feeling down. You might consider reaching out to a mental health professional.",
        "It seems you're going through a tough time. Remember, help is always available."
    ],
    "neutral": [
        "Thanks for sharing how you feel. If you’d like to talk more, I’m here.",
        "I understand. Feel free to share more if you want."
    ],
    "positive": [
        "I'm glad you’re feeling good! Keep up that positive mindset.",
        "Awesome! It sounds like you’re in a good place right now."
    ]
}

def classify_text(text):
    """
    Return the predicted sentiment: 'negative', 'neutral', or 'positive'.
    """
    text_vector = vectorizer.transform([text])
    prediction = clf.predict(text_vector)[0]
    return prediction

# ---------------------------------------------------
# 3. STREAMLIT APP
# ---------------------------------------------------
def main():
    st.title("Simple Mental Health Sentiment App")
    st.write(
        "This is a demonstration of a **basic sentiment classifier** "
        "for mental health-related text. **Not** medical advice!"
    )

    # Keep conversation in session state
    if "history" not in st.session_state:
        st.session_state.history = []

    user_input = st.text_input("Your message:", "")
    if st.button("Send") and user_input.strip():
        # Classify the user input
        sentiment_label = classify_text(user_input)
        
        # Add user query to history
        st.session_state.history.append(("You", user_input))
        
        # Get a suitable response
        # In a real system, you might randomize or generate a dynamic response
        response = responses[sentiment_label][0]
        
        # Add bot response to history
        st.session_state.history.append(("Bot", response))

    # Display conversation
    for speaker, msg in st.session_state.history:
        st.markdown(f"**{speaker}:** {msg}")

if __name__ == "__main__":
    main()
