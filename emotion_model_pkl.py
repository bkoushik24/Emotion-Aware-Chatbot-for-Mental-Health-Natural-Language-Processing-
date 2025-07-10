import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# 1. Sample training data (you can replace with your full dataset)
data = {
    'text': [
        "I'm feeling very happy today!",
        "I feel down and depressed.",
        "You're making me so angry!",
        "I'm afraid of what might happen.",
        "It's just a regular day."
    ],
    'label': ['joy', 'sadness', 'anger', 'fear', 'neutral']
}
df = pd.DataFrame(data)

# 2. Define model pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# 3. Train the model
pipeline.fit(df['text'], df['label'])

# 4. Save the model
os.makedirs('models', exist_ok=True)
with open('models/emotion_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("Model trained and saved to models/emotion_model.pkl")

"""**Step-2: Load the Emotion Model**"""

import pickle

with open('models/emotion_model.pkl', 'rb') as f:
    emotion_model = pickle.load(f)

emotion = emotion_model.predict(["I feel so lonely and sad"])
print("Predicted Emotion:", emotion[0])

"""**Step-3: Responses and Generates the Code**"""

import json
import pickle

# ✅ 1. Load the trained model
with open('models/emotion_model.pkl', 'rb') as f:
    emotion_model = pickle.load(f)

# ✅ 2. Load predefined responses
with open('/content/Emotions.json', 'r') as f:
    responses = json.load(f)

# ✅ 3. Function to generate chatbot response
import random

def get_response(user_input):
    emotion = emotion_model.predict([user_input])[0]
    reply = random.choice(responses.get(emotion, ["I'm here for you."]))
    return emotion, reply

"""**Step-5: Chatbot replies based on feelings and messages**"""

user_input = input("You: ")
emotion, reply = get_response(user_input)
print(f"[{emotion.upper()}] Bot: {reply}")

"""**Step-6: Save Chat Logs**"""

import pandas as pd
from datetime import datetime

log = {
    'timestamp': [datetime.now()],
    'user_input': [user_input],
    'detected_emotion': [emotion],
    'bot_response': [reply]
}

log_df = pd.DataFrame(log)
os.makedirs('outputs', exist_ok=True)
log_df.to_csv('outputs/chat_logs.csv', mode='a', header=not os.path.exists('outputs/chat_logs.csv'), index=False)

print("📝 Chat log saved.")
