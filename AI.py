import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from ai_utils import clean_text

model = load_model("sms_lstm_model.h5")

with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

max_len = 100

def predict_text(text):
    clean = clean_text(text)
    seq = tokenizer.texts_to_sequences([clean])
    seq_pad = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    prob = model.predict(seq_pad, verbose=0)[0][0]
    label = "spam" if prob >= 0.5 else "ham"
    return {"label": label, "probability": float(prob)}

st.title("SMS Spam Detection")
user_input = st.text_area("Enter your message:")
if st.button("Predict"):
    if user_input.strip() != "":
        result = predict_text(user_input)
        st.write(result)
    else:
        st.write("Please enter a message to predict.")

