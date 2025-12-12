
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from ai_utils import clean_text

model = load_model("sms_lstm_model.h5")

def predict_text(text):
    # preprocessing: clean, tokenize, pad
    clean = clean_text(text)
    seq = tokenizer.texts_to_sequences([clean])
    seq_pad = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    prob = model.predict(seq_pad, verbose=0)[0][0]
    label = "spam" if prob >= 0.5 else "ham"
    return {"label": label, "probability": float(prob)}

st.title("SMS Spam Detection")
user_input = st.text_area("Enter your message:")
if st.button("Predict"):
    result = predict_text(user_input)
    st.write(result)

