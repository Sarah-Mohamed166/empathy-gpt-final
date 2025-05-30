import streamlit as st
import sys
import os
import torch
import numpy as np
import scipy.special
import pandas as pd
import matplotlib.pyplot as plt
import re
import speech_recognition as sr
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from pydub import AudioSegment
from st_audiorec import st_audiorec
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="EMPATHY-GPT", layout="wide")

st.markdown("""
    <style>
    html, body, div, span, h1, h2, h3, h4, h5, h6, p, button, input, label {
        font-size: 22px !important;
    }
    .stTextInput, .stButton>button {
        font-size: 22px !important;
    }
    .element-container svg text {
        font-size: 12px !important;
    }
    </style>
""", unsafe_allow_html=True)

ENGLISH_LABELS = ['anger', 'joy', 'optimism', 'sadness']
ARABIC_LABELS = ['negative', 'neutral', 'positive']

model_path_en = "cardiffnlp/twitter-roberta-base-emotion"
model_path_ar = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
model_path_gpt = "microsoft/DialoGPT-medium"

tokenizer_en = AutoTokenizer.from_pretrained(model_path_en)
model_en = AutoModelForSequenceClassification.from_pretrained(model_path_en)

tokenizer_ar = AutoTokenizer.from_pretrained(model_path_ar)
model_ar = AutoModelForSequenceClassification.from_pretrained(model_path_ar)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
empathetic_tokenizer = AutoTokenizer.from_pretrained(model_path_gpt, padding_side="left")
empathetic_model = AutoModelForCausalLM.from_pretrained(model_path_gpt).to(device)

st.sidebar.title("🌍 Settings")
language = st.sidebar.selectbox("Choose Language", ["English", "Arabic"])
calm_mode = st.sidebar.checkbox("🧘 Enable Calm Mode")
show_mood = st.sidebar.checkbox("📊 Show Mood Timeline")
use_gpt_response = st.sidebar.checkbox("🧠 Enable Advanced GPT-2 Responses")

st.title("💬 EMPATHY-GPT")
st.markdown("An Emotionally Aware, Multilingual Mental Health Chatbot 🧟‍♂️💖")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "emotion_log" not in st.session_state:
    st.session_state.emotion_log = []
if "typed_input" not in st.session_state:
    st.session_state.typed_input = ""
if "dialog_history" not in st.session_state:
    st.session_state.dialog_history = []

true_emotions = []
pred_emotions = []

def is_arabic(text): return bool(re.search(r'[\u0600-\u06FF]', text))
def is_english(text): return bool(re.search(r'[a-zA-Z]', text))

def override_arabic_emotion(text, predicted):
    overrides = {
        "positive": ["سعيدة", "سعيد", "فرح", "مبسوطة", "أشعر بالسعادة"],
        "negative": ["حزين", "زعلان", "ضايق", "كئيب", "أبكي", "أشعر بالحزن"],
        "neutral": ["عادي", "لا بأس", "محايد", "لا أشعر بشيء"]
    }
    for emotion, keywords in overrides.items():
        if any(kw in text for kw in keywords):
            return emotion
    return predicted

def generate_gpt_reply(history, user_message, emotion):
    dialog_prefix = "\n".join(history[-4:])  # lightweight memory (last 4 lines)
    input_text = f"{dialog_prefix}\n{user_message}{empathetic_tokenizer.eos_token}"
    input_ids = empathetic_tokenizer.encode(input_text, return_tensors="pt").to(device)
    reply_ids = empathetic_model.generate(
        input_ids,
        max_length=150,
        pad_token_id=empathetic_tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    reply = empathetic_tokenizer.decode(reply_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return reply.strip()

st.markdown("**🎤 Voice Input:**")
audio = st_audiorec()

if audio:
    with open("input.wav", "wb") as f:
        f.write(audio)
    try:
        sound = AudioSegment.from_file("input.wav")
        sound.export("converted.wav", format="wav")
        recognizer = sr.Recognizer()
        with sr.AudioFile("converted.wav") as source:
            audio_data = recognizer.record(source)
            transcript = recognizer.recognize_google(audio_data, language='ar' if language == 'Arabic' else 'en')
            st.session_state["typed_input"] = transcript
            st.success(f"🎤 Transcribed: {transcript}")
    except Exception as e:
        st.error(f"Audio transcription failed: {e}")

user_input = st.text_input("Type your message here:", value=st.session_state.get("typed_input", ""))
send = st.button("Send")

if send and user_input.strip():
    detected_arabic = is_arabic(user_input)
    detected_english = is_english(user_input)

    if language == "English" and detected_arabic:
        st.session_state.chat_history.append(("EMPATHY-GPT", "You typed in Arabic. Please switch language to Arabic in the sidebar 🌐."))
    elif language == "Arabic" and detected_english:
        st.session_state.chat_history.append(("EMPATHY-GPT", "لقد كتبت باللغة الإنجليزية. من فضلك غيّر اللغة إلى الإنجليزية من الشريط الجانبي 🌐."))
    else:
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.dialog_history.append(f"You: {user_input}")

        if language == "English":
            inputs = tokenizer_en(user_input, return_tensors="pt", truncation=True)
            with torch.no_grad():
                logits = model_en(**inputs).logits
            probs = scipy.special.softmax(logits.cpu().numpy()[0])
            predicted_emotion = ENGLISH_LABELS[np.argmax(probs)]
            st.session_state.emotion_log.append((predicted_emotion, datetime.now().strftime("%H:%M:%S")))
            true_emotions.append("joy")
            pred_emotions.append(predicted_emotion)

            if use_gpt_response:
                bot_reply = generate_gpt_reply(st.session_state.dialog_history, user_input, predicted_emotion)
                bleu = sentence_bleu([user_input.split()], bot_reply.split(), smoothing_function=SmoothingFunction().method1)
                st.write(f"BLEU Score: {bleu:.2f}")
            else:
                responses = {
                    "joy": "That's wonderful! 😊",
                    "sadness": "I'm really sorry you're feeling this way. 💔",
                    "anger": "That sounds really frustrating. 😠 I'm here for you.",
                    "optimism": "I'm glad you're staying hopeful! 🌟"
                }
                reply = responses.get(predicted_emotion, "I'm here for you. ❤️")
                bot_reply = f"I sense you're feeling **{predicted_emotion}**. {reply}"
                if calm_mode and predicted_emotion in ["sadness", "anger"]:
                    bot_reply += "\n\n🧘 Let's take a moment. Breathe in slowly... and out. You're not alone."

        elif language == "Arabic":
            inputs = tokenizer_ar(user_input, return_tensors="pt", truncation=True)
            with torch.no_grad():
                logits = model_ar(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs.cpu(), dim=1).item()
            predicted_emotion = ARABIC_LABELS[pred_idx]
            predicted_emotion = override_arabic_emotion(user_input, predicted_emotion)
            st.session_state.emotion_log.append((predicted_emotion, datetime.now().strftime("%H:%M:%S")))

            emotion_words = {
                "positive": "إيجابية",
                "neutral": "محايدة",
                "negative": "سلبية"
            }
            responses = {
                "positive": "رائع! يسعدني أنك تشعر بهذه الإيجابية. 😊",
                "neutral": "أشعر أنك تمر بحالة محايدة. أنا هنا إذا احتجت التحدث في أي وقت. 🧑‍💬",
                "negative": "أشعر أنك تمر بحالة سلبية. أشعر بك، وأنا بجانبك دائمًا. خذ نفسًا عميقًا، كل شيء سيتحسن إن شاء الله. 💔"
            }
            word = emotion_words.get(predicted_emotion, predicted_emotion)
            reply = responses.get(predicted_emotion, "أنا هنا من أجلك. 💬")
            bot_reply = f"أشعر أنك تمر بحالة {word}. {reply}"
            if calm_mode and predicted_emotion == "negative":
                calm = "دعنا نهدأ قليلًا. خذ نفسًا عميقًا... وازفر ببطء. لست وحدك، أنا هنا معك. 🧘"
                bot_reply += f"\n\n<div style='direction: rtl; text-align: right; color: #4B0082;'>{calm}</div>"

        st.session_state.chat_history.append(("EMPATHY-GPT", bot_reply))
        st.session_state.dialog_history.append(f"EMPATHY-GPT: {bot_reply}")

for sender, msg in st.session_state.chat_history:
    st.markdown(f"**{sender}:** {msg}", unsafe_allow_html=True)

if show_mood and st.session_state.emotion_log:
    st.subheader("🧠 Mood Timeline")
    df = pd.DataFrame(st.session_state.emotion_log, columns=["Emotion", "Time"])
    all_labels = ENGLISH_LABELS + [lbl for lbl in ARABIC_LABELS if lbl not in ENGLISH_LABELS]
    counts = df["Emotion"].value_counts().reindex(all_labels, fill_value=0)

    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax, color="skyblue", edgecolor="black")
    ax.set_ylabel("Count")
    ax.set_title("Emotion Frequency")
    ax.set_xticklabels(counts.index, rotation=0)
    for container in ax.containers:
        ax.bar_label(container, label_type='edge', fontsize=10)
    st.pyplot(fig)

if true_emotions and pred_emotions:
    st.subheader("📊 Evaluation Metrics")
    score = f1_score(true_emotions, pred_emotions, average='macro')
    st.write(f"**F1 Score**: {score:.2f}")
