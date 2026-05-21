import os
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

import streamlit as st
import numpy as np
import cv2
import librosa
import torch
import onnxruntime as ort
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from scipy.special import softmax
import tempfile
import time

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="EmoForge",
    page_icon="🎭",
    layout="wide"
)

# ─────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600;700&family=Montserrat:wght@300;400;500;600&display=swap');

:root {
    --navy:      #020814;
    --navy-mid:  #061020;
    --navy-card: #0a1628;
    --silver:    #c0c8d8;
    --silver-lt: #e8edf5;
    --red:       #c0292b;
    --red-glow:  #e03335;
    --gold-hint: #8a7a5a;
}

/* Base */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--navy) !important;
    color: var(--silver-lt) !important;
    font-family: 'Montserrat', sans-serif !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display: none !important; }

/* Hide all top-right toolbar buttons including copy */
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
button[kind="header"] { display: none !important; }
.stActionButton { display: none !important; }
header[data-testid="stHeader"] { display: none !important; }

/* Tabs */
/* Full width tabs */
[data-testid="stTabs"] > div:first-child {
    width: 100% !important;
}
[data-testid="stTabs"] button {
    flex: 1 !important;
    text-align: center !important;
}

/* Reduce page padding */
section[data-testid="stMainBlockContainer"] {
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
}
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: var(--silver) !important;
    border-bottom: 2px solid transparent !important;
    padding: 10px 20px !important;
    background: transparent !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--red-glow) !important;
    border-bottom: 2px solid var(--red-glow) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--red) 0%, #8b1a1b 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    font-size: 12px !important;
    padding: 10px 0 !important;
    flex: 1 !important;
    text-align: center !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    box-shadow: 0 0 20px rgba(192, 41, 43, 0.5) !important;
    transform: translateY(-1px) !important;
}

/* Inputs */
.stTextArea textarea, .stFileUploader {
    background: var(--navy-card) !important;
    border: 1px solid rgba(192, 200, 216, 0.2) !important;
    border-radius: 2px !important;
    color: var(--silver-lt) !important;
    font-family: 'Montserrat', sans-serif !important;
}

/* Divider */
hr { border-color: rgba(192, 200, 216, 0.15) !important; }

/* Metric */
[data-testid="stMetric"] {
    background: var(--navy-card) !important;
    border: 1px solid rgba(192, 200, 216, 0.15) !important;
    border-radius: 4px !important;
    padding: 12px !important;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--red) 0%, var(--red-glow) 100%) !important;
}

/* Spinner */
.stSpinner { color: var(--red-glow) !important; }

/* Subheader */
h2, h3 {
    font-family: 'Cormorant Garamond', serif !important;
    color: var(--silver-lt) !important;
    letter-spacing: 1px !important;
}

/* Info/Warning boxes */
[data-testid="stAlert"] {
    background: var(--navy-card) !important;
    border-left: 3px solid var(--red) !important;
    border-radius: 2px !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# SPLASH SCREEN
# ─────────────────────────────────────────
if 'splash_done' not in st.session_state:
    st.session_state['splash_done'] = False

if not st.session_state['splash_done']:
    splash = st.empty()
    splash.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600;700&family=Montserrat:wght@300;400;500;600&display=swap');
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(30px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes lineExpand {
        from { width: 0; }
        to   { width: 120px; }
    }
    @keyframes subtitleFade {
        0%   { opacity: 0; }
        50%  { opacity: 0; }
        100% { opacity: 1; }
    }
    .splash-wrapper {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 85vh;
        text-align: center;
        background: #020814;
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        z-index: 9999;
    }
    .splash-title {
        font-family: 'Montserrat', sans-serif;
        font-size: 72px;
        font-weight: 300;
        letter-spacing: 24px;
        color: #e8edf5;
        animation: fadeIn 1.2s ease forwards;
        margin: 0;
        line-height: 1;
    }
    .splash-title span { color: #c0292b; }
    .splash-line {
        height: 1px;
        background: linear-gradient(90deg, transparent, #c0292b, transparent);
        animation: lineExpand 1s ease 0.8s forwards;
        width: 0;
        margin: 20px auto;
    }
    .splash-subtitle {
        font-family: 'Montserrat', sans-serif;
        font-size: 10px;
        letter-spacing: 5px;
        text-transform: uppercase;
        color: #c0c8d8;
        animation: subtitleFade 2s ease forwards;
        margin-top: 10px;
    }
    .splash-quote {
        font-family: 'Cormorant Garamond', serif;
        font-size: 20px;
        font-weight: 300;
        color: rgba(192,200,216,0.5);
        animation: subtitleFade 2.5s ease forwards;
        margin-top: 28px;
        font-style: italic;
        max-width: 420px;
        line-height: 1.6;
    }
    .splash-author {
        font-family: 'Montserrat', sans-serif;
        font-size: 9px;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #c0292b;
        animation: subtitleFade 2.8s ease forwards;
        margin-top: 12px;
    }
    </style>
    <div class="splash-wrapper">
        <h1 class="splash-title">EMO<span>FORGE</span></h1>
        <div class="splash-line"></div>
        <p class="splash-subtitle">Multimodal Emotion Recognition</p>
        <p class="splash-quote">"To understand is to perceive patterns"</p>
        <p class="splash-author">— Isaiah Berlin</p>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(3)
    st.session_state['splash_done'] = True
    splash.empty()
    st.rerun()
# ─────────────────────────────────────────
# PATHS & EMOTIONS
# ─────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTION_EMOJI = {
    'angry':    '😠', 'disgust': '🤢', 'fear':    '😨',
    'happy':    '😊', 'neutral': '😐', 'sad':     '😢',
    'surprise': '😲'
}
EMOTION_COLOR = {
    'angry':    '#c0292b', 'disgust': '#6b3a2a', 'fear':    '#4a2060',
    'happy':    '#8a7a2a', 'neutral': '#3a4a5a', 'sad':     '#1a3a5a',
    'surprise': '#8a1a3a'
}

# ─────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────
@st.cache_resource
def load_face_model():
    return ort.InferenceSession(
        os.path.join(MODEL_DIR, 'face_model.onnx'),
        providers=['CPUExecutionProvider'])

@st.cache_resource
def load_speech_model():
    return ort.InferenceSession(
        os.path.join(MODEL_DIR, 'speech_model.onnx'),
        providers=['CPUExecutionProvider'])

@st.cache_resource
def load_text_model():
    device = torch.device('cpu')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=7)
    model.load_state_dict(
        torch.load(
            os.path.join(MODEL_DIR, 'text_model_best.pt'),
            map_location=device))
    model.eval()
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    return model, tokenizer

with st.spinner("Initialising EmoForge..."):
    face_session          = load_face_model()
    speech_session        = load_speech_model()
    text_model, tokenizer = load_text_model()

# ─────────────────────────────────────────
# INFERENCE FUNCTIONS
# ─────────────────────────────────────────
def predict_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None, None, image
    x, y, w, h = faces[0]
    face_roi = gray[y:y+h, x:x+w]
    face_roi = cv2.resize(face_roi, (48, 48))
    face_roi = face_roi / 255.0
    face_roi = face_roi.reshape(1, 48, 48, 1).astype(np.float32)
    input_name = face_session.get_inputs()[0].name
    probs       = face_session.run(None, {input_name: face_roi})[0][0]
    emotion_idx = np.argmax(probs)
    emotion     = EMOTIONS[emotion_idx]
    cv2.rectangle(image, (x, y), (x+w, y+h), (192, 41, 43), 2)
    cv2.putText(image, f"{emotion} {probs[emotion_idx]:.0%}",
                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (192,41,43), 2)
    return emotion, probs, image

def predict_speech(audio_path):
    try:
        audio, sr = librosa.load(audio_path, sr=22050)
        mfcc   = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mel    = librosa.power_to_db(
            librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128), ref=np.max)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        features = np.vstack([mfcc, mel, chroma])
        if features.shape[1] < 128:
            features = np.pad(features, ((0,0),(0,128-features.shape[1])), mode='constant')
        else:
            features = features[:, :128]
        features    = features.transpose(1, 0).reshape(1, 128, 180).astype(np.float32)
        input_name  = speech_session.get_inputs()[0].name
        probs       = speech_session.run(None, {input_name: features})[0][0]
        emotion_idx = np.argmax(probs)
        return EMOTIONS[emotion_idx], probs
    except Exception as e:
        st.error(f"Speech error: {e}")
        return None, None

def predict_text(text):
    inputs = tokenizer(
        text, return_tensors='pt',
        truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = text_model(**inputs)
    probs       = softmax(outputs.logits.numpy()[0])
    emotion_idx = np.argmax(probs)
    return EMOTIONS[emotion_idx], probs

def show_result(emotion, confidence, title, highlight=False):
    if highlight:
        # Odd one out — deep red
        bg = "linear-gradient(135deg, #1e1a08 0%, #0e0d04 100%)"
        border_left = "#6b5e2a"
        border = "rgba(107,94,42,0.4)"
        emotion_color = "#e8edf5"
        conf_color = "rgba(192,200,216,0.5)"
    else:
        # Matching — deep muted olive gold
        bg = "linear-gradient(135deg, #1e1a08 0%, #0e0d04 100%)"
        border_left = "#6b5e2a"
        border = "rgba(107,94,42,0.4)"
        emotion_color = "#e8edf5"
        conf_color = "rgba(192,200,216,0.5)"
    st.markdown(f"""
    <div style="
        background: {bg};
        border: 1px solid {border};
        border-left: 3px solid {border_left};
        border-radius: 4px;
        padding: 18px;
        text-align: center;
        margin: 0 0 6px 0;
    ">
        <p style="color:rgba(192,200,216,0.5);font-family:'Montserrat',sans-serif;
            font-size:9px;letter-spacing:4px;text-transform:uppercase;margin:0 0 10px 0;">
            {title}</p>
        <div style="width:30px;height:1px;background:#c0292b;margin:0 auto 12px auto;"></div>
        <h2 style="color:{emotion_color};font-family:'Cormorant Garamond',serif;
            font-size:32px;font-weight:600;letter-spacing:4px;
            text-transform:uppercase;margin:0 0 4px 0;">{emotion}</h2>
        <p style="color:{conf_color};font-family:'Montserrat',sans-serif;
            font-size:11px;letter-spacing:2px;margin:4px 0;">
            {confidence:.1%} confidence</p>
    </div>""", unsafe_allow_html=True)
# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:32px 0 8px 0;border-bottom:1px solid rgba(192,200,216,0.1);margin-bottom:24px;">
    <h1 style="font-family:'Cormorant Garamond',serif;font-size:52px;font-weight:300;
        letter-spacing:14px;color:#e8edf5;margin:0;">
        EMO<span style="color:#c0292b;">FORGE</span>
    </h1>
    <p style="font-family:'Montserrat',sans-serif;font-size:10px;letter-spacing:5px;
        text-transform:uppercase;color:#c0c8d8;margin:8px 0 0 0;">
        Multimodal Deep Learning · Emotion Recognition
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "FACE", "SPEECH", "TEXT", "FUSION"])

# ─────────────────────────────────────────
# TAB 1 — FACE
# ─────────────────────────────────────────
with tab1:
    st.markdown("""
    <h2 style="font-family:'Cormorant Garamond',serif;font-size:36px;font-weight:400;
        letter-spacing:3px;color:#e8edf5;margin:0 0 4px 0;">Face Emotion Recognition</h2>
    <p style="font-family:'Montserrat',sans-serif;font-size:10px;letter-spacing:3px;
        text-transform:uppercase;color:rgba(192,200,216,0.45);margin:0 0 20px 0;">
        Look directly at the camera and capture your expression</p>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        img_file = st.camera_input("📷", label_visibility="collapsed")
    with col2:
        if img_file:
            if 'last_face_image' not in st.session_state or st.session_state.get('last_face_image') != img_file.file_id:
                img_file.seek(0)
                file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                emotion, probs, annotated = predict_face(image)
                if emotion:
                    st.session_state['face_annotated']  = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    st.session_state['face_probs']      = probs
                    st.session_state['face_emotion']    = emotion
                    st.session_state['last_face_image'] = img_file.file_id
            if 'face_emotion' in st.session_state:
                st.image(st.session_state['face_annotated'], use_container_width=True)
                show_result(st.session_state['face_emotion'], max(st.session_state['face_probs']), "Face Emotion")
            else:
                st.warning("No face detected. Please try again.")
        else:
            st.markdown("""
            <div style="border:1px dashed rgba(192,200,216,0.15);border-radius:4px;
                padding:40px;text-align:center;color:rgba(192,200,216,0.3);">
                <p style="font-family:'Montserrat',sans-serif;font-size:10px;
                    letter-spacing:3px;text-transform:uppercase;margin:0;">
                    Result appears here</p>
            </div>""", unsafe_allow_html=True)
  

# ─────────────────────────────────────────
# TAB 2 — SPEECH
# ─────────────────────────────────────────
with tab2:
    st.markdown("""
    <h2 style="font-family:'Cormorant Garamond',serif;font-size:36px;font-weight:400;
        letter-spacing:3px;color:#e8edf5;margin:0 0 4px 0;">Speech Emotion Recognition</h2>
    <p style="font-family:'Montserrat',sans-serif;font-size:10px;letter-spacing:3px;
        text-transform:uppercase;color:rgba(192,200,216,0.45);margin:0 0 20px 0;">
        Upload a recording — let the model listen</p>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        audio_file = st.file_uploader("Upload WAV or MP3", type=['wav', 'mp3'])
    with col2:
        if audio_file:
            st.audio(audio_file)
            audio_file.seek(0)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name
            emotion, probs = predict_speech(tmp_path)
            os.unlink(tmp_path)
            if emotion:
                show_result(emotion, max(probs), "Speech Emotion")
                st.session_state['speech_probs']   = probs
                st.session_state['speech_emotion'] = emotion
        elif 'speech_emotion' in st.session_state:
            show_result(
                st.session_state['speech_emotion'],
                max(st.session_state['speech_probs']), "Speech Emotion")
        else:
            st.markdown("""
            <div style="border:1px dashed rgba(192,200,216,0.2);border-radius:4px;
                padding:40px;text-align:center;color:rgba(192,200,216,0.4);">
                <p style="font-family:'Montserrat',sans-serif;font-size:11px;
                    letter-spacing:3px;text-transform:uppercase;">
                    Upload an audio file to begin</p>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# TAB 3 — TEXT
# ─────────────────────────────────────────
with tab3:
    st.markdown("""
    <h2 style="font-family:'Cormorant Garamond',serif;font-size:36px;font-weight:400;
        letter-spacing:3px;color:#e8edf5;margin:0 0 4px 0;">Text Emotion Recognition</h2>
    <p style="font-family:'Montserrat',sans-serif;font-size:10px;letter-spacing:3px;
        text-transform:uppercase;color:rgba(192,200,216,0.45);margin:0 0 20px 0;">
        Type anything — a thought, feeling, or sentence</p>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        text_input = st.text_area(
            "Enter text",
            placeholder="Type something expressive...",
            height=150,
            label_visibility="collapsed")
        if st.button("ANALYSE"):
            if text_input.strip():
                with st.spinner("Analysing..."):
                    emotion, probs = predict_text(text_input)
                st.session_state['text_probs']   = probs
                st.session_state['text_emotion'] = emotion
            else:
                st.warning("Please enter some text first.")
    with col2:
        if 'text_emotion' in st.session_state:
            show_result(
                st.session_state['text_emotion'],
                max(st.session_state['text_probs']), "Text Emotion")
            st.markdown("<br>", unsafe_allow_html=True)
            bars_html = ""
            for i, em in enumerate(EMOTIONS):
                prob = float(st.session_state['text_probs'][i])
                is_top = em == st.session_state['text_emotion']
                bar_color = "#c0292b" if is_top else "rgba(192,200,216,0.2)"
                label_color = "#e8edf5" if is_top else "rgba(192,200,216,0.5)"
                bars_html += f"""
                <div style="margin-bottom:10px;">
                    <div style="display:flex;justify-content:space-between;
                        font-family:'Montserrat',sans-serif;font-size:9px;
                        letter-spacing:2px;text-transform:uppercase;
                        color:{label_color};margin-bottom:4px;">
                        <span>{em}</span><span>{prob:.1%}</span>
                    </div>
                    <div style="background:rgba(192,200,216,0.08);border-radius:2px;height:3px;">
                        <div style="width:{prob*100:.1f}%;height:3px;
                            background:{bar_color};border-radius:2px;"></div>
                    </div>
                </div>"""
            st.markdown(bars_html, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="border:1px dashed rgba(192,200,216,0.15);border-radius:4px;
                padding:40px;text-align:center;color:rgba(192,200,216,0.3);">
                <p style="font-family:'Montserrat',sans-serif;font-size:10px;
                    letter-spacing:3px;text-transform:uppercase;margin:0;">
                    Result appears here</p>
            </div>""", unsafe_allow_html=True)
# ─────────────────────────────────────────
# TAB 4 — FUSION
# ─────────────────────────────────────────
with tab4:
    st.markdown("""
    <h2 style="font-family:'Cormorant Garamond',serif;font-size:36px;font-weight:400;
        letter-spacing:3px;color:#e8edf5;margin:0 0 4px 0;">Multimodal Fusion</h2>
    <p style="font-family:'Montserrat',sans-serif;font-size:10px;letter-spacing:3px;
        text-transform:uppercase;color:rgba(192,200,216,0.45);margin:0 0 20px 0;">
        Complete at least 2 tabs — results fuse automatically</p>
    """, unsafe_allow_html=True)

    face_done   = 'face_probs'   in st.session_state
    speech_done = 'speech_probs' in st.session_state
    text_done   = 'text_probs'   in st.session_state
    done_count  = sum([face_done, speech_done, text_done])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div style="border:1px solid rgba(192,200,216,0.1);border-radius:4px;
            padding:16px;text-align:center;">
            <p style="font-family:'Montserrat',sans-serif;font-size:9px;letter-spacing:3px;
                text-transform:uppercase;color:rgba(192,200,216,0.5);margin:0 0 6px 0;">Face</p>
            <p style="font-family:'Montserrat',sans-serif;font-size:12px;letter-spacing:2px;
                color:{'#c0292b' if face_done else 'rgba(192,200,216,0.3)'};margin:0;">
                {'READY' if face_done else 'PENDING'}</p>
            </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div style="border:1px solid rgba(192,200,216,0.1);border-radius:4px;
            padding:16px;text-align:center;">
            <p style="font-family:'Montserrat',sans-serif;font-size:9px;letter-spacing:3px;
                text-transform:uppercase;color:rgba(192,200,216,0.5);margin:0 0 6px 0;">Speech</p>
            <p style="font-family:'Montserrat',sans-serif;font-size:12px;letter-spacing:2px;
                color:{'#c0292b' if speech_done else 'rgba(192,200,216,0.3)'};margin:0;">
                {'READY' if speech_done else 'PENDING'}</p>
            </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div style="border:1px solid rgba(192,200,216,0.1);border-radius:4px;
            padding:16px;text-align:center;">
            <p style="font-family:'Montserrat',sans-serif;font-size:9px;letter-spacing:3px;
                text-transform:uppercase;color:rgba(192,200,216,0.5);margin:0 0 6px 0;">Text</p>
            <p style="font-family:'Montserrat',sans-serif;font-size:12px;letter-spacing:2px;
                color:{'#c0292b' if text_done else 'rgba(192,200,216,0.3)'};margin:0;">
                {'READY' if text_done else 'PENDING'}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if done_count < 2:
        missing = []
        if not face_done:   missing.append("Face")
        if not speech_done: missing.append("Speech")
        if not text_done:   missing.append("Text")
        st.markdown(f"""<p style="font-family:'Montserrat',sans-serif;font-size:10px;
            letter-spacing:3px;text-transform:uppercase;color:rgba(192,200,216,0.4);">
            Complete {' and '.join(missing)} to run fusion</p>""", unsafe_allow_html=True)
    else:
        if face_done and speech_done and text_done:
            combo_label = "Face · Speech · Text"
            weights = {"face": 0.33, "speech": 0.40, "text": 0.27}
        elif face_done and speech_done:
            combo_label = "Face · Speech"
            weights = {"face": 0.45, "speech": 0.55}
        elif face_done and text_done:
            combo_label = "Face · Text"
            weights = {"face": 0.55, "text": 0.45}
        elif speech_done and text_done:
            combo_label = "Speech · Text"
            weights = {"speech": 0.60, "text": 0.40}

        st.markdown(f"""<p style="font-family:'Montserrat',sans-serif;font-size:9px;
            letter-spacing:3px;text-transform:uppercase;color:rgba(192,200,216,0.4);">
            Auto-detected &nbsp;·&nbsp; {combo_label}</p>""", unsafe_allow_html=True)

        if st.button("RUN FUSION", type="primary"):
            with st.spinner("Fusing modalities..."):
                fused = np.zeros(7)
                for m, w in weights.items():
                    fused += w * np.array(st.session_state[f'{m}_probs'])
                emotion_idx   = np.argmax(fused)
                final_emotion = EMOTIONS[emotion_idx]
                confidence    = fused[emotion_idx]

            st.markdown("<br>", unsafe_allow_html=True)

            # Big final result
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #3a0a0a 0%, #1a0505 100%);
                border: 1px solid rgba(192,41,43,0.6);
                border-left: 4px solid #c0292b;
                border-radius: 4px;
                padding: 36px;
                text-align: center;
                margin: 0 0 20px 0;">
                <p style="font-family:'Montserrat',sans-serif;font-size:9px;
                    letter-spacing:4px;text-transform:uppercase;
                    color:rgba(192,200,216,0.5);margin:0 0 10px 0;">Final Fused Emotion</p>
                <div style="width:30px;height:1px;background:#c0292b;margin:0 auto 16px auto;"></div>
                <h1 style="font-family:'Cormorant Garamond',serif;font-size:72px;
                    font-weight:600;letter-spacing:8px;text-transform:uppercase;
                    color:#e8edf5;margin:0 0 8px 0;">{final_emotion}</h1>
                <p style="font-family:'Montserrat',sans-serif;font-size:11px;
                    letter-spacing:2px;color:#c0292b;margin:0;">
                    {confidence:.1%} confidence</p>
            </div>""", unsafe_allow_html=True)

            # Individual results
            st.markdown("""<p style="font-family:'Montserrat',sans-serif;font-size:9px;
                letter-spacing:3px;text-transform:uppercase;
                color:rgba(192,200,216,0.4);margin:0 0 12px 0;">
                Individual Modalities</p>""", unsafe_allow_html=True)

            icons = {"face": "Face", "speech": "Speech", "text": "Text"}
            mod_cols = st.columns(len(weights))
            for idx, (m, w) in enumerate(weights.items()):
                with mod_cols[idx]:
                    is_odd = st.session_state[f'{m}_emotion'] != final_emotion
                    show_result(
                        st.session_state[f'{m}_emotion'],
                        max(st.session_state[f'{m}_probs']),
                        icons[m],
                        highlight=is_odd)

            st.markdown("<br>", unsafe_allow_html=True)

            # Probability breakdown
            st.markdown("""<p style="font-family:'Montserrat',sans-serif;font-size:9px;
                letter-spacing:3px;text-transform:uppercase;
                color:rgba(192,200,216,0.4);margin:0 0 12px 0;">
                Probability Breakdown</p>""", unsafe_allow_html=True)
            bars_html = ""
            for i, em in enumerate(EMOTIONS):
                prob = float(fused[i])
                is_top = em == final_emotion
                bar_color = "#c0292b" if is_top else "rgba(192,200,216,0.2)"
                label_color = "#e8edf5" if is_top else "rgba(192,200,216,0.5)"
                bars_html += f"""
                <div style="margin-bottom:10px;">
                    <div style="display:flex;justify-content:space-between;
                        font-family:'Montserrat',sans-serif;font-size:9px;
                        letter-spacing:2px;text-transform:uppercase;
                        color:{label_color};margin-bottom:4px;">
                        <span>{em}</span><span>{prob:.1%}</span>
                    </div>
                    <div style="background:rgba(192,200,216,0.08);border-radius:2px;height:3px;">
                        <div style="width:{prob*100:.1f}%;height:3px;
                            background:{bar_color};border-radius:2px;"></div>
                    </div>
                </div>"""
            st.markdown(bars_html, unsafe_allow_html=True)