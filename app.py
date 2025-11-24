# app.py
# Beautiful Streamlit UI for Six Human Emotions Detection
# Save this file and run: streamlit run app.py

import streamlit as st
import numpy as np
import nltk
import re
from nltk.stem import PorterStemmer
import pickle
import os
from typing import Tuple, Dict

# ---------------------- Styling (minimal CSS) ----------------------
st.set_page_config(page_title="Emotion Detector", layout="wide", page_icon="😊")

# Small CSS tweaks for a cleaner card-like look
st.markdown(
    """
    <style>
    .app-header {padding: 0.75rem; border-radius: 12px;}
    .big-title {font-size:32px; font-weight:700;}
    .muted {color: #6c757d}
    .prediction-card {background: linear-gradient(135deg, #ffffff 0%, #f4f8fb 100%); padding: 1rem; border-radius:12px; box-shadow: 0 6px 18px rgba(28,45,79,0.06);}    
    .example-btn {margin-right: 8px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------- NLTK resource helper ----------------------

def ensure_nltk_resource(name: str):
    """Download an NLTK resource only if missing (quiet)."""
    try:
        nltk.data.find(name)
    except LookupError:
        nltk.download(name.split('/')[-1], quiet=True)

ensure_nltk_resource('corpora/stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

# ---------------------- Model loading (cached) ----------------------
@st.cache_resource
def load_models():
    """Load pickled artifacts once per server life-cycle."""
    base = os.getcwd()
    # if your pickles live in a subfolder, change these paths
    lg = pickle.load(open('logistic_regresion.pkl','rb'))
    tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl','rb'))
    lb = pickle.load(open('label_encoder.pkl','rb'))
    return lg, tfidf_vectorizer, lb

try:
    lg, tfidf_vectorizer, lb = load_models()
except Exception as e:
    lg = tfidf_vectorizer = lb = None
    # don't crash the app — show helpful message later
    load_error = e
else:
    load_error = None

# ---------------------- Text processing & prediction ----------------------

def clean_text(text: str) -> str:
    stemmer = PorterStemmer()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stopwords]
    return " ".join(words)


def predict_emotion_and_probs(text: str) -> Tuple[str, float, Dict[str, float]]:
    """Return (predicted_emotion, top_probability, probs_by_label).
    probs_by_label is a dict mapping emotion->probability (0..1).
    """
    if tfidf_vectorizer is None or lg is None or lb is None:
        raise RuntimeError("Model artifacts are not loaded.")

    cleaned = clean_text(text)
    X = tfidf_vectorizer.transform([cleaned])

    # Get probabilities when available
    if hasattr(lg, 'predict_proba'):
        probs = lg.predict_proba(X)[0]
        labels = lg.classes_ if hasattr(lg, 'classes_') else None
        # labels may be encoded ints; map via lb if necessary
        if labels is not None and labels.dtype.kind in ('i', 'u', 'f'):
            # labels are numeric encoded; convert with lb
            labels = lb.inverse_transform(labels.astype(int))
    else:
        # fallback: use decision_function and softmax
        if hasattr(lg, 'decision_function'):
            scores = lg.decision_function(X)
            if scores.ndim == 1:
                # binary -> convert to two-class probs
                s = 1.0 / (1.0 + np.exp(-scores))
                probs = np.vstack([1 - s, s]).T[0]
                labels = lb.classes_
            else:
                exps = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                probs = (exps / exps.sum(axis=1, keepdims=True))[0]
                labels = lb.classes_
        else:
            # last-resort: treat predict as deterministic
            pred = lg.predict(X)[0]
            labels = lb.inverse_transform(lg.classes_) if hasattr(lg, 'classes_') else lb.classes_
            probs = np.array([1.0 if l == pred else 0.0 for l in lg.classes_])

    # build dict mapping textual labels to probabilities
    try:
        text_labels = [str(l) for l in labels]
    except Exception:
        # if labels are encoded, convert via label encoder
        text_labels = [str(l) for l in lb.inverse_transform(labels)]

    probs_by_label = {text_labels[i]: float(probs[i]) for i in range(len(text_labels))}

    # pick top
    top_label = max(probs_by_label, key=probs_by_label.get)
    top_prob = probs_by_label[top_label]

    return top_label, float(top_prob), probs_by_label

# ---------------------- UI layout ----------------------

EMOJI_MAP = {
    'Joy': '😊',
    'Fear': '😨',
    'Anger': '😡',
    'Love': '❤️',
    'Sadness': '😔',
    'Surprise': '😲'
}

with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<div class="app-header">', unsafe_allow_html=True)
        st.markdown('<div class="big-title">Six Human Emotions Detection</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.write('Detect the dominant emotion in a short piece of text. Powered by a TF-IDF + Logistic Regression pipeline.')
    with col2:
        # no image or emoji — keeping header clean and plain
        st.write("")("")        # no image or emoji — keeping header clean and plain\n        st.write("")

st.markdown('---')

# Left column: input & examples
left, right = st.columns((2, 3))

with left:
    st.subheader('Enter text')
    user_input = st.text_area('Write a sentence or paragraph here...', height=160)

    st.write('Examples')
    ex_col1, ex_col2 = st.columns(2)
    with ex_col1:
        if st.button('She got promoted and was ecstatic 🎉'):
            user_input = 'She got promoted and was ecstatic'
    with ex_col2:
        if st.button('He lost his wallet and panicked 😨'):
            user_input = 'He lost his wallet and panicked'

    st.write('Options')
    show_probs = st.checkbox('Show probability bar chart', value=True)
    show_raw = st.checkbox('Show cleaned text & raw vector', value=False)

    # optional upload
    uploaded = st.file_uploader('Or upload a .txt file', type=['txt'])
    if uploaded is not None:
        try:
            content = uploaded.read().decode('utf-8')
            user_input = content
            st.success('File loaded into input box')
        except Exception:
            st.error('Could not read file; ensure UTF-8 encoded .txt')

    if st.button('Predict', type='primary'):
        if not user_input or user_input.strip() == '':
            st.warning('Please enter some text or upload a file first.')
        else:
            if load_error:
                st.error('Model artifacts failed to load. See server logs. Exception:')
                st.exception(load_error)
            else:
                try:
                    pred, prob, probs_by_label = predict_emotion_and_probs(user_input)
                    # stash results in session state to show on right
                    st.session_state['last_result'] = (pred, prob, probs_by_label, user_input)
                except Exception as e:
                    st.error('Prediction failed — check server logs or model objects.')
                    st.exception(e)

with right:
    st.subheader('Prediction')
    result = st.session_state.get('last_result', None)
    if result is None:
        st.info('No prediction yet. Enter text and press Predict.')
    else:
        pred, prob, probs_by_label, original_text = result
        emoji = EMOJI_MAP.get(pred, '')
        st.markdown(f'<div class="prediction-card">', unsafe_allow_html=True)
        st.markdown(f'### {emoji}  **{pred}**')
        st.write(f'**Confidence:** {prob:.2%}')

        # show a simple bar chart of probabilities
        if show_probs:
            # bar_chart accepts a dict-of-lists or pandas frame, but it works with dict too
            # sort labels by prob
            sorted_items = sorted(probs_by_label.items(), key=lambda x: x[1], reverse=True)
            labels = [k for k, _ in sorted_items]
            values = [v for _, v in sorted_items]
            chart_data = {labels[i]: values[i] for i in range(len(labels))}
            st.bar_chart(chart_data, height=220)

        if show_raw:
            st.write('**Cleaned text**:')
            st.code(clean_text(original_text))

        st.markdown('</div>', unsafe_allow_html=True)

# Footer / About
st.markdown('---')
with st.expander('About this app'):
    st.write(
        """
        This demo shows a TF-IDF + Logistic Regression pipeline saved as pickles.

        Tips:
        - Make sure `logistic_regresion.pkl`, `tfidf_vectorizer.pkl`, and `label_encoder.pkl` are in the same folder as this script.
        - If you change scikit-learn versions and your pickles were created with a different version, you may need to re-create them.
        """
    )

# small note on running
st.caption('Save this file as app.py and run `streamlit run app.py` in the same folder where your pickles live.')

