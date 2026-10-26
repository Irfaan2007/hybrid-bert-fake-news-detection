from pathlib import Path
from collections import Counter
import re

import joblib
import numpy as np
import pandas as pd
import shap
import spacy
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel


# ============================================================
# PATHS AND PAGE CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_PATH = DATA_DIR / "logistic_regression_model.pkl"
BACKGROUND_PATH = DATA_DIR / "shap_background.npy"

BERT_MODEL_NAME = "bert-base-uncased"
EXPECTED_BERT_FEATURES = 768
EXPECTED_LINGUISTIC_FEATURES = 19
EXPECTED_TOTAL_FEATURES = (
    EXPECTED_BERT_FEATURES + EXPECTED_LINGUISTIC_FEATURES
)

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
)

st.title("📰 Fake News Detection System")
st.markdown(
    """
    ### Hybrid BERT-Based Fake News Detection Using
    Linguistic Features and Explainable AI

    Enter a news title and article content to classify the text as
    **Real** or **Fake**. This is a text classifier, not an independent
    fact-checking service.
    """
)


# ============================================================
# LOAD CLASSIFIER, SPACY AND BERT
# ============================================================

@st.cache_resource
def load_classifier():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    if getattr(model, "n_features_in_", None) != EXPECTED_TOTAL_FEATURES:
        raise ValueError(
            "The saved model does not expect 787 features. "
            f"Expected: {EXPECTED_TOTAL_FEATURES}; "
            f"model expects: {getattr(model, 'n_features_in_', 'unknown')}."
        )

    if list(model.classes_) != [0, 1]:
        raise ValueError(
            "Expected model classes [0, 1], where 0 = Real and 1 = Fake. "
            f"Found: {list(model.classes_)}"
        )

    return model


@st.cache_resource
def load_spacy():
    # This matches the training notebook: spaCy English model with NER disabled.
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    return nlp


@st.cache_resource
def load_bert():
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = AutoModel.from_pretrained(BERT_MODEL_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, device


try:
    classifier = load_classifier()
    nlp = load_spacy()
except Exception as exc:
    st.error("Could not load the classifier or spaCy model.")
    st.exception(exc)
    st.stop()


# ============================================================
# LINGUISTIC FEATURE EXTRACTION
# Keep this logic aligned with the training notebook.
# Feature order must remain identical to training.
# ============================================================

LINGUISTIC_FEATURE_NAMES = [
    "word_count",
    "sentence_count",
    "character_count",
    "average_word_length",
    "average_sentence_length",
    "noun_count",
    "verb_count",
    "adjective_count",
    "adverb_count",
    "pronoun_count",
    "proper_noun_count",
    "exclamation_count",
    "question_count",
    "uppercase_count",
    "url_count",
    "hashtag_count",
    "mention_count",
    "special_character_count",
    "repeated_word_count",
]


def extract_linguistic_features(doc):
    """
    Extract the 19 linguistic/anomaly features using the definitions
    in the training notebook. Return them in the training feature order.
    """
    text = doc.text

    # POS features
    noun_count = sum(1 for token in doc if token.pos_ == "NOUN")
    verb_count = sum(1 for token in doc if token.pos_ == "VERB")
    adjective_count = sum(1 for token in doc if token.pos_ == "ADJ")
    adverb_count = sum(1 for token in doc if token.pos_ == "ADV")
    pronoun_count = sum(1 for token in doc if token.pos_ == "PRON")
    proper_noun_count = sum(1 for token in doc if token.pos_ == "PROPN")

    # Basic text features. Match the training notebook's token filtering.
    words = [
        token.text
        for token in doc
        if not token.is_punct and not token.is_space
    ]

    word_count = len(words)
    sentence_count = len(list(doc.sents))
    character_count = len(text)

    average_word_length = (
        sum(len(word) for word in words) / word_count
        if word_count > 0
        else 0
    )

    average_sentence_length = (
        word_count / sentence_count
        if sentence_count > 0
        else 0
    )

    # Anomaly features: use the same definitions as training.
    exclamation_count = text.count("!")
    question_count = text.count("?")

    uppercase_count = sum(
        1 for word in words if word.isupper() and len(word) > 1
    )

    url_count = len(
        re.findall(r"https?://\S+|www\.\S+", text)
    )

    hashtag_count = len(
        re.findall(r"#\w+", text)
    )

    mention_count = len(
        re.findall(r"@\w+", text)
    )

    special_character_count = len(
        re.findall(r"[^a-zA-Z0-9\s]", text)
    )

    word_list = [word.lower() for word in words]
    word_counter = Counter(word_list)

    # Number of distinct word types that occur more than twice.
    repeated_word_count = sum(
        1 for count in word_counter.values() if count > 2
    )

    features = [
        word_count,
        sentence_count,
        character_count,
        average_word_length,
        average_sentence_length,
        noun_count,
        verb_count,
        adjective_count,
        adverb_count,
        pronoun_count,
        proper_noun_count,
        exclamation_count,
        question_count,
        uppercase_count,
        url_count,
        hashtag_count,
        mention_count,
        special_character_count,
        repeated_word_count,
    ]

    result = np.asarray(features, dtype=np.float64)

    if result.shape != (EXPECTED_LINGUISTIC_FEATURES,):
        raise ValueError(
            f"Expected 19 linguistic features, got shape {result.shape}."
        )

    if not np.isfinite(result).all():
        raise ValueError("Linguistic features contain non-finite values.")

    return result


# ============================================================
# BERT FEATURE EXTRACTION
# Match training: bert-base-uncased, CLS embedding, max_length=256.
# ============================================================

def extract_bert_features(text):
    tokenizer, bert_model, device = load_bert()

    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = bert_model(**inputs)

    cls_embedding = outputs.last_hidden_state[:, 0, :]
    features = cls_embedding.detach().cpu().numpy()

    if features.shape != (1, EXPECTED_BERT_FEATURES):
        raise ValueError(
            f"Expected BERT shape (1, 768), got {features.shape}."
        )

    if not np.isfinite(features).all():
        raise ValueError("BERT features contain non-finite values.")

    return features


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Model Information")
st.sidebar.write("**Classifier:** Logistic Regression")
st.sidebar.write("**BERT model:** bert-base-uncased")
st.sidebar.write("**BERT features:** 768")
st.sidebar.write("**Linguistic features:** 19")
st.sidebar.write("**Total fused features:** 787")
st.sidebar.write("**Saved held-out accuracy:** 99.20%")
st.sidebar.caption(
    "Recomputed from the saved held-out feature array. This is not a "
    "fresh training run; performance on new articles may differ."
)


# ============================================================
# INPUT
# ============================================================

st.subheader("📝 Enter News Article")

title = st.text_input(
    "News Title",
    placeholder="Enter the news headline...",
)

article = st.text_area(
    "Article Content",
    height=300,
    placeholder="Paste the complete news article here...",
)

if st.button("🔍 Detect Fake News", type="primary"):
    if not title.strip() and not article.strip():
        st.warning("Please enter a news title or article content.")
        st.stop()

    # Match training document construction:
    # (title + " " + text).strip()
    # Do not independently strip each component before concatenation.
    full_text = (title + " " + article).strip()

    if not full_text:
        st.warning("Please enter a news title or article content.")
        st.stop()

    try:
        with st.spinner("Extracting linguistic features..."):
            doc = nlp(full_text)
            linguistic_features = extract_linguistic_features(doc)

        with st.spinner("Extracting BERT contextual features..."):
            bert_features = extract_bert_features(full_text)

        # Fuse in the exact training order: 768 BERT + 19 linguistic.
        fused_features = np.concatenate(
            [bert_features[0], linguistic_features],
            axis=0,
        ).reshape(1, -1)

        if fused_features.shape != (1, EXPECTED_TOTAL_FEATURES):
            st.error(
                "Feature dimension mismatch. "
                f"Expected (1, {EXPECTED_TOTAL_FEATURES}), "
                f"got {fused_features.shape}."
            )
            st.stop()

        if not np.isfinite(fused_features).all():
            st.error("The fused feature vector contains non-finite values.")
            st.stop()

        # Current app uses the saved UN-SCALED model.
        prediction = int(classifier.predict(fused_features)[0])
        probabilities = classifier.predict_proba(fused_features)[0]

        # Model classes are validated as [0, 1].
        real_probability = float(probabilities[0])
        fake_probability = float(probabilities[1])

    except Exception as exc:
        st.error("Prediction could not be completed.")
        st.exception(exc)
        st.stop()

    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    if prediction == 1:
        st.error("🚨 FAKE NEWS PATTERN DETECTED")
        confidence = fake_probability
    else:
        st.success("✅ REAL NEWS PATTERN DETECTED")
        confidence = real_probability

    st.metric("Model score for predicted class", f"{confidence * 100:.2f}%")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Real class score", f"{real_probability * 100:.2f}%")
    with col2:
        st.metric("Fake class score", f"{fake_probability * 100:.2f}%")

    st.caption(
        "These are model scores learned from the training dataset; "
        "they are not independent proof that the article is true or false."
    )

    # ========================================================
    # LINGUISTIC ANALYSIS
    # ========================================================

    st.divider()
    st.subheader("🔤 Linguistic Analysis")

    linguistic_df = pd.DataFrame(
        {
            "Feature": LINGUISTIC_FEATURE_NAMES,
            "Value": linguistic_features,
        }
    )
    st.dataframe(linguistic_df, width="stretch", hide_index=True)

    # ========================================================
    # SHAP EXPLANATION
    # ========================================================

    st.divider()
    st.subheader("🔎 Explainable AI — SHAP")

    try:
        if not BACKGROUND_PATH.exists():
            raise FileNotFoundError(
                f"SHAP background data not found: {BACKGROUND_PATH}"
            )

        background_all = np.load(BACKGROUND_PATH)

        if (
            background_all.ndim != 2
            or background_all.shape[1] != EXPECTED_TOTAL_FEATURES
        ):
            raise ValueError(
                "SHAP background must have shape (n_samples, 787); "
                f"got {background_all.shape}."
            )

        # Deterministically sample across the training rows rather than
        # always taking only the first 100 rows.
        background_size = min(100, len(background_all))
        sample_indices = np.linspace(
            0,
            len(background_all) - 1,
            num=background_size,
            dtype=int,
        )
        background = background_all[sample_indices]

        with st.spinner("Generating SHAP explanation..."):
            explainer = shap.LinearExplainer(classifier, background)
            shap_explanation = explainer(fused_features)

        shap_values = np.asarray(shap_explanation.values)

        # Handle binary output conventions across SHAP versions.
        if shap_values.ndim == 3:
            # samples x features x classes; class 1 = Fake
            shap_values = shap_values[0, :, 1]
        elif shap_values.ndim == 2:
            # samples x features
            shap_values = shap_values[0]
        elif shap_values.ndim == 1:
            pass
        else:
            raise ValueError(
                f"Unexpected SHAP output dimensions: {shap_values.shape}"
            )

        if shap_values.shape[0] != EXPECTED_TOTAL_FEATURES:
            raise ValueError(
                f"Expected 787 SHAP values, got {shap_values.shape[0]}."
            )

        linguistic_shap = shap_values[-EXPECTED_LINGUISTIC_FEATURES:]

        shap_df = pd.DataFrame(
            {
                "Feature": LINGUISTIC_FEATURE_NAMES,
                "Signed SHAP value (Fake class)": linguistic_shap,
                "Absolute SHAP magnitude": np.abs(linguistic_shap),
            }
        ).sort_values(
            "Absolute SHAP magnitude",
            ascending=False,
        ).head(10).reset_index(drop=True)

        st.write("Top 10 linguistic features by absolute SHAP magnitude:")
        st.dataframe(shap_df, width="stretch", hide_index=True)

        st.caption(
            "Positive signed SHAP values push the model toward the Fake class; "
            "negative values push it toward the Real class. Absolute magnitude "
            "shows strength, not direction. SHAP explains model behavior and "
            "does not verify factual truth."
        )

    except Exception as exc:
        st.warning("SHAP explanation could not be generated.")
        st.exception(exc)


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()
st.caption(
    "⚠️ This system identifies patterns learned from labeled text. "
    "It does not independently verify the factual truth of claims "
    "contained in a news article."
)
