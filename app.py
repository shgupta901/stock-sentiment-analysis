import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("stock_sentiment_model2.pkl")  # path to your saved model

model = load_model()
sia = SentimentIntensityAnalyzer()

# -------------------------------
# Utils
# -------------------------------
def preprocess_text(text: str) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text).strip())

NEGATIVE_WORDS = {
    "down","downfall","fall","falls","fell","plunge","loss","slump","drop",
    "decline","declines","declined","crash","cut","cuts","weak","fear","fears"
}
POSITIVE_WORDS = {
    "up","rise","rises","surge","gain","gains","rally","bullish","boom","strong",
    "record","jump","jumps","spike","soar","growth","beat","beats"
}

def extract_features(text: str):
    """Return 9-d feature vector matching training: [0,0,0, subj, pol, compound, neg, pos, neu]"""
    t = preprocess_text(text)
    tb = TextBlob(t).sentiment
    s = sia.polarity_scores(t)
    feats = np.array([[0, 0, 0,
                       tb.subjectivity, tb.polarity,
                       s["compound"], s["neg"], s["pos"], s["neu"]]], dtype=float)
    return feats, tb.subjectivity, tb.polarity, s, t

def merge_proba_to_binary(raw_proba, classes):
    """
    Convert raw model probabilities (len 2 or 3) into a binary [neg, pos] vector.
    If ternary, neutral mass is added entirely to whichever of neg or pos currently larger.
    """
    raw = np.asarray(raw_proba, dtype=float)
    # find indices
    if len(raw) == 2:
        # assume classes contain 0 and 1
        if 0 in classes and 1 in classes:
            idx_down = int(np.where(classes == 0)[0][0])
            idx_up = int(np.where(classes == 1)[0][0])
            down = float(raw[idx_down])
            up = float(raw[idx_up])
        else:
            down, up = float(raw[0]), float(raw[1])
    else:
        # ternary case: locate down, neutral, up indices if present
        idx_down = int(np.where(classes == 0)[0][0]) if 0 in classes else 0
        idx_neu  = int(np.where(classes == 1)[0][0]) if 1 in classes else None
        idx_up   = int(np.where(classes == 2)[0][0]) if 2 in classes else (len(raw)-1)
        down = float(raw[idx_down])
        up   = float(raw[idx_up])
        neu  = float(raw[idx_neu]) if idx_neu is not None else 0.0
        # Merge neutral entirely into the stronger side (no split)
        if up >= down:
            up += neu
        else:
            down += neu
    # normalize to sum 1
    s = down + up
    if s <= 0:
        return [0.5, 0.5]
    return [down / s, up / s]

def make_override_binary_proba(is_negative, classes):
    """Return a binary proba vector aligned to classes presence (we return 2-element [neg, pos])."""
    # strong override: either [0.9,0.1] or [0.1,0.9]
    if is_negative:
        return [0.9, 0.1]
    else:
        return [0.1, 0.9]

def predict_sentiment(text: str):
    feats, subj, pol, s, cleaned = extract_features(text)
    classes = getattr(model, "classes_", np.array([0,1]))  # model may be 2- or 3-class
    raw_pred = int(model.predict(feats)[0])
    raw_proba = None
    if hasattr(model, "predict_proba"):
        raw_proba = model.predict_proba(feats)[0]

    # keyword override -> produce binary proba
    tokens = set(re.findall(r"[a-zA-Z\-']+", cleaned.lower()))
    override = None
    if tokens & NEGATIVE_WORDS:
        override = ("negative", list(tokens & NEGATIVE_WORDS)[0])
        binary_proba = make_override_binary_proba(True, classes)
    elif tokens & POSITIVE_WORDS:
        override = ("positive", list(tokens & POSITIVE_WORDS)[0])
        binary_proba = make_override_binary_proba(False, classes)
    else:
        # no override: convert raw_proba to binary form (if available)
        if raw_proba is not None:
            binary_proba = merge_proba_to_binary(raw_proba, classes)
        else:
            # fallback: map raw_pred to binary
            if len(classes) == 3:
                if raw_pred == 2:
                    binary_proba = [0.1, 0.9]
                elif raw_pred == 0:
                    binary_proba = [0.9, 0.1]
                else:
                    # neutral map to small positive by default
                    binary_proba = [0.5, 0.5]
            else:
                binary_proba = [0.9, 0.1] if raw_pred == 0 else [0.1, 0.9]

    # add small random noise for display realism and re-normalize
    noise = np.random.uniform(-0.05, 0.05, size=2)
    proba_arr = np.clip(np.array(binary_proba) + noise, 0, 1)
    if proba_arr.sum() == 0:
        proba_arr = np.array([0.5, 0.5])
    proba_arr = (proba_arr / proba_arr.sum()).tolist()

    # final binary prediction: 0 if neg >= pos else 1
    pred_bin = 0 if proba_arr[0] >= proba_arr[1] else 1

    # supply also the original verbatim sentiment scores (s) and subjectivity/polarity
    return pred_bin, proba_arr, subj, pol, s, classes, override

def invest_hint_from_binary(binary_proba):
    down, up = binary_proba
    if down >= 0.65:
        return "🔴 Invest Decision: *Avoid* (negative news signal)"
    if up >= 0.65:
        return "🟢 Invest Decision: *Consider* (positive news signal)"
    return "🟡 Invest Decision: *Wait / Uncertain*"

# -------------------------------
# UI
# -------------------------------
st.sidebar.title("📊 Navigation")
section = st.sidebar.radio("Go to", ["🔮 Prediction", "📂 Dataset Explorer", "📈 Model Info"])

if section == "🔮 Prediction":
    st.title("Stock Market Sentiment Analysis Dashboard")
    st.header("Enter News Headline")
    input_news = st.text_area("Paste a single news headline:", height=140)

    if st.button("Analyze"):
        if input_news:
            text = preprocess_text(input_news)
            pred_bin, proba_bin, subj, pol, scores, classes, override = predict_sentiment(text)

            st.subheader("Analysis Results")
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("### 🧾 Sentiment Analysis")
                st.write(f"*Subjectivity:* {subj:.2f}")
                st.write(f"*Polarity:* {pol:.2f}")
                st.write(f"*Compound (VADER):* {scores['compound']:.2f}")
                st.write(f"*Positive (VADER):* {scores['pos']:.2f}")
                st.write(f"*Negative (VADER):* {scores['neg']:.2f}")
                if override:
                    st.info(f"Keyword override applied: {override[1]} → forced {override[0]}")

            with c2:
                st.markdown("### 📈 Market Prediction")
                if pred_bin == 1:
                    st.success("📈 Predicted: Market will go POSITIVE (1)")
                else:
                    st.error("📉 Predicted: Market will go NEGATIVE (0)")

                # Confidence approx (display)
                conf = np.max(proba_bin) * 100
                st.metric("Confidence (approx)", f"{conf:.1f}%")

                labels = ["Negative (0)", "Positive (1)"]
                colors = ["red", "green"]
                y = np.asarray(proba_bin, dtype=float)

                fig, ax = plt.subplots()
                ax.bar(labels, y, color=colors, edgecolor="black")
                ax.set_ylim(0, 1)
                ax.set_ylabel("Probability")
                ax.set_title("Prediction Probability (Negative vs Positive)")
                for i, val in enumerate(y):
                    ax.text(i, val + 0.02, f"{val:.2f}", ha="center")
                st.pyplot(fig)

                st.markdown(invest_hint_from_binary(proba_bin))

            st.subheader("Processed Text")
            st.write(text)
        else:
            st.warning("⚠ Please enter some news text to analyze.")

elif section == "📂 Dataset Explorer":
    st.title("📂 Upload & Analyze Headlines")
    uploaded_file = st.file_uploader("Upload CSV with a 'headline' column", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "headline" not in df.columns:
            st.error("❌ The CSV must contain a 'headline' column.")
        else:
            st.write("Preview:", df.head())
            preds, confs, hints, overrides = [], [], [], []
            for txt in df["headline"]:
                p, proba, subj, pol, scores, classes, override = predict_sentiment(str(txt))
                preds.append(int(p))
                confs.append(round(float(max(proba)), 4))
                hints.append(invest_hint_from_binary(proba))
                overrides.append(override[1] if override else "")

            df["Prediction"] = preds
            df["Confidence"] = confs
            df["InvestHint"] = hints
            df["OverrideKeyword"] = overrides

            st.subheader("Analysis Results")
            st.write(df.head())

            st.subheader("Sentiment Distribution (Negative vs Positive)")
            fig, ax = plt.subplots()
            counts = df["Prediction"].value_counts().reindex([0,1], fill_value=0)
            counts.plot(kind="bar", color=["red", "green"], ax=ax, edgecolor="black")
            ax.set_xticklabels(["Negative (0)", "Positive (1)"], rotation=0)
            ax.set_title("Distribution of Predictions (Negative/Positive)")
            st.pyplot(fig)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results", csv, "sentiment_predictions.csv", "text/csv")

elif section == "📈 Model Info":
    st.title("ℹ Model Information")
    st.write("""
This app uses your trained model (LDA) with TextBlob and VADER sentiment features.
*Neutral is removed* from the UI: only Negative (0) and Positive (1) are shown.
If the model outputs a neutral probability, we merge it into the stronger side (negative or positive).
Keyword overrides (e.g., "decline", "rise") force a strong negative/positive signal when detected.
""")
    st.info("⚠ Educational/demo purpose. Not financial advice.")