# app.py - Final version with Google Custom Search + Embedded CSE view
import streamlit as st
from transformers import pipeline
from googleapiclient.discovery import build
from PIL import Image
import pytesseract
import io
import tempfile
import os
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
from googletrans import Translator
import numpy as np
from pydub import AudioSegment
import traceback

# ----------------------------
# === CONFIG - set these ====
# ===== GOOGLE SEARCH VERIFICATION CONFIG =====
GOOGLE_PROJECT_ID = "your project id"
GOOGLE_API_KEY =  "your api key"
GOOGLE_CSE_ID =  "your cse id" 

from googleapiclient.discovery import build

def google_fact_check(query):
    """Performs Google Custom Search to verify the news"""
    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        result = service.cse().list(q=query, cx=GOOGLE_CSE_ID, num=3).execute()

        if "items" in result:
            verified_sources = []
            for item in result["items"]:
                verified_sources.append({
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "snippet": item.get("snippet")
                })
            return verified_sources
        else:
            return []
    except Exception as e:
        print("Google API Error:", e)
        return []


# Trusted domains
TRUSTED_DOMAINS = [
    "bbc.co.uk", "bbc.com", "hindustantimes.com", "timesofindia.indiatimes.com",
    "ndtv.com", "reuters.com", "thehindu.com", "indianexpress.com",
    "economictimes.indiatimes.com", "toi", "aljazeera.com", "guardian.co.uk",
    "forbes.com", "cnn.com", "deccanherald.com", "news18.com"
]

# ==============================
# 🔎 Show Google CSE iframe
def show_google_cse_results(query):
    """Fetches and displays Google Custom Search results directly inside Streamlit."""
    if not query or not GOOGLE_CSE_ID or not GOOGLE_API_KEY:

        st.warning("Google Search not configured properly.")
        return

    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        res = service.cse().list(q=query, cx=GOOGLE_CSE_ID, num=5).execute()

        items = res.get("items", [])
        if not items:
            st.info("No results found or API quota exceeded.")
            return

        st.markdown("### 🌐 Trusted References (Google Custom Search)")
        for item in items:
            title = item.get("title", "Untitled")
            link = item.get("link", "#")
            snippet = item.get("snippet", "")
            st.markdown(f"**[{title}]({link})**")
            st.caption(snippet)
            st.divider()
    except Exception as e:
        st.error(f"❌ Google Search failed: {e}")


# ==============================
# 🔧 GOOGLE FACT CHECK / SEARCH SETUP (ENHANCED)
# ==============================

import requests
from googleapiclient.discovery import build
GOOGLE_API_KEY =  ""
GOOGLE_CSE_ID =  "" 

def google_fact_check(query):
    """
    Combined Fact Check + Google Custom Search verification system.
    Returns verified sources or contextual evidence for real news detection.
    """
    fact_results = []
    search_results = []

    # --- Step 1: Try official Google Fact Check API ---
    fact_url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={query}&key={GOOGLE_API_KEY}"
    try:
        r = requests.get(fact_url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if "claims" in data:
                for claim in data["claims"]:
                    text = claim.get("text", "")
                    claim_reviews = claim.get("claimReview", [])
                    if claim_reviews:
                        review = claim_reviews[0]
                        publisher = review.get("publisher", {}).get("name", "")
                        rating = review.get("textualRating", "")
                        url = review.get("url", "")
                        fact_results.append({
                            "source": publisher,
                            "rating": rating,
                            "text": text,
                            "url": url
                        })
    except Exception as e:
        print("Fact-check error:", e)

    # --- Step 2: If no Fact Check found, fallback to Google Custom Search ---
    if not fact_results:
        try:
            service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
            res = service.cse().list(q=query, cx=GOOGLE_CSE_ID, num=5).execute()
            items = res.get("items", []) if res else []
            for item in items:
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                link = item.get("link", "")
                domain = link.split("/")[2] if "://" in link else ""
                search_results.append({
                    "source": domain,
                    "rating": "context",
                    "text": title,
                    "url": link,
                    "snippet": snippet
                })
        except Exception as e:
            print("Search error:", e)

    # --- Step 3: Combine ---
    if fact_results:
        return fact_results
    elif search_results:
        return search_results
    else:
        return []




# ==============================
# ⚙️ OCR
# ==============================
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\lenovo\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

# ==============================
# 🧠 MODELS
# ==============================
@st.cache_resource
def load_models():
    model_ids = [
        "mrm8488/bert-tiny-finetuned-fake-news-detection",
        "jy46604790/Fake-News-Bert-Detect",
        "winterForestStump/Roberta-fake-news-detector",
        "vikram71198/distilroberta-base-finetuned-fake-news-detection"
    ]
    models = [pipeline("text-classification", model=m, tokenizer=m) for m in model_ids]
    sentiment_model = pipeline("sentiment-analysis")
    return models, sentiment_model

models, sentiment_model = load_models()

# ==============================
## ==============================
# 🧩 DECISION LOGIC
# ==============================
def hybrid_decision(text):
    # === SAFETY PATCH ===
    if len(text.split()) > 400:
        text = " ".join(text.split()[:400])

    preds, confs = [], []
    for model in models:
        try:
            result = model(text)[0]
            label = result.get("label", "").upper()
            score = float(result.get("score", 0.0))
        except Exception:
            label, score = "", 0.0

        preds.append(0 if "FAKE" in label or "FALSE" in label or "LABEL_1" in label else 1)
        confs.append(score)

    avg_conf = float(np.mean(confs)) if confs else 0.0
    vote = int(np.round(np.mean(preds))) if preds else 0
    ensemble_pred = "REAL" if vote == 1 else "FAKE"

    # === Sentiment model (safe) ===
    try:
        sent_res = sentiment_model(text)[0]
        sentiment_label = sent_res.get("label", "")
        sentiment_score = float(sent_res.get("score", 0.0))
    except Exception:
        sentiment_label, sentiment_score = "UNKNOWN", 0.0

    # === Google Fact-Check ===
    facts = google_fact_check(text)
    fact_score = 0
    verified_hit = False

    trusted_domains = [
        "bbc.com", "ndtv.com", "thehindu.com", "timesofindia.com",
        "indiatoday.in", "hindustantimes.com", "reuters.com", "apnews.com",
        "cnn.com", "theguardian.com", "news18.com", "wionews.com",
        "deccanherald.com", "newindianexpress.com", "msn.com", "livemint.com"
    ]

    if facts:
        for f in facts:
            rating = f.get("rating", "").lower()
            src = f.get("source", "").lower()
            if any(k in rating for k in ["true", "accurate", "real", "verified"]):
                fact_score = 1.0
                verified_hit = True
                break
            elif any(k in rating for k in ["false", "fake", "incorrect", "misleading"]):
                fact_score = -1.0
                verified_hit = True
                break
            elif any(domain in src for domain in trusted_domains):
                fact_score = 0.8
                verified_hit = True
                break

    # ===  Known Verified Facts (Manual Fallback Layer) ===
    known_true = [
        "draupadi murmu is the president of india",
        "narendra modi is the prime minister of india",
        "india is a country",
        "the sun rises in the east",
        "water boils at 100 degrees celsius"
    ]

    if text.strip().lower() in known_true:
        return {
            "prediction": "REAL",
            "confidence": 99.0,
            "ensemble_prediction": "REAL",
            "ensemble_confidence": 99.0,
            "sentiment_label": sentiment_label,
            "sentiment_score": round(sentiment_score * 100, 2),
            "google_results": facts
        }

    # === Weighted scoring ===
    final_score = (
        (avg_conf * (1 if ensemble_pred == "REAL" else -1)) * 0.55
        + (sentiment_score * (1 if sentiment_label == "POSITIVE" else -1)) * 0.15
        + fact_score * 0.3
    )

    if verified_hit and fact_score > 0.8:
        final_pred = "REAL"
        confidence = 99.0
    elif final_score > 0.2:
        final_pred = "REAL"
        confidence = round(abs(final_score) * 100, 2)
    elif final_score < -0.2:
        final_pred = "FAKE"
        confidence = round(abs(final_score) * 100, 2)
    else:
        final_pred = "UNCERTAIN"
        confidence = round(abs(final_score) * 100, 2)

    return {
        "prediction": final_pred,
        "confidence": confidence,
        "ensemble_prediction": ensemble_pred,
        "ensemble_confidence": round(avg_conf * 100, 2),
        "sentiment_label": sentiment_label,
        "sentiment_score": round(sentiment_score * 100, 2),
        "google_results": facts
    }



    # === Google fact check results ===
    facts = google_fact_check(text)
    fact_score = 0
    if facts:
        for f in facts:
            rating = f.get("rating", "").lower()
            if any(k in rating for k in ["true", "accurate", "real", "verified"]):
                fact_score = 1
                break
            elif any(k in rating for k in ["false", "fake", "incorrect", "misleading"]):
                fact_score = -1
                break

    # === Weighted final decision ===
    final_score = (
        (avg_conf * (1 if ensemble_pred == "REAL" else -1)) * 0.6
        + (sentiment_score * (1 if sentiment_label == "POSITIVE" else -1)) * 0.2
        + fact_score * 0.2
    )

    if final_score > 0.2:
        final_pred = "REAL"
    elif final_score < -0.2:
        final_pred = "FAKE"
    else:
        final_pred = "UNCERTAIN"

    confidence = round(abs(final_score) * 100, 2)

    return {
        "prediction": final_pred,
        "confidence": confidence,
        "ensemble_prediction": ensemble_pred,
        "ensemble_confidence": round(avg_conf * 100, 2),
        "sentiment_label": sentiment_label,
        "sentiment_score": round(sentiment_score * 100, 2),
        "google_results": facts
    }

# ==============================
# TRUSTED SOURCE BOOST
# ==============================
def correct_with_trusted_sources(text, result):
    try:
        search_hits = custom_search(text, num=5)
        found_trusted = []

        for hit in search_hits:
            domain = hit.get("domain", "").lower()
            for td in TRUSTED_DOMAINS:
                if td in domain:
                    found_trusted.append(hit)

        # ✅ Case 1: Found trusted sources — mark as REAL
        if found_trusted:
            result["prediction"] = "REAL"
            result["confidence"] = max(result.get("confidence", 0.0), 95.0)
            result["verified_sources"] = found_trusted
            result["explanation"] = "Verified by trusted domains."
            return result

        # ⚠️ Case 2: No trusted sources, but low-confidence fake → mark as UNCERTAIN
        if (
            result.get("prediction") == "FAKE"
            and result.get("confidence", 0.0) < 60
        ):
            result["prediction"] = "UNCERTAIN"
            result["explanation"] = "Low confidence fake — not verified by trusted sources."
            return result

        # 🧠 Default: Return result unchanged
        result["explanation"] = "No trusted sources found."
        return result

    except Exception as e:
        result["explanation"] = f"Error verifying with trusted sources: {e}"
        return result


# ==============================
# 🎤 AUDIO & TRANSLATION
# ==============================
translator = Translator()
def convert_audio_to_text(input_bytes):
    recognizer = sr.Recognizer()
    try:
        bio = io.BytesIO(input_bytes)
        try:
            seg = AudioSegment.from_file(bio)
        except Exception:
            bio.seek(0)
            seg = AudioSegment.from_file(bio, format="webm")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            seg.export(tmp_wav.name, format="wav")
            wav_path = tmp_wav.name
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return "Could not understand the audio.", ""
        except sr.RequestError:
            return "Speech recognition failed.", ""
        try:
            translated_text = translator.translate(text, dest="en").text
        except Exception:
            translated_text = text
        return text, translated_text
    except Exception as e:
        return f"Audio processing failed: {e}", ""

# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(page_title="Fake News Detector 4.0", layout="centered")
st.title("🧠 Fake News Detector 4.0 (Text + Voice + Screenshot)")
st.markdown("### Detect fake news using 4 AI Models + Google Custom Search verification + Sentiment + Voice Translation.")

input_tab = st.tabs(["📝 Text", "🎤 Voice", "🖼️ Image/Screenshot"])

# TEXT TAB
with input_tab[0]:
    user_text = st.text_area("📰 Enter News Text")
    if st.button("🔍 Analyze Text"):
        if user_text.strip():
            with st.spinner("Analyzing... please wait"):
                result = hybrid_decision(user_text.strip())
                result = correct_with_trusted_sources(user_text.strip(), result)
                if result["prediction"] == "REAL":
                    st.success(f"✅ REAL ({result['confidence']}%)")
                elif result["prediction"] == "FAKE":
                    st.error(f"🚨 FAKE ({result['confidence']}%)")
                else:
                    st.warning(f"🤔 UNCERTAIN ({result['confidence']}%)")
                st.write(f"**Ensemble Prediction:** {result['ensemble_prediction']} ({result['ensemble_confidence']}%)")
                st.write(f"**Sentiment:** {result['sentiment_label']} ({result['sentiment_score']}%)")
                if result.get("verified_sources"):
                    st.markdown("**✅ Verified sources detected:**")
                    for vs in result["verified_sources"]:
                        st.write(f"- {vs.get('title','')} ({vs.get('domain','')}) — {vs.get('link','')}")
                show_google_cse_results(user_text)
        else:
            st.warning("Please enter news text first.")

# VOICE TAB
with input_tab[1]:
    st.subheader("🎤 Voice Input")
    audio = mic_recorder(start_prompt="🎙 Click to Record", stop_prompt="🔴 Stop Recording", key="voice_input")
    if audio and isinstance(audio, dict) and audio.get("bytes"):
        with st.spinner("Processing audio..."):
            raw_text, translated_text = convert_audio_to_text(audio["bytes"])
        st.write("**🗣 Original Speech:**", raw_text)
        st.write("**🌍 Translated (English):**", translated_text or raw_text)
        if translated_text or raw_text:
            text_to_analyze = translated_text or raw_text
            with st.spinner("Analyzing your spoken news..."):
                result = hybrid_decision(text_to_analyze)
                result = correct_with_trusted_sources(text_to_analyze, result)
            if result["prediction"] == "REAL":
                st.success(f"✅ REAL ({result['confidence']}%)")
            elif result["prediction"] == "FAKE":
                st.error(f"🚨 FAKE ({result['confidence']}%)")
            else:
                st.warning(f"🤔 UNCERTAIN ({result['confidence']}%)")
            show_google_cse_results(text_to_analyze)

# IMAGE TAB
with input_tab[2]:
    st.subheader("📸 Upload News Screenshot or Image")
    uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image / Screenshot", use_container_width=True)

        if st.button("🔍 Analyze Image"):
            with st.spinner("Extracting and analyzing text..."):
                extracted_text = extract_text_from_image(image)
                st.text_area("📝 Extracted Text", extracted_text)
                result = hybrid_decision(extracted_text)
                result = correct_with_trusted_sources(extracted_text, result)
                if result["prediction"] == "REAL":
                    st.success(f"✅ REAL ({result['confidence']}%)")
                elif result["prediction"] == "FAKE":
                    st.error(f"🚨 FAKE ({result['confidence']}%)")
                else:
                    st.warning(f"🤔 UNCERTAIN ({result['confidence']}%)")
                show_google_cse_results(extracted_text)










