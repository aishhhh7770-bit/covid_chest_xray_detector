"""
Chest X-Ray COVID-19 & Pneumonia Detector
University Project — Streamlit App with AI Chatbot + RAG
"""

import os
import re
import math
import numpy as np
import cv2
import streamlit as st
from groq import Groq

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Chest X-Ray AI Detector",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
CLASSES       = ['COVID19', 'NORMAL', 'PNEUMONIA']
IMG_DIMS      = 150
RAG_DOCS_DIR  = os.path.join(os.path.dirname(__file__), "rag_docs")
MODEL_PATH    = os.path.join(os.path.dirname(__file__), "chest_xray_covid_detector.keras")

CLASS_COLORS = {
    "COVID19":    "#ef4444",
    "NORMAL":     "#22c55e",
    "PNEUMONIA":  "#f97316",
}

CLASS_ICONS = {
    "COVID19":    "🦠",
    "NORMAL":     "✅",
    "PNEUMONIA":  "🫁",
}


# ─────────────────────────────────────────────
# CSS STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* App background */
    .stApp { background: #0f172a; color: #e2e8f0; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background: #1e293b; border-right: 1px solid #334155; }
    
    /* Cards */
    .card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }
    
    /* Prediction badge */
    .pred-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 999px;
        font-size: 1.2rem;
        font-weight: 700;
        margin: 8px 0;
    }
    
    /* Prob bar */
    .prob-bar-bg {
        background: #334155;
        border-radius: 999px;
        height: 12px;
        margin: 4px 0;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.4s ease;
    }
    
    /* Chat bubbles */
    .chat-user {
        background: #1d4ed8;
        color: #fff;
        border-radius: 16px 16px 4px 16px;
        padding: 10px 16px;
        margin: 6px 0 6px 60px;
        font-size: 0.95rem;
    }
    .chat-bot {
        background: #1e293b;
        border: 1px solid #334155;
        color: #e2e8f0;
        border-radius: 16px 16px 16px 4px;
        padding: 10px 16px;
        margin: 6px 60px 6px 0;
        font-size: 0.95rem;
    }
    
    /* Disclaimer */
    .disclaimer {
        background: #431407;
        border: 1px solid #c2410c;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.85rem;
        color: #fed7aa;
        margin: 10px 0;
    }
    
    /* Section header */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 16px 0 8px 0;
    }
    
    /* Metric chip */
    .metric-chip {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 10px 14px;
        text-align: center;
        margin: 4px;
    }
    
    /* Hide default streamlit header */
    #MainMenu, footer { visibility: hidden; }
    
    /* Input box */
    .stTextInput > div > div > input {
        background: #0f172a !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# RAG: LOAD KNOWLEDGE BASE
# ─────────────────────────────────────────────
@st.cache_resource
def load_knowledge_base():
    """Load all .txt files from rag_docs/ into memory as chunked docs."""
    docs = []
    if not os.path.exists(RAG_DOCS_DIR):
        return docs
    for fname in os.listdir(RAG_DOCS_DIR):
        if fname.endswith(".txt"):
            with open(os.path.join(RAG_DOCS_DIR, fname), "r", encoding="utf-8") as f:
                content = f.read()
            # Split into sections by SECTION headers
            sections = re.split(r'\nSECTION \d+:', content)
            for sec in sections:
                sec = sec.strip()
                if len(sec) > 50:
                    docs.append(sec)
    return docs


def simple_keyword_search(query: str, docs: list, top_k: int = 3) -> list:
    """
    Lightweight keyword-based retrieval (no embeddings needed — zero extra deps).
    Scores each chunk by how many query words appear in it.
    """
    query_words = set(re.findall(r'\w+', query.lower()))
    scores = []
    for i, doc in enumerate(docs):
        doc_lower = doc.lower()
        hits = sum(1 for w in query_words if w in doc_lower)
        scores.append((hits, i))
    scores.sort(reverse=True)
    return [docs[i] for _, i in scores[:top_k] if scores[0][0] > 0]


# ─────────────────────────────────────────────
# LOAD MODEL (lazy, cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        return model, None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
def predict_xray(image_bytes, model):
    """Run CNN inference and return (class, confidence, probs_dict)."""
    nparr  = np.frombuffer(image_bytes, np.uint8)
    img    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img    = cv2.resize(img, (IMG_DIMS, IMG_DIMS))
    inp    = img.astype("float32") / 255.0
    inp    = np.expand_dims(inp, axis=0)
    probs  = model.predict(inp, verbose=0)[0]
    idx    = int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx]) * 100, {c: float(p)*100 for c, p in zip(CLASSES, probs)}


# ─────────────────────────────────────────────
# CHATBOT WITH RAG
# ─────────────────────────────────────────────
def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    if not api_key:
        return None
    return Groq(api_key=api_key)


def chat_with_rag(user_msg: str, chat_history: list, docs: list,
                   prediction_context: str = "") -> str:
    """Call Groq with RAG-retrieved context injected into the system prompt."""
    client = get_groq_client()
    if not client:
        return "⚠️ Groq API key not configured. Add it to `.streamlit/secrets.toml` as `GROQ_API_KEY`."

    # RAG retrieval
    retrieved = simple_keyword_search(user_msg, docs, top_k=3)
    rag_context = "\n\n---\n\n".join(retrieved) if retrieved else "No specific context found."

    # System prompt
    system = f"""You are a helpful medical AI assistant for a Chest X-Ray COVID-19 & Pneumonia Detector app.
This is a university project built with a custom CNN (94% accuracy) classifying X-rays into: COVID19, NORMAL, PNEUMONIA.

RETRIEVED KNOWLEDGE BASE CONTEXT (use this to answer accurately):
{rag_context}

{f'CURRENT PREDICTION CONTEXT: {prediction_context}' if prediction_context else ''}

RULES:
- Answer questions about X-ray findings, COVID-19, pneumonia, the model, and image analysis
- Always remind users this is a UNIVERSITY PROJECT and NOT a clinical diagnostic tool
- Be empathetic and clear; use simple language for medical terms
- If asked about specific symptoms or personal medical advice, always recommend seeing a doctor
- Keep answers concise and helpful (2-4 sentences unless more is needed)
"""

    messages = [{"role": "system", "content": system}]
    for h in chat_history[-6:]:  # Last 6 messages for context window
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_msg})

    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.4,
            max_tokens=400,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"❌ API error: {str(e)}"


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""

# Load knowledge base
kb_docs = load_knowledge_base()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫁 Chest X-Ray AI")
    st.markdown("**University Project** — UMT")
    st.markdown("---")

    st.markdown("### 📋 About This App")
    st.markdown("""
- Custom **CNN** with 5 SeparableConv2D blocks  
- Trained on Kaggle dataset  
- **94% test accuracy**  
- Classes: COVID19 · NORMAL · PNEUMONIA  
- RAG-powered AI chatbot  
""")

    st.markdown("---")
    st.markdown("### ℹ️ Model Details")
    st.markdown("""
| Parameter | Value |
|-----------|-------|
| Input size | 150×150 RGB |
| Optimizer | Adam (lr=1e-3) |
| Loss | Categorical CE |
| Epochs | 15 |
| Accuracy | ~94% |
""")

    st.markdown("---")
    st.markdown('<div class="disclaimer">⚠️ <b>For educational purposes only.</b> Not a substitute for professional medical diagnosis.</div>', unsafe_allow_html=True)

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ─────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────
st.markdown("# 🫁 Chest X-Ray COVID-19 & Pneumonia Detector")
st.markdown("Upload a chest X-ray image to get an AI prediction, then ask the chatbot anything about the result.")
st.markdown("---")

col_left, col_right = st.columns([1, 1], gap="large")

# ── LEFT COLUMN: Upload + Prediction ──
with col_left:
    st.markdown('<div class="section-title">📤 Upload Chest X-Ray</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload a chest X-ray image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded:
        img_bytes = uploaded.read()

        # Show image
        st.image(img_bytes, caption="Uploaded X-Ray", use_column_width=True)

        # Predict button
        if st.button("🔍 Analyze X-Ray", use_container_width=True, type="primary"):
            model, err = load_model()
            if err:
                st.error(f"❌ Model not loaded: {err}\n\nMake sure `chest_xray_covid_detector.keras` is in the app folder.")
                st.info("💡 **Demo mode:** Place your trained model file next to `app.py` and rerun.")
            else:
                with st.spinner("Analyzing..."):
                    pred_class, confidence, probs = predict_xray(img_bytes, model)
                    st.session_state.prediction = {
                        "class": pred_class,
                        "confidence": confidence,
                        "probs": probs,
                    }
                st.rerun()

        # Show prediction result
        if st.session_state.prediction:
            p = st.session_state.prediction
            color = CLASS_COLORS[p["class"]]
            icon  = CLASS_ICONS[p["class"]]

            st.markdown(f"""
<div class="card">
  <div class="section-title">🎯 Prediction Result</div>
  <div style="text-align:center; margin: 12px 0;">
    <span class="pred-badge" style="background:{color}20; color:{color}; border: 2px solid {color};">
      {icon} {p['class']}
    </span>
    <div style="font-size:2rem; font-weight:800; color:{color}; margin-top:4px;">
      {p['confidence']:.1f}% confident
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

            # Probability bars
            st.markdown('<div class="section-title">📊 Class Probabilities</div>', unsafe_allow_html=True)
            for cls in CLASSES:
                prob  = p["probs"][cls]
                clr   = CLASS_COLORS[cls]
                icon2 = CLASS_ICONS[cls]
                st.markdown(f"""
<div style="margin: 8px 0;">
  <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
    <span>{icon2} <b>{cls}</b></span>
    <span style="color:{clr}; font-weight:700;">{prob:.1f}%</span>
  </div>
  <div class="prob-bar-bg">
    <div class="prob-bar-fill" style="width:{prob}%; background:{clr};"></div>
  </div>
</div>
""", unsafe_allow_html=True)

            # Disclaimer
            st.markdown("""
<div class="disclaimer">
⚠️ This prediction is generated by a university AI project and is <b>not a medical diagnosis</b>. 
Always consult a qualified radiologist or physician.
</div>
""", unsafe_allow_html=True)

    else:
        # Placeholder
        st.markdown("""
<div class="card" style="text-align:center; padding:40px;">
  <div style="font-size:3rem; margin-bottom:12px;">🫁</div>
  <div style="color:#64748b; font-size:1rem;">
    Upload a chest X-ray image (JPG/PNG)<br>to get an AI-powered analysis
  </div>
</div>
""", unsafe_allow_html=True)


# ── RIGHT COLUMN: AI Chatbot ──
with col_right:
    st.markdown('<div class="section-title">🤖 AI Medical Chatbot (RAG-Powered)</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:#64748b; font-size:0.85rem;">Knowledge base: {len(kb_docs)} sections loaded · Ask anything about X-ray findings, COVID-19, pneumonia, or the model.</div>', unsafe_allow_html=True)

    # Chat window
    chat_container = st.container(height=440)
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
<div class="chat-bot">
👋 Hi! I'm your AI assistant for this Chest X-Ray detector.<br>
Ask me about:
<ul style="margin:6px 0 0 0; padding-left:18px;">
<li>What does a COVID-19 X-ray look like?</li>
<li>How is pneumonia diagnosed?</li>
<li>What does the prediction mean?</li>
<li>What are the model's limitations?</li>
</ul>
</div>
""", unsafe_allow_html=True)

        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bot">{msg["content"]}</div>', unsafe_allow_html=True)

    # Input
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Your message",
            placeholder="Ask about X-ray findings, COVID-19, the model...",
            label_visibility="collapsed"
        )
        submitted = st.form_submit_button("Send 💬", use_container_width=True, type="primary")

    if submitted and user_input.strip():
        # Build prediction context if available
        pred_ctx = ""
        if st.session_state.prediction:
            p = st.session_state.prediction
            pred_ctx = (
                f"The model just predicted '{p['class']}' with {p['confidence']:.1f}% confidence. "
                f"Probabilities: COVID19={p['probs']['COVID19']:.1f}%, "
                f"NORMAL={p['probs']['NORMAL']:.1f}%, "
                f"PNEUMONIA={p['probs']['PNEUMONIA']:.1f}%."
            )

        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.spinner("Thinking..."):
            reply = chat_with_rag(user_input, st.session_state.chat_history, kb_docs, pred_ctx)

        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

    # Quick question chips
    st.markdown('<div class="section-title">💡 Quick Questions</div>', unsafe_allow_html=True)
    quick_qs = [
        "What does COVID-19 look like on X-ray?",
        "What is the model accuracy?",
        "What are the limitations of this AI?",
        "How is pneumonia different from COVID-19?",
    ]
    q_cols = st.columns(2)
    for i, q in enumerate(quick_qs):
        with q_cols[i % 2]:
            if st.button(q, use_container_width=True, key=f"quick_{i}"):
                pred_ctx = ""
                if st.session_state.prediction:
                    p = st.session_state.prediction
                    pred_ctx = f"Prediction: {p['class']} ({p['confidence']:.1f}% confidence)."
                st.session_state.chat_history.append({"role": "user", "content": q})
                reply = chat_with_rag(q, st.session_state.chat_history, kb_docs, pred_ctx)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                st.rerun()

# ─────────────────────────────────────────────
# BOTTOM: HOW IT WORKS
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔬 How It Works")
how_cols = st.columns(4)
steps = [
    ("1️⃣", "Upload X-Ray", "Upload a chest X-ray image (JPG/PNG)"),
    ("2️⃣", "CNN Inference", "5-block SeparableConv2D CNN analyzes the image"),
    ("3️⃣", "Prediction", "Model outputs COVID19 / NORMAL / PNEUMONIA with confidence"),
    ("4️⃣", "Ask the Bot", "RAG chatbot retrieves clinical knowledge to explain results"),
]
for col, (icon, title, desc) in zip(how_cols, steps):
    with col:
        st.markdown(f"""
<div class="card" style="text-align:center;">
  <div style="font-size:2rem;">{icon}</div>
  <div style="font-weight:700; margin:6px 0; color:#e2e8f0;">{title}</div>
  <div style="color:#64748b; font-size:0.85rem;">{desc}</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div style="text-align:center; color:#475569; font-size:0.8rem; margin-top:20px;">University Project · UMT · Built with TensorFlow, Streamlit & RAG · For Educational Use Only</div>', unsafe_allow_html=True)
