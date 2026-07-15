# 🫁 Chest X-Ray COVID-19 & Pneumonia Detector

A Streamlit web app with Claude AI chatbot integration for chest X-ray classification.

## Features
- **X-Ray Analysis tab** — upload any chest X-ray and get COVID19 / NORMAL / PNEUMONIA prediction with confidence scores
- **AI Chatbot tab** — Claude-powered assistant to explain results, answer medical questions, and explain confusion matrix values
- **Model Insights tab** — full confusion matrix breakdown, explanation of the 20 and 46 values, training summary

## Setup & Run Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your trained model
Place `chest_xray_covid_detector.keras` (saved from your Colab notebook) in the same folder as `app.py`.

### 3. Run
```bash
streamlit run app.py
```

### 4. Enter API Key
In the sidebar, paste your Anthropic API key (`sk-ant-...`) from https://console.anthropic.com

---

## Deploy on Streamlit Cloud (Free)

1. Push this folder to a GitHub repo
2. Go to https://share.streamlit.io
3. Connect your repo → set `app.py` as the main file
4. In **Advanced settings → Secrets**, add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-your-key-here"
   ```
5. Deploy!

> **Note:** For Streamlit Cloud deployment, you can read the API key from secrets:
> ```python
> api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
> ```

---

## Confusion Matrix — Why 20 and 46?

| Predicted → | COVID19 | NORMAL | PNEUMONIA |
|-------------|---------|--------|-----------|
| **Actual COVID19** (116) | **114** | 0 | 2 |
| **Actual NORMAL** (317)  | 0 | **269** | **46** ← |
| **Actual PNEUMONIA** (855)| 0 | **20** ← | **835** |

- **46** = NORMAL images predicted as PNEUMONIA (False Positives for Pneumonia). NORMAL recall = 85%, so ~15% of 317 ≈ 46 were misclassified. This happens because subtle lung density variations in healthy X-rays can resemble pneumonia patterns.
- **20** = PNEUMONIA images predicted as NORMAL (False Negatives — more dangerous!). PNEUMONIA recall = 98%, so ~2% of 855 ≈ 17–20 missed. Early/mild pneumonia consolidation is hard to distinguish from normal tissue.
