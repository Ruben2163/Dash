import streamlit as st
import yfinance as yf
import requests
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import pandas as pd

# --- Settings ---
PASSWORD = "yourpass"
TICKERS = ["AAPL", "MSFT", "GOOG", "TSLA", "AMD", "META", "TSM", "BTC-USD", "BRK-B"]
NEWS_API_KEY = "your_news_api_key_here"

# --- FinBERT Sentiment Setup ---
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
labels = ['negative', 'neutral', 'positive']

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)
    sentiment = labels[torch.argmax(probs)]
    confidence = torch.max(probs).item()
    return sentiment, confidence

# --- Authentication ---
st.set_page_config(page_title="📊 Market Dashboard", layout="wide")
password_input = st.text_input("Enter password to access dashboard:", type="password")
if password_input != PASSWORD:
    st.stop()

st.title("📊 In-Depth Market Dashboard")
st.caption(f"Last updated: {datetime.now().strftime('%A, %d %B %Y | %H:%M')}")

# --- Live Stock Data ---
st.subheader("📈 Market Snapshot")
data = []
for t in TICKERS:
    try:
        ticker = yf.Ticker(t)
        hist = ticker.history(period="2d")
        if len(hist) >= 2:
            change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100
            data.append({
                "Ticker": t,
                "Price": round(hist['Close'].iloc[-1], 2),
                "Change (%)": round(change, 2)
            })
    except:
        continue
df = pd.DataFrame(data)
st.dataframe(df.style.applymap(lambda v: 'color: red' if isinstance(v, float) and v < 0 else 'color: green', subset=["Change (%)"]))

# --- News Section ---
st.subheader("📰 Financial Headlines with Sentiment")
try:
    url = f"https://newsapi.org/v2/top-headlines?category=business&language=en&apiKey={NEWS_API_KEY}"
    res = requests.get(url)
    articles = res.json().get("articles", [])[:5]
    for article in articles:
        title = article['title']
        sentiment, confidence = analyze_sentiment(title)
        st.markdown(f"**[{title}]({article['url']})**")
        st.markdown(f"Sentiment: `{sentiment}` | Confidence: `{round(confidence * 100, 1)}%`")
        st.markdown("---")
except Exception as e:
    st.error("Failed to fetch news.")

