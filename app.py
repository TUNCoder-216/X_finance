import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from groq import Groq
from transformers import pipeline
from utils import FinancialTextPreprocessor  # Your local preprocessor
import os
from pathlib import Path

# 1. PAGE CONFIG
st.set_page_config(page_title="Finetech Intelligence", layout="wide", page_icon="📈")

# 2. LOAD LOCAL MODELS (Cached to prevent reloading on every click)
@st.cache_resource
def load_models():
    # 1. Get the directory where app.py lives
    base_path = Path(__file__).parent.absolute()
    
    # 2. Hardcoded path to your exact folder
    model_dir = base_path / "models" / "financial_topic_model"
    
    # 3. Convert to string with forward slashes (Fixes the HF Repo ID error)
    model_path = str(model_dir.as_posix())
    
    if not (model_dir / "config.json").exists():
        st.error(f"❌ config.json not found in: {model_path}")
        st.stop()

    # 4. Load models
    topic_pipe = pipeline("text-classification", model=model_path)
    sent_pipe = pipeline("text-classification", model="ProsusAI/finbert")
    preprocessor = FinancialTextPreprocessor()
    
    return topic_pipe, sent_pipe, preprocessor

topic_model, sentiment_model, preprocessor = load_models()

# 3. LABEL MAPPING
LABEL_MAPPING = {
    "LABEL_0": "Analyst Update",
    "LABEL_1": "Fed | Central Banks",
    "LABEL_2": "Company | Product News",
    "LABEL_3": "Treasuries | Corporate Debt",
    "LABEL_4": "Dividend",
    "LABEL_5": "Earnings",
    "LABEL_6": "Energy | Oil",
    "LABEL_7": "Financials",
    "LABEL_8": "Currencies",
    "LABEL_9": "General News | Opinion",
    "LABEL_10": "Gold | Metals | Materials",
    "LABEL_11": "IPO",
    "LABEL_12": "Legal | Regulation",
    "LABEL_13": "M&A | Investments",
    "LABEL_14": "Macro",
    "LABEL_15": "Markets",
    "LABEL_16": "Politics",
    "LABEL_17": "Personnel Change",
    "LABEL_18": "Stock Commentary",
    "LABEL_19": "Stock Movement"
}

# 4. SIDEBAR - Settings & API Keys
with st.sidebar:
    st.title("⚙️ Settings")
    api_key = st.text_input("Groq API Key", type="password")
    st.info("This app uses local BERTs for classification and Llama 3 for market reasoning.")

# 5. MAIN UI
st.title("🦅 Finetech: AI Market Intelligence")
tweet_input = st.text_area("Paste a financial tweet/news here:", 
                          placeholder="$AAPL just announced a massive dividend increase...")

if st.button("Analyze Market Impact"):
    if not tweet_input:
        st.warning("Please enter some text first!")
    else:
        # --- A. PREPROCESSING ---
        # We define these variables right here so they are available for everything below
        clean_text = preprocessor.clean_text(tweet_input) # Fixed: calling the actual cleaning function
        tickers = preprocessor.extract_tickers(tweet_input)

        # --- B. CLASSIFICATION (Local BERTs) ---
        col1, col2 = st.columns(2) # Define columns so they aren't missing
        
        # 1. Topic Classification
        topic_res = topic_model(clean_text)[0]
        raw_label = str(topic_res['label']) 
        
        # Handle if model returns "2" instead of "LABEL_2"
        lookup_key = f"LABEL_{raw_label}" if raw_label.isdigit() else raw_label
        display_name = LABEL_MAPPING.get(lookup_key, lookup_key)
        
        with col1:
            st.metric("Predicted Topic", display_name, f"{topic_res['score']:.1%}")
            
        # 2. Sentiment Classification
        sent_res = sentiment_model(clean_text)[0]
        with col2:
            st.metric("Market Sentiment", sent_res['label'], f"{sent_res['score']:.1%}")

        st.divider()

        # --- C. RAG PART - Real-time Stock Data (Yahoo Finance) ---
        if tickers:
            st.subheader("📊 Real-Time Market Validation (RAG)")
            for ticker in tickers[:3]: 
                symbol = ticker.replace('$', '')
                try:
                    stock = yf.Ticker(symbol)
                    hist = stock.history(period="1mo")
                    if not hist.empty:
                        curr_p = hist['Close'].iloc[-1]
                        prev_p = hist['Close'].iloc[0]
                        change = ((curr_p - prev_p) / prev_p) * 100
                        
                        exp = st.expander(f"Live View: {symbol}", expanded=True)
                        c_a, c_b = exp.columns([1, 2])
                        c_a.metric(f"{symbol} Price", f"${curr_p:.2f}", f"{change:.2f}%")
                        
                        fig = go.Figure(data=[go.Candlestick(x=hist.index,
                                        open=hist['Open'], high=hist['High'],
                                        low=hist['Low'], close=hist['Close'])])
                        fig.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), xaxis_rangeslider_visible=False)
                        c_b.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not load data for {symbol}: {e}")
        else:
            st.info("No stock tickers detected in the text.")

        # --- D. LLM ANALYSIS (Groq - Llama 3.3) ---
        if api_key:
            st.subheader("🤖 AI Market Commentary")
            try:
                client = Groq(api_key=api_key)
                prompt = f"""
                Analyze this financial tweet: "{tweet_input}"
                Classified Topic: {display_name}
                Sentiment: {sent_res['label']}
                
                Provide a 3-point bulleted analysis on why this matters to investors.
                """
                
                chat = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile"
                )
                st.chat_message("assistant").write(chat.choices[0].message.content)
            except Exception as e:
                st.error(f"Groq Error: {e}")
        else:
            st.info("🔑 Enter your Groq API key in the sidebar for AI Commentary.")