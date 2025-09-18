import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import praw
import spacy
from nltk.stem import WordNetLemmatizer
import re
import json
import google.generativeai as genai
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Premium page configuration
st.set_page_config(
    page_title="Reddit Query Analyzer Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium CSS styling for the entire app
st.markdown("""
<style>
    @import url('https://fonts.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600&display=swap');
    
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --dark-bg: #0a0e27;
        --card-bg: rgba(16, 20, 39, 0.8);
        --glass-bg: rgba(255, 255, 255, 0.05);
        --neon-purple: #b794f4;
        --neon-blue: #63b3ed;
        --neon-pink: #f687b3;
        --success-color: #48bb78;
        --warning-color: #f6ad55;
        --error-color: #f56565;
        --text-primary: #ffffff;
        --text-secondary: rgba(255, 255, 255, 0.7);
        --border-radius: 12px;
    }
    
    body {
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
        background-color: var(--dark-bg);
    }
    
    .stApp {
        background-color: var(--dark-bg);
    }
    
    .main-container {
        max-width: 900px;
        margin: auto;
        padding: 2rem;
    }

    /* Header */
    .header {
        padding: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    h1.logo {
        font-family: 'Orbitron', monospace;
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(45deg, #667eea, #764ba2, #f093fb, #f5576c);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradient-shift 3s ease infinite;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        position: relative;
    }
    p.tagline {
        margin-top: 1rem;
        font-size: 1.2rem;
        color: var(--text-secondary);
        letter-spacing: 0.1em;
        animation: fade-in 1s ease-out;
    }
    
    /* Input and Button */
    [data-testid="stTextInput"] > div > div > input {
        border-radius: var(--border-radius);
        background-color: var(--glass-bg);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1rem;
        color: var(--text-primary);
        transition: all 0.3s ease;
    }
    
    [data-testid="stTextInput"] > div > div > input:focus {
        border-color: var(--neon-purple);
        box-shadow: 0 0 15px rgba(183, 148, 244, 0.5);
        outline: none;
    }

    [data-testid="stButton"] > button {
        background: var(--primary-gradient);
        border: none;
        border-radius: var(--border-radius);
        font-size: 1.1rem;
        font-weight: 600;
        color: white;
        cursor: pointer;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        min-width: 150px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    [data-testid="stButton"] > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    }
    
    [data-testid="stButton"] > button:active {
        transform: translateY(1px);
    }

    /* Status Messages */
    .status-info, .status-success, .status-warning {
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid;
        border-radius: 0 var(--border-radius) var(--border-radius) 0;
        font-style: italic;
        color: var(--text-secondary);
        background-color: var(--glass-bg);
    }
    .status-info { border-color: var(--neon-blue); }
    .status-success { border-color: var(--success-color); }
    .status-warning { border-color: var(--warning-color); }

    /* Results Panel */
    .analytic-panel {
        background-color: var(--card-bg);
        border-radius: var(--border-radius);
        padding: 2rem;
        margin-top: 2rem;
        border: 1px solid var(--glass-bg);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    .analytics-header {
        font-family: 'Orbitron', monospace;
        color: var(--neon-purple);
        text-align: center;
        margin-bottom: 2rem;
        font-size: 2rem;
        text-shadow: 0 0 10px rgba(183, 148, 244, 0.5);
    }

    /* Cards */
    .stance-card, .summary-card {
        padding: 1.5rem;
        border-radius: var(--border-radius);
        background-color: rgba(16, 20, 39, 0.6);
        box-shadow: inset 0 0 15px rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stance-title {
        font-size: 1rem;
        color: var(--text-secondary);
        font-weight: 300;
        margin-bottom: 0.5rem;
    }
    .stance-value {
        font-family: 'Orbitron', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: 0.1em;
    }
    .stance-confidence {
        font-size: 0.9rem;
        color: var(--text-secondary);
        font-style: italic;
    }
    .summary-card p {
        line-height: 1.8;
        font-size: 1.1rem;
        margin: 0;
    }

    /* Sources Expander */
    .st-emotion-cache-16 {
        border-radius: var(--border-radius);
        background-color: var(--card-bg);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 2rem;
    }
    .source-card {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: var(--border-radius);
        margin-bottom: 0.75rem;
        border-left: 3px solid var(--neon-blue);
        transition: transform 0.2s ease;
    }
    .source-card:hover {
        transform: translateX(5px);
        background-color: rgba(255, 255, 255, 0.1);
    }
    .source-title {
        font-weight: 600;
        color: var(--neon-purple);
    }
    .source-link {
        color: var(--neon-blue);
        text-decoration: none;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        display: inline-block;
    }
    .source-link:hover {
        text-decoration: underline;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 2rem;
        color: rgba(255, 255, 255, 0.6);
    }

    /* Animations */
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes pulse {
        0% { opacity: 0; transform: scale(0.8); }
        50% { opacity: 0.2; transform: scale(1.1); }
        100% { opacity: 0; transform: scale(0.8); }
    }
    @keyframes fade-in {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

</style>
""", unsafe_allow_html=True)

PRE_SUMMARIZER_PATH = "sshleifer/distilbart-cnn-12-6"
BASE_MODEL_PATH = "facebook/bart-large-cnn"

@st.cache_resource
def load_models_and_clients():
    """Initializes and caches all necessary models and clients."""
    print("--- Initializing models and clients (this runs only once) ---")
    
    # Check that all environment variables are present
    reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
    reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    reddit_user_agent = os.getenv("REDDIT_USER_AGENT")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not all([reddit_client_id, reddit_client_secret, reddit_user_agent, gemini_api_key]):
        st.error("Missing API keys. Please create a `.env` file with all the required credentials.")
        st.stop()
        
    try:
        reddit_client = praw.Reddit(
            client_id=reddit_client_id, 
            client_secret=reddit_client_secret, 
            user_agent=reddit_user_agent
        )
        genai.configure(api_key=gemini_api_key)
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        st.error(f"Failed to initialize clients. Check your API keys and network connection. Error: {e}")
        st.stop()
        
    nlp = spacy.load("en_core_web_sm")
    lemmatizer = WordNetLemmatizer()
    
    # Use GPU if available, otherwise CPU
    device = 0 if torch.cuda.is_available() else -1
    
    # Load models from Hugging Face Hub directly
    pre_summarizer = pipeline("summarization", model=PRE_SUMMARIZER_PATH, device=device)
    final_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_PATH)
    final_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    final_model.to(device)
    
    print("--- Initialization Complete ---")
    return nlp, lemmatizer, reddit_client, pre_summarizer, final_model, final_tokenizer, device, gemini_model

def extract_keywords_pos(query, nlp, lemmatizer):
    """Extracts keywords from a user query using NLP."""
    query = query.replace("bangaluru", "bengaluru")
    doc = nlp(query)
    keywords = [
        lemmatizer.lemmatize(token.text.lower())
        for token in doc
        if token.pos_ in ["NOUN", "PROPN", "ADJ"] and not token.is_stop and len(token.text) > 2
    ]
    return list(set(keywords))

def scrape_reddit(search_term, reddit_client):
    """Scrapes Reddit for relevant posts based on a search term."""
    try:
        submissions = reddit_client.subreddit("all").search(
            query=search_term, sort="relevance", time_filter="year", limit=5
        )
        posts = []
        seen_ids = set()
        for post in submissions:
            if post.selftext and not post.over_18 and post.id not in seen_ids:
                posts.append({
                    "title": post.title, "body": post.selftext, "url": f"https://www.reddit.com{post.permalink}"
                })
                seen_ids.add(post.id)
        return posts
    except Exception as e:
        st.error(f"Reddit scraping failed: {e}")
        return []

def get_stance_prediction(query, posts, gemini_model):
    """Uses Gemini to predict the overall stance of the Reddit posts."""
    if not posts:
        return {"stance": "unrelated", "confidence": 0.0}

    prompt_template = """You are given ONE user's question (stance target) and MULTIPLE Reddit posts.
    Task:
    - Decide the OVERALL stance of THESE POSTS with respect to the Question.
    - Choose exactly one label from: "support", "oppose", "neutral/mixed", "unrelated".
    - Provide a confidence in [0,1].
    - Return STRICT JSON only: {{"stance":"...","confidence":0.xx}}

    Question:
    {question}

    Posts:
    {posts_block}
    """
    
    post_texts = [f"Title: {p['title']}\nBody: {p['body']}" for p in posts]
    posts_block = "\n\n====\n\n".join(post_texts)
    
    prompt = prompt_template.format(question=query, posts_block=posts_block)
    
    try:
        response = gemini_model.generate_content(prompt)
        json_text = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_text:
            return json.loads(json_text.group())
        return {"stance": "neutral/mixed", "confidence": 0.5} # Fallback
    except Exception as e:
        st.error(f"Gemini API call failed: {e}")
        return {"stance": "error", "confidence": 0.0}

def run_pipeline(user_query):
    """Executes the full analysis pipeline and updates the UI."""
    status_container = st.empty()
    progress_bar = st.progress(0)
    
    def update_status(progress, message, status_type):
        icon_map = {
            "info": "🔎", "success": "📚", "analysis": "🧠", "distill": "📝", "generate": "✨", "warning": "⚠"
        }
        icon = icon_map.get(status_type, "➡️")
        markdown_str = f'<div class="status-{status_type}">{icon} {message}</div>'
        progress_bar.progress(progress)
        status_container.markdown(markdown_str, unsafe_allow_html=True)
        time.sleep(0.1)
    
    # 1. Keyword Extraction
    update_status(10, "Extracting keywords and searching Reddit...", "info")
    keywords = extract_keywords_pos(user_query, nlp, lemmatizer)
    search_term = " ".join(keywords) if keywords else user_query
    
    # 2. Reddit Scraping
    update_status(30, f"Searching Reddit for posts on \"{search_term}\"...", "info")
    scraped_posts = scrape_reddit(search_term, reddit_client)
    
    if not scraped_posts:
        update_status(100, "No relevant posts found on Reddit.", "warning")
        return "I couldn't find enough information on Reddit.", {"stance": "unrelated", "confidence": 0.0}, []
    
    update_status(50, f"Found {len(scraped_posts)} relevant posts.", "success")
    
    # 3. Stance Prediction
    update_status(65, "Analyzing post stance with Gemini AI...", "analysis")
    stance_result = get_stance_prediction(user_query, scraped_posts, gemini_model)
    
    # 4. Pre-summarization
    update_status(80, "Distilling posts with DistilBART...", "distill")
    post_bodies = [post['body'] for post in scraped_posts]
    pre_summaries = pre_summarizer(post_bodies, max_length=150, min_length=30, truncation=True)
    summary_texts = [s['summary_text'] for s in pre_summaries]
    
    # 5. Final Summarization
    update_status(95, "Generating final summary with BART-Large...", "generate")
    final_context = f" {final_tokenizer.sep_token} ".join(summary_texts)
    prefix = "summarize from multiple posts based on the query: "
    input_text = f"{prefix}{user_query} | context: {final_context}"
    
    inputs = final_tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    summary_ids = final_model.generate(**inputs, max_length=500, num_beams=4, early_stopping=True)
    final_summary = final_tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0]
    
    update_status(100, "Analysis Complete!", "success")
    time.sleep(0.5)
    
    # Clear the progress indicators
    status_container.empty()
    progress_bar.empty()

    return final_summary, stance_result, scraped_posts

# Load models and clients
nlp, lemmatizer, reddit_client, pre_summarizer, final_model, final_tokenizer, device, gemini_model = load_models_and_clients()

st.markdown('<div class="main-container">', unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="header">
    <h1 class="logo">Query-Conditioned Reddit Stance Summarize</h1>
    <p class="tagline">AI-powered analysis of Reddit discussions with stance detection and intelligent summarization</p>
</div>
""", unsafe_allow_html=True)

# --- Main container ---
col1, col2 = st.columns([3, 1])

with col1:
    user_query = st.text_input(
        "",
        placeholder="Enter your question... (e.g., Should I move to Bengaluru?)",
        value="Should I move to Bengaluru?",
        label_visibility="collapsed"
    )

with col2:
    analyze_button = st.button("Analyze", type="primary", use_container_width=True)

# --- Results Section ---
if analyze_button and user_query:
    summary, stance, sources = run_pipeline(user_query)
    
    st.markdown('<h2 class="analytics-header">Analysis Results</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    
    with col1:
        stance_label = stance.get('stance', 'N/A').upper()
        confidence = stance.get('confidence', 0) * 100
        
        stance_colors = {
            'SUPPORT': '#10b981',
            'OPPOSE': '#ef4444', 
            'NEUTRAL/MIXED': '#f59e0b',
            'UNRELATED': '#6b7280'
        }
        stance_color = stance_colors.get(stance_label, '#6b7280')

        st.markdown(f"""
        <div class="stance-card" style="border-left: 4px solid {stance_color};">
            <div class="stance-title">Overall Stance</div>
            <div class="stance-value" style="color: {stance_color};">{stance_label}</div>
            <div class="stance-confidence">Confidence: {confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="summary-card"><p>{summary}</p></div>', unsafe_allow_html=True)
    
    if sources:
        st.markdown('<div class="sources-expander-container">', unsafe_allow_html=True)
        with st.expander(f"📚 View {len(sources)} Source Posts", expanded=False):
            for i, post in enumerate(sources, 1):
                st.markdown(f"""
                <div class="source-card">
                    <div class="source-title">{i}. {post['title']}</div>
                    <a href="{post['url']}" target="_blank" class="source-link">📖 Read full discussion on Reddit</a>
                </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

elif analyze_button and not user_query:
    st.markdown('<div class="status-warning">⚠ Please enter a query to analyze.</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<div class="footer">
    <p>Powered by advanced AI models including BART, DistilBART, spaCy, and Google Gemini</p>
</div>
""", unsafe_allow_html=True)
st.markdown("""</div>""", unsafe_allow_html=True)
