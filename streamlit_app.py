import os
import streamlit as st
from rag_chatbot import ClassBasedRAGChatbot

# 0. Configure page settings (must be the first Streamlit command)
st.set_page_config(
    page_title="WEBDEV Assistant — Smart RAG Chatbot",
    page_icon="💻",
    layout="centered"
)

# Resolve paths relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AVATAR_PATH = os.path.join(SCRIPT_DIR, "Bot.avif")
USER_AVATAR_PATH = os.path.join(SCRIPT_DIR, "User_images.png")

# App title styled with beautiful dark modern webdev theme
st.markdown('<div class="title-m">DevBot — WEBDEV RAG Chatbot</div>', unsafe_allow_html=True)

# Custom CSS for clean layout styling (White base, Slate text/headers, Royal Blue highlights, Muted Slate elements)
st.markdown("""
<style>
    /* Main app background color - Pure White (#FFFFFF) */
    .stApp {
        background-color: #FFFFFF;
        color: #1E293B;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    /* Title text color styling - Slate Blue-Gray (#1E293B) with Royal Blue gradient hint */
    .title-m {
        font-family: 'Outfit', sans-serif;
        color: #1E293B;
        background: linear-gradient(135deg, #1E293B 0%, #2563EB 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 2rem;
    }
    
    /* Customize chat message containers (Keep dark chat style for contrast) */
    div[data-testid="stChatMessage"] {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(30, 41, 59, 0.1);
        border-radius: 16px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    div[data-testid="stChatMessage"]:hover {
        border-color: rgba(37, 99, 235, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(37, 99, 235, 0.15);
    }
    
    /* Style the name header inside chat messages */
    div[data-testid="stChatMessage"] h5 {
        color: #3b82f6 !important;
        font-weight: 700;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
        letter-spacing: 0.05em;
    }
    
    /* User chat message customization - Royal Blue (#2563EB) */
    div[data-testid="stChatMessage"] img[src*="user"] {
        border: 2px solid #2563EB;
    }
    
    /* Source expanders styling - Muted Element (#F1F5F9) */
    .streamlit-expanderHeader {
        background-color: #F1F5F9 !important;
        border: 1px solid rgba(0, 0, 0, 0.06) !important;
        border-radius: 12px !important;
        color: #1E293B !important;
        font-weight: 600 !important;
    }
    
    /* Chat input element styling - Royal Blue (#2563EB) border details */
    div[data-testid="stChatInput"] {
        border: 1px solid rgba(30, 41, 59, 0.15) !important;
        border-radius: 14px !important;
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
    }
    
    /* Customize scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #FFFFFF;
    }
    ::-webkit-scrollbar-thumb {
        background: #F1F5F9;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #2563EB;
    }
</style>
""", unsafe_allow_html=True)

# 1. Configure GEMINI_API_KEY in the environment for the ClassBasedRAGChatbot
gemini_key = None

try:
    # Prioritize Streamlit secrets (so secrets.toml changes take effect immediately)
    if "GEMINI_API_KEY" in st.secrets:
        gemini_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

# Fallback to ambient environment variable if secrets are not defined
if not gemini_key:
    gemini_key = os.environ.get("GEMINI_API_KEY")

if gemini_key:
    os.environ["GEMINI_API_KEY"] = gemini_key
else:
    st.error("🔑 GEMINI_API_KEY is not set. Please set the GEMINI_API_KEY environment variable or configure Streamlit secrets.")
    st.stop()

# 2. Hardcode PDF Path (relative to the app)
HARDCODED_PDF = os.path.join(SCRIPT_DIR, "webdev-tutorial.pdf")

# Dynamic DB dir based on the PDF name to prevent data cross-contamination
PDF_BASE_NAME = os.path.splitext(os.path.basename(HARDCODED_PDF))[0]
DB_DIR = os.path.join(SCRIPT_DIR, f"rag_db_{PDF_BASE_NAME}")

# 3. Cache the Chatbot instance to load models only once
@st.cache_resource
def get_chatbot():
    bot = ClassBasedRAGChatbot(
        db_dir=DB_DIR,
        llm_model="gemini-2.5-flash"
    )
    if os.path.exists(HARDCODED_PDF):
        bot.ingest_pdf(HARDCODED_PDF)
        print("SUCCESS: PDF Ingested!")      # Shows in your terminal
        st.success("✅ SUCCESS: PDF Ingested!") # Shows in your web app
    else:
        print("ERROR: PDF not found!")
        st.error("❌ ERROR: PDF not found!")
    return bot
# Initialize RAG chatbot
chatbot = get_chatbot()

# Initialize chat history with a welcome message from Mona Lisa
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Welcome! I am DevBot, your WEBDEV Tutorial assistant. How can I help you design, develop, and publish your site today?",
            "sources": []
        }
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar = AVATAR_PATH if message["role"] == "assistant" else USER_AVATAR_PATH
    with st.chat_message(message["role"], avatar=avatar):
        if message["role"] == "assistant":
            st.markdown("##### DevBot")
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("View Source References"):
                for idx, src in enumerate(message["sources"], 1):
                    doc = src["document"]
                    st.write(f"**Reference {idx} — Page {doc.metadata.get('page')} (Score: {src['score']:.4f})**")
                    st.write(doc.page_content)

# Accept user input
if prompt := st.chat_input("Ask a question about the document..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user", avatar=USER_AVATAR_PATH):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant", avatar=AVATAR_PATH):
        st.markdown("##### DevBot")
        with st.spinner("Retrieving and generating..."):
            # Use the .ask() method defined in ClassBasedRAGChatbot
            result = chatbot.ask(prompt, initial_k=20, final_k=3)
            answer = result["answer"]
            sources = result["sources"]

            # Display generated answer
            st.markdown(answer)
            
            # Display source expander if sources were retrieved
            if sources:
                with st.expander("View Source References"):
                    for idx, src in enumerate(sources, 1):
                        doc = src["document"]
                        st.write(f"**Reference {idx} — Page {doc.metadata.get('page')} (Score: {src['score']:.4f})**")
                        st.write(doc.page_content)

        # Add assistant message to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })