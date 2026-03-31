import os
import openai
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Doc Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Sora', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        color: #e2e8f0;
    }

    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }

    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }

    .main-header p {
        color: #94a3b8;
        font-size: 1rem;
    }

    .chat-bubble-user {
        background: linear-gradient(135deg, #1e40af, #1d4ed8);
        border-radius: 18px 18px 4px 18px;
        padding: 0.75rem 1.1rem;
        margin: 0.5rem 0;
        margin-left: 20%;
        color: #e0f2fe;
        font-size: 0.95rem;
    }

    .chat-bubble-ai {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 18px 18px 18px 4px;
        padding: 0.75rem 1.1rem;
        margin: 0.5rem 0;
        margin-right: 20%;
        color: #cbd5e1;
        font-size: 0.95rem;
    }

    .doc-preview {
        background: #0f172a;
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #7dd3fc;
        max-height: 200px;
        overflow-y: auto;
        white-space: pre-wrap;
    }

    .stats-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }

    .stats-number {
        font-size: 1.6rem;
        font-weight: 700;
        color: #60a5fa;
    }

    .stats-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .stButton > button {
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        font-family: 'Sora', sans-serif;
        font-weight: 600;
        transition: all 0.2s;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4);
    }

    .stTextInput > div > div > input {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        color: #e2e8f0;
        font-family: 'Sora', sans-serif;
    }

    .sidebar .stFileUploader {
        background: #1e293b;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid #334155;
    }

    .feature-badge {
        display: inline-block;
        background: linear-gradient(135deg, #065f46, #047857);
        color: #6ee7b7;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        margin: 0.2rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# ─── open ai Client ───────────────────────────────────────────────────────────
@st.cache_resource
def get_client():
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        st.error("⚠️ OPENAI_API_KEY not set.")
        st.stop()
    openai.api_key = api_key
    return openai

# ─── Text Extraction ─────────────────────────────────────────────────────────
def extract_text_pdf(file) -> str:
    pdf = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in pdf.pages)

def extract_text_docx(file) -> str:
    doc = Document(file)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

# ─── Smart Chunking (prevents token overflow) ────────────────────────────────
def smart_truncate(text: str, max_chars: int = 12000) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n[... middle truncated for length ...]\n\n" + text[-half:]

# ─── Openai API Call ─────────────────────────────────────────────────────────
def ask_openai(client, system_prompt: str, messages: list) -> str:
    full_messages = [{"role": "system", "content": system_prompt}] + messages

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # you can change to gpt-4o if needed
        messages=full_messages,
        temperature=0.3,
        max_tokens=1024
    )

    return response.choices[0].message.content

# ─── Session State Init ──────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""
if "doc_name" not in st.session_state:
    st.session_state.doc_name = ""

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📄 AI Doc Assistant</h1>
    <p>Upload any document and chat with it using Openai AI</p>
    <div>
        <span class="feature-badge">✦ Multi-turn Chat</span>
        <span class="feature-badge">✦ PDF & DOCX & TXT</span>
        <span class="feature-badge">✦ Smart Summarizer</span>
        <span class="feature-badge">✦ Key Points Extractor</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Upload Document")
    uploaded_file = st.file_uploader(
        "Supported: PDF, DOCX, TXT",
        type=["pdf", "docx", "txt"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        with st.spinner("Reading document..."):
            if uploaded_file.type == "application/pdf":
                text = extract_text_pdf(uploaded_file)
            elif "wordprocessingml" in uploaded_file.type:
                text = extract_text_docx(uploaded_file)
            else:
                text = uploaded_file.read().decode("utf-8", errors="ignore")

            st.session_state.doc_text = text
            st.session_state.doc_name = uploaded_file.name
            st.session_state.chat_history = []  # Reset chat on new doc

        st.success(f"✅ Loaded: **{uploaded_file.name}**")

        # Stats
        words = len(text.split())
        chars = len(text)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""<div class="stats-card">
                <div class="stats-number">{words:,}</div>
                <div class="stats-label">Words</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="stats-card">
                <div class="stats-number">{chars:,}</div>
                <div class="stats-label">Chars</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Quick Actions
        st.markdown("### ⚡ Quick Actions")

        client = get_client()
        doc_ctx = smart_truncate(st.session_state.doc_text)
        system = f"You are a helpful document assistant. The document is:\n\n{doc_ctx}"

        if st.button("📝 Summarize (5 lines)"):
            with st.spinner("Summarizing..."):
                summary = ask_openai(client, system, [
                    {"role": "user", "content": "Summarize this document in exactly 5 concise bullet points."}
                ])
                st.session_state.chat_history.append({"role": "assistant", "content": f"**📝 Summary:**\n{summary}"})

        if st.button("🔑 Extract Key Points"):
            with st.spinner("Extracting..."):
                key_points = ask_openai(client, system, [
                    {"role": "user", "content": "List the 5 most important key points from this document as numbered bullets."}
                ])
                st.session_state.chat_history.append({"role": "assistant", "content": f"**🔑 Key Points:**\n{key_points}"})

        if st.button("❓ Generate Quiz Questions"):
            with st.spinner("Generating questions..."):
                quiz = ask_openai(client, system, [
                    {"role": "user", "content": "Generate 3 quiz questions with answers based on this document."}
                ])
                st.session_state.chat_history.append({"role": "assistant", "content": f"**❓ Quiz:**\n{quiz}"})

        st.markdown("---")

        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

        # Doc Preview
        if st.session_state.doc_text:
            with st.expander("👁️ Preview Document Text"):
                st.markdown(f'<div class="doc-preview">{st.session_state.doc_text[:1500]}...</div>', unsafe_allow_html=True)

# ─── Main Chat Area ──────────────────────────────────────────────────────────
if not st.session_state.doc_text:
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; color: #475569;">
        <div style="font-size:4rem;">📂</div>
        <h3 style="color:#64748b;">Upload a document to get started</h3>
        <p>Supports PDF, DOCX, and TXT files</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Chat history display
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown(f"""
            <div style="text-align:center; padding:2rem; color:#475569;">
                <div style="font-size:2rem;">💬</div>
                <p>Document loaded! Ask anything about <strong style="color:#60a5fa">{st.session_state.doc_name}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-bubble-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-bubble-ai">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Input area
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_question = st.text_input(
            "Ask a question",
            placeholder="e.g. What is the main conclusion of this document?",
            label_visibility="collapsed",
            key="user_input"
        )
    with col_btn:
        send = st.button("Send ➤")

    if send and user_question.strip():
        client = get_client()
        doc_ctx = smart_truncate(st.session_state.doc_text)
        system = f"You are a helpful document assistant. Answer questions based ONLY on the document below. If the answer is not in the document, say so clearly.\n\nDocument:\n{doc_ctx}"

        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        # Build messages list for multi-turn context (last 6 turns)
        recent = st.session_state.chat_history[-6:]
        api_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in recent
            if m["role"] in ("user", "assistant")
        ]

        with st.spinner("Thinking..."):
            try:
                answer = ask_openai(client, system, api_messages)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"API Error: {e}")

        st.rerun()
