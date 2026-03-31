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
    html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        color: #e2e8f0;
    }
    .main-header { text-align: center; padding: 1.5rem 0 0.5rem 0; }
    .main-header h1 {
        font-size: 2.4rem; font-weight: 700;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #34d399);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .main-header p { color: #94a3b8; font-size: 0.95rem; }
    .feature-badge {
        display: inline-block;
        background: linear-gradient(135deg, #065f46, #047857);
        color: #6ee7b7; padding: 0.15rem 0.5rem; border-radius: 20px;
        font-size: 0.65rem; font-weight: 600; margin: 0.15rem;
        text-transform: uppercase; letter-spacing: 0.05em;
    }
    .chat-bubble-user {
        background: linear-gradient(135deg, #1e40af, #1d4ed8);
        border-radius: 18px 18px 4px 18px;
        padding: 0.7rem 1rem; margin: 0.4rem 0; margin-left: 15%;
        color: #e0f2fe; font-size: 0.9rem; line-height: 1.5;
    }
    .chat-bubble-ai {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155; border-radius: 18px 18px 18px 4px;
        padding: 0.7rem 1rem; margin: 0.4rem 0; margin-right: 15%;
        color: #cbd5e1; font-size: 0.9rem; line-height: 1.5;
    }
    .doc-badge {
        background: linear-gradient(135deg, #1e3a5f, #1e293b);
        border: 1px solid #1e40af; border-radius: 8px;
        padding: 0.5rem 1rem; color: #93c5fd; font-size: 0.8rem;
        text-align: center; margin-bottom: 0.5rem;
    }
    .stats-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155; border-radius: 10px;
        padding: 0.8rem; text-align: center;
    }
    .stats-number { font-size: 1.4rem; font-weight: 700; color: #60a5fa; }
    .stats-label { font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid #334155;
    }
    .stButton > button {
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        color: white; border: none; border-radius: 8px;
        font-family: 'Sora', sans-serif; font-weight: 600; width: 100%; transition: all 0.2s;
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 15px rgba(124,58,237,0.4); }
    .welcome-box {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px dashed #334155; border-radius: 16px;
        padding: 2.5rem; text-align: center; color: #475569; margin: 1rem 0;
    }
    .welcome-box h3 { color: #64748b; font-size: 1.1rem; margin-bottom: 0.5rem; }
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

def smart_truncate(text: str, max_chars: int = 12000) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n[... middle section truncated ...]\n\n" + text[-half:]

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
    <p>Chat with AI — upload a document to ask questions about it</p>
    <div>
        <span class="feature-badge">✦ Always-on Chat</span>
        <span class="feature-badge">✦ PDF · DOCX · TXT</span>
        <span class="feature-badge">✦ Multi-turn Memory</span>
        <span class="feature-badge">✦ Smart Summarizer</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Upload Document *(optional)*")
    st.caption("Chat works without a document too!")

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

            if st.session_state.doc_name != uploaded_file.name:
                st.session_state.doc_text = text
                st.session_state.doc_name = uploaded_file.name
                st.session_state.chat_history = []

        st.success(f"✅ **{uploaded_file.name}**")

        words = len(text.split())
        chars = len(text)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="stats-card"><div class="stats-number">{words:,}</div><div class="stats-label">Words</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="stats-card"><div class="stats-number">{chars:,}</div><div class="stats-label">Chars</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### ⚡ Quick Actions")

        client = get_client()
        doc_ctx = smart_truncate(st.session_state.doc_text)
        sys_doc = f"You are a helpful document assistant. The document is:\n\n{doc_ctx}"

        if st.button("📝 Summarize in 5 bullets"):
            with st.spinner("Summarizing..."):
                result = ask_openai(client, sys_doc, [{"role": "user", "content": "Summarize this document in exactly 5 concise bullet points."}])
                st.session_state.chat_history.append({"role": "assistant", "content": f"**📝 Summary:**\n\n{result}"})
            st.rerun()

        if st.button("🔑 Extract Key Points"):
            with st.spinner("Extracting..."):
                result = ask_openai(client, sys_doc, [{"role": "user", "content": "List the 5 most important key points as numbered bullets."}])
                st.session_state.chat_history.append({"role": "assistant", "content": f"**🔑 Key Points:**\n\n{result}"})
            st.rerun()

        if st.button("❓ Generate Quiz Questions"):
            with st.spinner("Generating..."):
                result = ask_openai(client, sys_doc, [{"role": "user", "content": "Generate 3 quiz questions with answers based on this document."}])
                st.session_state.chat_history.append({"role": "assistant", "content": f"**❓ Quiz:**\n\n{result}"})
            st.rerun()

        with st.expander("👁️ Preview Text"):
            st.text(st.session_state.doc_text[:1000] + "...")

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ─── Main Chat Area ──────────────────────────────────────────────────────────
if st.session_state.doc_name:
    st.markdown(f'<div class="doc-badge">📄 Chatting about: <strong>{st.session_state.doc_name}</strong></div>', unsafe_allow_html=True)

if not st.session_state.chat_history:
    st.markdown("""
    <div class="welcome-box">
        <div style="font-size:3rem;">🤖</div>
        <h3>Hello! I'm your AI assistant.</h3>
        <p>Ask me anything — coding, writing, ideas, analysis.<br>
        Or <strong>upload a document</strong> on the left to chat about it!</p>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-bubble-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble-ai">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

# ─── Chat Input (ALWAYS VISIBLE) ─────────────────────────────────────────────
user_input = st.chat_input("Ask me anything... or upload a document and ask about it!")

if user_input:
    client = get_client()

    if st.session_state.doc_text:
        doc_ctx = smart_truncate(st.session_state.doc_text)
        system = (
            f"You are a helpful AI assistant. A document has been uploaded. "
            f"Answer questions based on the document when relevant, otherwise answer generally.\n\n"
            f"Document ({st.session_state.doc_name}):\n{doc_ctx}"
        )
    else:
        system = (
            "You are a helpful, knowledgeable AI assistant. "
            "Answer questions clearly and helpfully. "
            "If the user mentions a document, remind them they can upload one using the sidebar on the left."
        )

    st.session_state.chat_history.append({"role": "user", "content": user_input})

    recent = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.chat_history[-10:]
        if m["role"] in ("user", "assistant")
    ]

    with st.spinner("Thinking..."):
        try:
            reply = ask_openai(client, system, recent)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
        except Exception as e:
            st.session_state.chat_history.append({"role": "assistant", "content": f"⚠️ Error: {e}"})

    st.rerun()
