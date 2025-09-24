import openai
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document

# Function to extract text from PDF
def extract_text_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

# Function to extract text from DOCX
def extract_text_docx(file):
    doc = Document(file)
    text = " ".join([p.text for p in doc.paragraphs])
    return text

# Function to call OpenAI LLM
def ask_llm(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response['choices'][0]['message']['content']

# Streamlit UI
st.title("AI Document Assistant")

uploaded_file = st.file_uploader("Upload your document", type=["pdf", "docx", "txt"])
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        doc_text = extract_text_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc_text = extract_text_docx(uploaded_file)
    else:
        doc_text = uploaded_file.read().decode("utf-8")

    st.text_area("Document Content", doc_text, height=300)

    question = st.text_input("Ask a question about this document:")
    if st.button("Get Answer") and question:
        answer = ask_llm(f"Answer this question based on the document: {doc_text}\nQuestion: {question}")
        st.write(answer)

    if st.button("Summarize Document"):
        summary = ask_llm(f"Summarize the following document in 5 lines: {doc_text}")
        st.write(summary)
