import io
import os
import hashlib
import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import pptx  # python-pptx
import pickle

# HuggingFace Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Azure AI Foundry Configuration
AZURE_ENDPOINT = "https://saljjjjj"
AZURE_MODEL_NAME = "DeepSeek-V3"
AZURE_API_KEY = "kkkkkk"

st.set_page_config(page_title="Customer Service AI", layout="centered")

icon = Image.open("assets/Customer_Service.png")
col1, col2 = st.columns([1, 12])
with col1:
    st.image(icon, width=200)



client = ChatCompletionsClient(
    endpoint=AZURE_ENDPOINT,
    credential=AzureKeyCredential(AZURE_API_KEY),
)

# Text extraction functions
def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    extracted_text = []
    ocr_text = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            extracted_text.append(f"\n--- Page {i+1} [Text Layer] ---\n{text}")
        else:
            pix = page.get_pixmap(dpi=150)
            image = Image.open(io.BytesIO(pix.tobytes("png")))
            image_ocr = pytesseract.image_to_string(image)
            if image_ocr.strip():
                ocr_text.append(f"\n--- Page {i+1} [OCR Layer] ---\n{image_ocr.strip()}")
    return "\n".join(extracted_text + ocr_text).strip()

def extract_text_from_pptx(file_path):
    prs = pptx.Presentation(file_path)
    text_runs = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text_runs.append(shape.text)
    return "\n".join(text_runs)

# Embedding creation
def create_vector_store_from_folder(folder_path):
    full_text = ""
    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)
        if filename.lower().endswith(".pdf"):
            full_text += "\n" + extract_text_from_pdf(full_path)
        elif filename.lower().endswith(('.ppt', '.pptx')):
            full_text += "\n" + extract_text_from_pptx(full_path)

    if not full_text.strip():
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_text(full_text)

    embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_texts(chunks, embedding=embed_model)

# Streamlit UI
st.markdown("<h1 style='color:black; text-align:center;'>AI Customer Service</h1>", unsafe_allow_html=True)
st.markdown("---")

# Folder selection
st.markdown("<h3 style='color: black; margin-bottom: 0.2rem;'>📂 Choose a product folder:</h3>", unsafe_allow_html=True)
category = st.selectbox("", ["TV", "OneDrive", "Cookers", "SDA"])
folder_paths = {
    "TV": "manuals/TV",
    "OneDrive": "manuals/OneDrive",
    "Cookers": "manuals/Cookers",
    "SDA": "manuals/SDA"
}
selected_folder = folder_paths[category]
cache_name = f".cache/faiss_folder_{category}.pkl"

# Load or create vector store
vector_store = None
if os.path.exists(cache_name):
    with open(cache_name, "rb") as f:
        vector_store = pickle.load(f)
    st.success("✅ Loaded cached embeddings.")
else:
    with st.spinner("🔍 Processing and embedding documents..."):
        vector_store = create_vector_store_from_folder(selected_folder)
        if vector_store:
            os.makedirs(".cache", exist_ok=True)
            with open(cache_name, "wb") as f:
                pickle.dump(vector_store, f)
            st.success("✅ Folder processed and cached.")
        else:
            st.error("❌ No supported documents found in the selected folder.")

# Prompt input
st.markdown("<h3 style='color: black;'>💬 Ask a question based on the documents:</h3>", unsafe_allow_html=True)
user_prompt = st.text_area("", placeholder="e.g., What is the warranty policy?", height=100)

# Generate answer

if st.button("🚀 Generate Response"):
    if user_prompt.strip() and vector_store:
        with st.spinner("🧠 Thinking..."):
            docs = vector_store.similarity_search(user_prompt, k=5)
            context = "\n\n".join([doc.page_content for doc in docs])

            messages = [
                SystemMessage(content="You are a helpful assistant that answers customer questions based on product documents."),
                UserMessage(content=f"Context:\n{context}"),
                UserMessage(content=f"Customer Question: {user_prompt}")
            ]

            response = client.complete(
                stream=True,
                messages=messages,
                max_tokens=2048,
                temperature=0.7,
                top_p=0.95,
                model=AZURE_MODEL_NAME
            )

            answer = ""
            for update in response:
                if update.choices:
                    answer += update.choices[0].delta.content or ""

            st.success("✅ AI Response:")
            st.markdown(f"<div style='background-color:#f1f1f1;padding:15px;border-radius:10px'>{answer}</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a question first.")

st.markdown("---")
st.markdown("<div style='text-align:center; color: gray;'>Made with ❤️ using Azure + HuggingFace + Streamlit</div>", unsafe_allow_html=True)
