import os

# Set the PyTorch CUDA allocation configuration to avoid memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM
import PyPDF2
import torch
from langchain.schema import Document
from langchain.llms.base import LLM
from typing import Optional

# Set up paths for your local environment
pdf_folder = './pdfs/'
model_path = './gpt-neo-2.7B/'  # path to the GPT-Neo 2.7B model directory
vector_store_path = './vector_store/'

# Check if a GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load PDF documents and structure as Document objects
def load_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            with open(os.path.join(folder_path, filename), "rb") as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                content = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        content += page_text
                if content.strip():
                    doc = Document(page_content=content, metadata={"filename": filename})
                    documents.append(doc)
                else:
                    print(f"Skipped {filename} due to empty content.")
    return documents

documents = load_documents(pdf_folder)

# Create or load vector store
vector_store = Chroma(persist_directory=vector_store_path, embedding_function=embedding_model)

# Index documents and store in vector store
vector_store.add_documents(documents)
vector_store.persist()

# Set up LLM (using GPT-Neo 2.7B with transformers) and move to GPU if available
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

# Define a custom class to wrap the transformer model for LangChain compatibility
class TransformersLLMWrapper(LLM):
    def _call(self, prompt: str, stop: Optional[list] = None) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_length=200)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    @property
    def _identifying_params(self):
        return {"model": "gpt-neo-2.7B"}

    @property
    def _llm_type(self):
        return "transformers"

# Initialize the custom LLM wrapper
llm = TransformersLLMWrapper()

# Create the LangChain RetrievalQA chain with the custom LLM
query_engine = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_type="similarity")
)

# Streamlit interface
st.title("Content Engine: PDF Comparison and Insights")

query = st.text_input("Enter your query (e.g., 'What are the risk factors associated with Google and Tesla?')")

if query:
    response = query_engine.run({"query": query})
    st.write("**Response:**", response)

st.write("\nUpload your PDF documents to analyze their content or make more complex comparisons!")

# Allow users to upload new PDF files
uploaded_files = st.file_uploader("Upload PDF documents", accept_multiple_files=True, type=["pdf"])
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(pdf_folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
    st.write("Files uploaded successfully. Please refresh the app to process them.")
