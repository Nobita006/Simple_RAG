import os
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
model_path = 'distilgpt2'  # Hugging Face identifier for DistilGPT-2
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

# Set up LLM (using DistilGPT-2 with transformers) and move to GPU if available
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

# Define a custom class to wrap the transformer model for LangChain compatibility
class TransformersLLMWrapper(LLM):
    def _call(self, prompt: str, stop: Optional[list] = None) -> str:
        # Format and truncate the input if it exceeds a certain length (e.g., 512 tokens)
        formatted_prompt = f"Answer concisely based on the following context: {prompt}"
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        # Use `max_new_tokens` to specify the number of tokens to generate
        outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.5)  # Generate up to 100 new tokens
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    @property
    def _identifying_params(self):
        return {"model": "distilgpt2"}

    @property
    def _llm_type(self):
        return "transformers"

# Initialize the custom LLM wrapper
llm = TransformersLLMWrapper()

# Create the LangChain RetrievalQA chain with the custom LLM and limit retrieval to top 3 documents
query_engine = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_type="similarity", top_k=3)  # Retrieve top 3 most relevant documents
)

# Streamlit interface
st.title("Content Engine: PDF Comparison and Insights")

query = st.text_input("Enter your query (e.g., 'What are the risk factors associated with Google and Tesla?')")

if query:
    # Debug: Print query and retrieved documents
    print("Query:", query)
    retrieved_docs = vector_store.as_retriever(search_type="similarity", top_k=3).get_relevant_documents(query)
    print("Retrieved Documents:", [doc.page_content for doc in retrieved_docs])

    # Generate response based on the query
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
