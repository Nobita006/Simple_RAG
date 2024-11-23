
# **Content Engine: PDF Comparison and Insights**

## Overview

This application allows users to upload PDF documents, store their content, and ask questions based on these documents. The application utilizes **LangChain** for handling question-answering tasks and **DistilGPT-2** or **GPT-Neo 2.7B** as a language model to generate responses (because of resource limitation). It's designed to extract relevant information from PDFs and answer user queries based on the uploaded documents.

The application has a simple, interactive **Streamlit** user interface that enables users to:
- Enter queries about the content in uploaded PDFs.
- Upload new PDF documents as needed.
- View responses generated based on the context of their uploaded files.

---

## Screenshots

![Screenshot_12-11-2024_114834_localhost](https://github.com/user-attachments/assets/934fbab0-0e14-4e3a-a743-7ecba8fc8b67)


![Screenshot_12-11-2024_114729_localhost](https://github.com/user-attachments/assets/f10d2dbc-000b-49a5-95e1-003b3a82118d)

## Features

### 1. **Query Input**
   - Users can enter a question related to the contents of uploaded PDF documents.
   - The response will be generated based on the context of the stored PDF content.

### 2. **PDF Uploading**
   - Users can upload PDF documents via a file uploader in the UI.
   - Once uploaded, documents are automatically processed and stored in a vector database for retrieval.
   - Uploaded documents are saved in the `./pdfs/` folder locally.

### 3. **Automatic Context Retrieval**
   - When a query is entered, the application fetches the top 3 most relevant documents related to the query to improve answer accuracy.
   - The application uses a similarity-based retrieval mechanism to find context from uploaded PDFs.

### 4. **Embedding and Vector Store**
   - The application uses **MiniLM-L6-v2** embeddings to represent document content.
   - Documents are stored in a **Chroma** vector database, allowing fast retrieval based on content similarity.

### 5. **Language Model Integration**
   - Uses **DistilGPT-2** for generating responses.
   - If a GPU is available, the model will leverage it to optimize performance.

---

## Setup Guide

### Prerequisites
1. **Python** (recommended version 3.8 or higher).
2. **GPU Setup (Optional)**: A CUDA-enabled GPU is recommended for faster response generation, especially when working with large document sets or complex queries.
3. **Streamlit**: This library is used for the web UI.
4. **PyTorch**: Needed for running `DistilGPT-2` model or `gpt-neo-2.7B` model.
5. **Other Python Libraries**: Install dependencies listed below.

### Installation Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Nobita006/Alemeno_LLM_Assignment.git
   cd Alemeno_LLM_Assignment
   ```

2. **Install Dependencies**:
   Ensure you are in the project directory, then install required libraries:
   ```bash
   pip install streamlit torch transformers langchain chromadb PyPDF2 sentence-transformers
   ```

3. **Prepare Model and Vector Store Directories**:
   Make sure the following directories exist:
   - `./pdfs/`: For storing uploaded PDFs.
   - `./vector_store/`: For storing document embeddings.

4. **Download the Model**:
    ```bash
   git lfs install
   git clone https://huggingface.co/distilbert/distilgpt2
   ```
    or
    ```bash
   git lfs install
   git clone https://huggingface.co/EleutherAI/gpt-neo-2.7B
   ```

5. **Run the Application**:
   Start the Streamlit server by running:
   ```bash
   streamlit run usingdistilgpt2.py
   ```
   or
   ```bash
   streamlit run usingGpt-neo-2.7B.py
   ```

5. **Access the Application**:
   - By default, Streamlit runs on `http://localhost:8501`.
   - Open a browser and navigate to this URL to interact with the app.

---

## User Guide

### Home Interface

Upon starting the application, you will see the main interface:

1. **Query Input**:
   - Enter a question in the text box. For example: `"What are the risk factors associated with Google and Tesla?"`.
   - Press `Enter` or click outside the text box to submit the query.
   - The application will process the query and display a response based on the uploaded documents.

2. **Response Display**:
   - The response area below the query box shows the answer generated based on the retrieved PDF content.
   - The response may take a few moments to load, depending on the document size and processing power.

3. **Uploading New PDFs**:
   - Scroll down to the **Upload PDF Documents** section.
   - Use the file uploader to drag and drop PDF files or browse your local files.
   - Once uploaded, the files will be stored in the `./pdfs/` folder and indexed in the vector store for future queries.
   - **Note**: Pdfs that are already present in `./pdfs/` folder do not need to be uploaded.

---

## Code Overview and Explanation

### 1. **Load and Process PDFs**
   - The function `load_documents(folder_path)` reads all PDF files in the specified folder.
   - Extracts text from each page and creates a LangChain `Document` object for each PDF.

### 2. **Embedding and Vector Store**
   - Using **MiniLM-L6-v2** model for embeddings, each document is represented as a vector.
   - The **Chroma** vector database stores these embeddings, allowing fast and accurate retrieval based on query similarity.

### 3. **Retrieving Relevant Documents**
   - When a query is submitted, the system retrieves the top 3 documents related to the query.
   - This limits the content scope, improving the relevance and quality of generated responses.

### 4. **Language Model Setup**
   - The **GPT** model is loaded and configured to run on GPU if available.
   - A custom LangChain-compatible wrapper (`TransformersLLMWrapper`) was created to integrate DistilGPT-2 with LangChain.

### 5. **Query and Response Generation**
   - The query is processed by **RetrievalQA** to find the most relevant content.
   - The selected content is formatted into a prompt for the language model, and the model generates a concise response.

---


