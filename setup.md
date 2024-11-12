py --list
py -3.10 -m venv env

# On Windows
env\Scripts\activate
# On Linux/macOS
env/bin/activate

python --version

pip install langchain streamlit PyMuPDF chromadb sentence-transformers transformers llama-cpp-python
pip install nvidia-pyindex
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118  

streamlit run demo.py