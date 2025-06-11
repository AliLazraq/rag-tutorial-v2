# get_embedding_function.py
from langchain_ollama import OllamaEmbeddings

def get_embedding_function():
    model_name = "mxbai-embed-large"
    embeddings = OllamaEmbeddings(model=model_name)
    print(f"✅ Using Ollama embedding model: {model_name}")
    return embeddings
