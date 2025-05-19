from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings


def get_embedding_function():
    # Create an instance of the OllamaEmbeddings class with the model name "mistral"
    embeddings = OllamaEmbeddings(model="mistral")
    # Print the model name used for embeddings
    print(f"Using model: {embeddings.model}")
    # Return the embedding function
    return embeddings
