# --- populate_database.py ---
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from get_embedding_function import get_embedding_function

DATA_PATH = "../data"
CHROMA_FOLDER = "chroma_store"

def main():
    loader = DirectoryLoader(
        path=DATA_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_documents(docs)

    embeddings = get_embedding_function()
    db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_FOLDER)
    # db.persist()
    print(f"✅ ChromaDB populated with {len(chunks)} chunks.")

if __name__ == "__main__":
    main()
