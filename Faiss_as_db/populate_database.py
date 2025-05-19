import argparse
import os
import shutil
import pickle
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain.vectorstores import FAISS

DATA_PATH = "../data"
FAISS_PATH = "faiss_store"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the FAISS database.")
    args = parser.parse_args()

    if args.reset:
        print("✨ Clearing FAISS Database")
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_faiss(chunks)


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def calculate_chunk_ids(chunks: list[Document]):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        chunk.metadata["id"] = chunk_id
        last_page_id = current_page_id

    return chunks


from langchain_community.vectorstores import FAISS

FAISS_FOLDER = "faiss_store"  # a folder, not a file

def add_to_faiss(chunks: list[Document]):
    embedding_function = get_embedding_function()
    chunks_with_ids = calculate_chunk_ids(chunks)

    if os.path.exists(FAISS_FOLDER):
        print("📂 Loading existing FAISS index...")
        db = FAISS.load_local(FAISS_FOLDER, embedding_function)

        existing_docs = db.similarity_search("", k=len(chunks_with_ids))
        existing_ids = set(doc.metadata.get("id") for doc in existing_docs if "id" in doc.metadata)

        print(f"Number of existing documents in FAISS DB: {len(existing_ids)}")

        new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]
        if new_chunks:
            print(f"👉 Adding new documents: {len(new_chunks)}")
            db.add_documents(new_chunks)
            db.save_local(FAISS_FOLDER)
        else:
            print("✅ No new documents to add")
    else:
        print("✨ Creating new FAISS index...")
        db = FAISS.from_documents(chunks_with_ids, embedding_function)
        db.save_local(FAISS_FOLDER)
        print("✅ FAISS DB created and saved to disk")


def clear_database():
    if os.path.exists(FAISS_FOLDER):
        shutil.rmtree(FAISS_FOLDER)
        print("🗑️ Removed FAISS store folder")

if __name__ == "__main__":
    main()
