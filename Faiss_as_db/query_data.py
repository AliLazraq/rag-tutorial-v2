# --- query_data.py ---
import argparse
import os
import shutil
from langchain_community.vectorstores import Chroma, FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function

CHROMA_FOLDER = "chroma_store"
FAISS_FOLDER = "faiss_store_confirmed"

PROMPT_TEMPLATE = """
You are an expert assistant. You must answer the question below using **only** the information in the context. 

If the answer is a number, return the number **only**, with no units or explanation.
If the answer is a boolean (yes/no), return either **\"true\"** or **\"false\"**, exactly and in lowercase.
If the answer is not found in the context, respond with: **\"N/A\"**.

Context:
{context}

---

Question: {question}
Answer:
"""

def query_rag(query_text: str):
    embedding_function = get_embedding_function()

    print("[INFO] Searching in ChromaDB...")
    chroma_db = Chroma(persist_directory=CHROMA_FOLDER, embedding_function=embedding_function)
    chroma_results = chroma_db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in chroma_results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = OllamaLLM(model="mistral")
    response_text = model.invoke(prompt)

    print("[INFO] Storing top-5 chunks into FAISS...")
    top_docs = [doc for doc, _ in chroma_results]

    if os.path.exists(FAISS_FOLDER):
        shutil.rmtree(FAISS_FOLDER)
    faiss_db = FAISS.from_documents(top_docs, embedding_function)
    faiss_db.save_local(FAISS_FOLDER)
    print(f"✅ Stored {len(top_docs)} chunks to FAISS.")

    sources = []
    for doc in top_docs:
        doc_id = doc.metadata.get("id") or doc.metadata.get("source") or "N/A"
        sources.append({"id": doc_id})

    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_rag(args.query_text)


if __name__ == "__main__":
    main()
