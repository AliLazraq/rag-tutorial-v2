# --- query_data.py ---
import argparse
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function

CHROMA_FOLDER = "chroma_store"

PROMPT_TEMPLATE = """
Vous êtes un assistant expert. Vous devez répondre à la question ci-dessous en vous basant **uniquement** sur les informations fournies dans le contexte.

Si la réponse est un nombre, renvoyez uniquement le nombre, sans unité ni explication.
Si la réponse est un booléen (oui/non), répondez **"true"** ou **"false"**, en minuscules et sans guillemets.
Si la réponse ne se trouve pas dans le contexte, répondez par : **"N/A"**.

⚠️ Vous devez répondre dans la **même langue que le contexte**, qui est le **français**.

Contexte :
{context}

---

Question : {question}
Réponse :
"""


def query_rag(query_text: str):
    embedding_function = get_embedding_function()

    print("[INFO] Searching in ChromaDB...")
    chroma_db = Chroma(persist_directory=CHROMA_FOLDER, embedding_function=embedding_function)
    chroma_results = chroma_db.similarity_search_with_score(query_text, k=5)

    top_docs = [doc for doc, _ in chroma_results]
    top_texts = [doc.page_content for doc in top_docs]

    print("[INFO] Embedding top-5 chunks and creating in-memory FAISS index...")
    top_embeddings = embedding_function.embed_documents(top_texts)

    # ✅ Fix: Combine texts and embeddings into pairs
    text_embedding_pairs = list(zip(top_texts, top_embeddings))

    faiss_db = FAISS.from_embeddings(
        text_embedding_pairs,
        embedding_function,  # ✅ this is the missing required arg
        metadatas=[doc.metadata for doc in top_docs]
    )

    print("[INFO] Performing rerank in FAISS...")
    reranked = faiss_db.similarity_search_with_score(query_text, k=1)
    if not reranked:
        print("❌ No relevant chunks found.")
        return "N/A"

    final_context = reranked[0][0].page_content
    print(f"[DEBUG] Reranked context preview:\n{final_context[:300]}...\n")

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=final_context, question=query_text)

    model = OllamaLLM(model="mistral")
    response_text = model.invoke(prompt)

    source = reranked[0][0].metadata.get("source", "N/A")
    print(f"✅ Final Answer: {response_text}")
    print(f"📎 Source: {source}")
    return response_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_rag(args.query_text)


if __name__ == "__main__":
    main()
