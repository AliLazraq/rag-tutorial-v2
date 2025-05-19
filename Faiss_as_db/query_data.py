import argparse
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from get_embedding_function import get_embedding_function

FAISS_FOLDER = "faiss_store"

PROMPT_TEMPLATE = """
You are an expert assistant. You must answer the question below using **only** the information in the context. 

If the answer is a number, return the number **only**, with no units or explanation.

If the answer is a boolean (yes/no), return either **"true"** or **"false"**, exactly and in lowercase.

If the answer is not found in the context, respond with: **"N/A"**.

Context:
{context}

---

Question: {question}
Answer:
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    embedding_function = get_embedding_function()

    # Load FAISS index
    print("📂 Loading FAISS DB...")
    db = FAISS.load_local(FAISS_FOLDER, embedding_function, allow_dangerous_deserialization=True)


    # Perform similarity search
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = OllamaLLM(model="mistral")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _ in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()
