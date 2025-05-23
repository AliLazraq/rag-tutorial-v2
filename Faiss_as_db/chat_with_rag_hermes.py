from query_data_hermes2 import query_rag

print("🧠 Chat with your documents (type 'exit' to quit)")
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break
    response = query_rag(query)
    print("RAG:", response)
