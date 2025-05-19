from query_data import query_rag
from langchain_ollama import OllamaLLM

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""


def test_monopoly_rules():
    assert query_and_validate(
        question="How much total money does a player start with in Monopoly? (Answer with the number only)",
        expected_response="$1500",
    )


def test_ticket_to_ride_rules():
    assert query_and_validate(
        question="How many points does the longest continuous train get in Ticket to Ride? (Answer with the number only)",
        expected_response="10 points",
    )


def query_and_validate(question: str, expected_response: str):
    print(f"\n🧪 Question: {question}")
    
    # Step 1: Query the RAG system
    response_text = query_rag(question)

    # Step 2: Prepare the evaluation prompt
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response,
        actual_response=response_text.strip()
    )

    # Step 3: Use the model to validate
    model = OllamaLLM(model="mistral")
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    # Step 4: Display evaluation
    print("\n📊 Evaluation Prompt:")
    print(prompt)
    
    if "true" in evaluation_results_str_cleaned:
        print("\033[92m" + f"✅ Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        print("\033[91m" + f"❌ Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            "Invalid evaluation result. Expected 'true' or 'false', got: "
            + evaluation_results_str_cleaned
        )


if __name__ == "__main__":
    print("🔍 Running RAG evaluation tests...\n")
    results = []
    results.append(("Monopoly", test_monopoly_rules()))
    results.append(("Ticket to Ride", test_ticket_to_ride_rules()))
    print("\n✅ Summary:")
    for name, result in results:
        print(f"{name}: {'✅ Passed' if result else '❌ Failed'}")

