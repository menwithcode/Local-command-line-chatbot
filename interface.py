# interface.py

from model_loader import load_llm
from chat_memory import get_conversation_chain

def chat():
    print("🧠 Chat started. Type '/exit' to stop.\n")

    llm = load_llm()
    conversation = get_conversation_chain(llm)

    while True:
        user_input = input("🧑 You: ")
        if user_input.lower().strip() == "/exit":
            print("🧠 Bot: Conversation ended.")
            break

        full_output = conversation.run(user_input)

        if "AI:" in full_output:
            response = full_output.split("AI:")[-1].strip()
        else:
            response = full_output.strip()

        print("🤖 Bot:", response)

if __name__ == "__main__":
    chat()
