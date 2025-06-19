# chat_memory.py

from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain

def get_custom_prompt():
    return PromptTemplate(
        input_variables=["history", "input"],
        template="{history}\nHuman: {input}\nAI:"
    )

def get_conversation_chain(llm, k=5):
    memory = ConversationBufferWindowMemory(k=k, return_messages=False)
    prompt = get_custom_prompt()
    
    return ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
        verbose=False
    )
