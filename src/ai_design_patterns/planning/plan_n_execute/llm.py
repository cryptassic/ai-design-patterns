from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="granite4:7b-a1b-h",
    # model = "granite4:3b",
    temperature=0.2
)