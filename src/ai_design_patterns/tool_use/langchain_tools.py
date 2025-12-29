import os
from dotenv import load_dotenv; load_dotenv()

from langchain.agents import create_agent
from langchain_openai.chat_models import ChatOpenAI

from ai_design_patterns.tool_use.tools import factorio, add_numbers, power_calc

MODEL="nvidia/nemotron-3-nano-30b-a3b"

all_tools = [add_numbers, factorio, power_calc]

llm = ChatOpenAI(
    model=MODEL,
    api_key=os.environ["OPENAI_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
).bind_tools(all_tools, tool_choice="required")

agent = create_agent(
    model=llm,
    tools=all_tools,
    system_prompt="You are math assistant. Use tools for calculations",
)

queries = [
    "2^10 + 3^2?",
    "What's 5!?",
    "Factorial of -1?",  # Error handling
    "How many moons does Mars have? Then factorial of that.",
]
for q in queries:
    print(f"Q: {q}")
    result = agent.invoke({"messages": [("human", q)]})

    last_msg = result['messages'][-1]
    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
        print("Tool called, but not executed.")
    else:
        print(f"A: {last_msg.content}")
