import asyncio
from langchain.agents import create_agent
from langchain_tavily import TavilySearch

from langgraph.graph import START, END, StateGraph

from ai_design_patterns.planning.plan_n_execute.state import State
from ai_design_patterns.planning.plan_n_execute.llm import llm
from ai_design_patterns.planning.plan_n_execute.planner import planner, replanner, Response

tools = [TavilySearch(max_results=5)]

prompt = "You are helpful assistant"
agent_executor = create_agent(system_prompt=prompt, model=llm, tools=tools)


async def execute_step(state: State):
    plan = state["plan"]
    plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
    task = plan[0] if plan else ""
    task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    agent_response = await agent_executor.ainvoke(
        {"messages": [("user", task_formatted)]}
    )
    return {
        "past_steps": [(task, agent_response["messages"][-1].content)],
    }

async def plan_step(state: State):
    plan = await planner.ainvoke({"messages": [("user", state["input"])]})
    return {"plan": plan.steps}


async def replan_step(state: State):
    output = await replanner.ainvoke(state)
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {"plan": output.action.steps}


def should_end(state: State):
    if "response" in state and state["response"]:
        return END
    else:
        return "agent"



async def main():
    workflow = StateGraph(State)

    workflow.add_node("planner", plan_step)
    workflow.add_node("agent", execute_step)
    workflow.add_node("replan", replan_step)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "agent")
    workflow.add_edge("agent", "replan")

    workflow.add_conditional_edges(
        "replan",
        # Next, we pass in the function that will determine which node is called next.
        should_end,
        ["agent", END],
    )

    app = workflow.compile()

    config = {"recursion_limit": 50}

    inputs = {"input": "what is the hometown of the mens 2024 Australia open winner?"}

    async for event in app.astream(inputs, config=config):
        for k,v in event.items():
            if k != "__end__":
                print(v)


if __name__ == "__main__":
    asyncio.run(main())