"""
This module implements a reflection agent using LangChain to iteratively generate and refine startup pitches.

The agent consists of three main components:
1. A pitch generation chain (`pitch_gen_chain`) that creates startup pitches based on user input and feedback.
2. A pitch evaluation chain (`pitch_eval_chain`) that critiques generated pitches and provides structured feedback.
3. A pitch revision chain (`pitch_revisor`) that summarizes pitch history for reflection.

The `run_reflection_agent` function orchestrates these components to generate the best possible pitch within a given number of iterations.
"""

import os
from typing import Literal
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

# Define the model to be used for pitch generation
MODEL = "nvidia/nemotron-3-nano-30b-a3b"

class Reflection(BaseModel):
    """
    Represents the structured feedback from the pitch evaluation chain.

    Attributes:
        score (float): A score from 1 to 10 indicating the quality of the pitch.
        critique (str): A detailed critique of the pitch.
        continue_ (Literal["yes", "no"]): Indicates whether the agent should continue refining the pitch.
        suggestions (str): Suggestions for improving the next iteration of the pitch.
    """
    score: float = Field(ge=1, le=10, description="Score from 1-10 for the pitch quality")
    critique: str = Field(description="Detailed critique of the pitch")
    continue_: Literal["yes", "no"] = Field(description="Whether to continue refining the pitch")
    suggestions: str = Field(description="Suggestions for improving the pitch")


# Initialize the language model for pitch generation
llm = ChatOpenAI(
    model=MODEL,
    api_key=os.environ["OPENAI_KEY"], 
    base_url=os.environ["OPENAI_BASE_URL"],
    temperature=0.8
)

# Initialize a separate language model for pitch evaluation (can be a different model with lower temperature for more objective evaluation)
llm_eval = ChatOpenAI(
    model="x-ai/grok-4.1-fast",
    api_key=os.environ["OPENAI_KEY"], 
    base_url=os.environ["OPENAI_BASE_URL"],
    temperature=0.5
)


# Define the pitch generation chain
pitch_gen_chain = (
    ChatPromptTemplate.from_template(
        """You are a creative startup pitch maker. Your task is to generate a compelling 150-200 word pitch from user input.
Structure: 1. Hook, 2. Problem, 3. Solution/Product, 4. Market/Traction, 5. CTA/Ask.

User input: {{user_input}}

{% if feedback %}
Summary from previous pitch generations and judge input, use it for reflection, but keep more importance on latest feedback.
{{memory}}

Feedback from critic for most recent pitch: {{feedback}}
{% endif %}
""",
        template_format="jinja2"
    )
    | llm
    | StrOutputParser()
)

# Define the pitch evaluation chain
pitch_eval_chain = (
    ChatPromptTemplate.from_template(
        """You are a startup pitch critic and evaluator. Your task is to review a user pitch and evaluate it on key criteria:
criteria: hook, clarity, market fit, CTA, conciseness. Score 1-10.
Additionally, allow the user to use their full attempts in generating the perfect pitch.


Current Attempt: {c_iter}/{max_iters}
User Pitch: {pitch}

Output ONLY JSON
"""
    )
    | llm_eval.with_structured_output(Reflection)
)


# Define the pitch revision chain for summarizing feedback and previous pitches
pitch_revisor = (
    ChatPromptTemplate.from_template(
        """You summarize pitch history as concise memory for a reflection agent.

Keep ONLY: core idea, hook/problem/solution/market/CTA + key feedback (strengths/fixes).
Output EXACTLY a numbered list:

1. Pitch: [1-line summary]. Feedback: [bullets or short].

Current Memory: {memory}
Latest Pitch: {new_pitch}
Latest Feedback: {feedback}
"""
    )
    | llm
    | StrOutputParser()
)


def run_reflection_agent(idea: str, max_iters: int = 5) -> str:
    """
    Runs the reflection agent to iteratively generate and refine a startup pitch.

    Args:
        idea (str): The initial business idea for which to generate a pitch.
        max_iters (int): The maximum number of iterations to refine the pitch.

    Returns:
        str: The best generated pitch based on the evaluation scores.
    """
    feedback = None
    current_pitch = None
    current_memory_ctx = None

    all_pitches = []

    for itr in range(0, max_iters):
        # Generate a new pitch
        current_pitch = pitch_gen_chain.invoke({"user_input": idea, "feedback": feedback, "memory": current_memory_ctx})

        # Evaluate the generated pitch
        feedback = pitch_eval_chain.invoke({"pitch": current_pitch, "c_iter": itr + 1, "max_iters": max_iters})

        # Store the pitch and its score
        all_pitches.append({"pitch": current_pitch, "score": feedback.score})

        # If the evaluator says "no" to continuing, break the loop
        if feedback.continue_ == "no":
            break
        
        # Revise the memory context for the next iteration
        current_memory_ctx = pitch_revisor.invoke({"memory": current_memory_ctx, "new_pitch": current_pitch, "feedback": feedback})

    # Sort all generated pitches by score in descending order and return the best one
    best_pitch = sorted(all_pitches, key=lambda pitch: pitch['score'], reverse=True)

    return best_pitch[0]['pitch']

if __name__ == "__main__":
    # Example usage of the reflection agent
    final_pitch = run_reflection_agent("Business idea for dating and effective matchmaking, tailored towards male pain point: lack of courage to pickup in real life")
    print("Final Best Pitch:")
    print(final_pitch)
