import os
from typing import Literal
from pydantic import BaseModel, Field

from dotenv import load_dotenv; load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnablePassthrough

# Define the language model to be used.
# Consider using a model that supports structured output for best results.
MODEL ="google/gemini-2.5-flash"

class RouteQuery(BaseModel):
    """
    Represents the output structure for classifying user query sentiment.
    Attributes:
        reasoning: Explanation for the chosen route.
        confidence: Confidence score of the classification (0.0 to 1.0).
        route: The determined sentiment route: "positive", "negative", or "neutral".
    """
    reasoning: str = Field(description="Reasoning for the chosen route.")
    confidence: float = Field(description="Confidence score of the route, from 0.0 to 1.0.")
    route: Literal["positive", "negative", "neutral"] = Field(description="Determined route for the query.")


# Initialize the ChatOpenAI language model.
# Ensure OPENAI_KEY and OPENAI_BASE_URL are set in your environment variables.
llm = ChatOpenAI(
    model=MODEL,
    api_key=os.environ["OPENAI_KEY"], 
    base_url=os.environ["OPENAI_BASE_URL"],
    temperature=0.1
)

# Prompt template for handling negative user queries, focusing on empathy and support.
support = ChatPromptTemplate.from_template(
    "Answer this negative user query with human like language and empathy, support him in solving the issue. User Query:\n\n{input}"
)

# Prompt template for upselling to users with positive queries.
upsell = ChatPromptTemplate.from_template(
    "Try upselling this user according to it's positive query. User Query:\n\n{input}"
)

# Prompt template for general FAQ-like queries.
faq = ChatPromptTemplate.from_template(
    "Answer user query according to your best knowledge: User Query: \n\n{input}"
)


# Chain for classifying the sentiment of a user's question.
# It uses the LLM with structured output to parse the response into a RouteQuery object.
sentiment_classifier = (
    ChatPromptTemplate.from_template("Classify the user provided question sentiment as neutral, positive or negative. Respond according to provided output scheme. User question: \n{input}")
    | llm.with_structured_output(RouteQuery)
)

# Chains for each sentiment route, combining a specific prompt with the LLM.
support_chain = support | llm
upsell_chain = upsell | llm
faq_chain = faq | llm

# The routing mechanism, a RunnableBranch, directs the flow based on the sentiment classification.
# It checks the 'route' field from the sentiment_classifier output and directs to the appropriate chain.
# A fallback (faq_chain) is provided for cases where no specific route matches.
router = RunnableBranch(
    (lambda x: x["route"] == "positive", upsell_chain),
    (lambda x: x["route"] == "negative", support_chain),
    (lambda x: x["route"] == "neutral", faq_chain),
    faq_chain # fallback
)

# The main chain that orchestrates the routing.
# 1. Takes the input.
# 2. Classifies the sentiment using the sentiment_classifier and assigns it to 'route'.
# 3. Passes the input and the determined 'route' to the router for final processing.
routed_chain = (
    {"input": RunnablePassthrough()} 
    | RunnablePassthrough.assign(route=sentiment_classifier)
    | router
)

# Invoke the routed chain with an example query and print the result.
result = routed_chain.invoke("This AI routing pattern is fantastic! How can I integrate it with more chains?")
print(result)
