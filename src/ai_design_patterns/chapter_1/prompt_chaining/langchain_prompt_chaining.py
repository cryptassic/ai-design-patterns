import os
from dotenv import load_dotenv; load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from ai_design_patterns.data_models.product import Product

# Define the language model to use (OpenRouter)
MODEL = "x-ai/grok-4.1-fast"

# Initialize the ChatOpenAI model with API key and base URL from environment variables
llm = ChatOpenAI(
    model=MODEL,
    api_key=os.environ["OPENAI_KEY"], 
    base_url=os.environ["OPENAI_BASE_URL"]
)

# Define a prompt template for extracting information
extractor_template = ChatPromptTemplate(
    [
        ("system", "You are a precise information extractor. Your only task is to analyze the given text and output only a valid JSON object."),
        ("human", "Extract the technical specifications from the following text:\n\n{text}")
    ]
)

# Define a prompt template for enriching information
enricher_template = ChatPromptTemplate(
    [
        ("system", "You are a information enricher. Your only task is to analyze the given text and update missing fields, then output only a valid JSON object."),
        ("ai", "Data to enrich:\n\n{json_data}")
    ]
)

# Configure the language model to output structured data based on the Product data model
structured_llm = llm.with_structured_output(Product)

# Create a chain for extraction: extractor_template -> structured_llm
extract_chain = extractor_template | structured_llm

# Create a full chain: extract_chain -> enricher_template -> structured_llm
full_chain = ( {"json_data": extract_chain} | enricher_template | llm.with_structured_output(Product) )

# Invoke the full chain with an example text and print the result
result = full_chain.invoke({"text":{"The new iPhone 15 Pro Max features a titanium design, the A17 Pro chip, and an advanced camera system with a 5x optical zoom telephoto lens. It comes with 256GB, 512GB, or 1TB of storage."}})
