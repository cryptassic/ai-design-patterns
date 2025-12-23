import os
import json
from dotenv import load_dotenv; load_dotenv()

from haystack import Pipeline
from haystack.utils import Secret
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders import PromptBuilder
from haystack.components.converters import OutputAdapter

from ai_design_patterns.data_models.product import Product

MODEL = "x-ai/grok-4.1-fast"

prompt_template = """
    Given the following text, extract the product information and return expected JSON object.
    
    Text: {{text}}
    

    Return JSON parsable object.

    JSON Output Schema: {{output_schema}}
    """

prompt_builder = PromptBuilder(template=prompt_template, required_variables=["text"])

llm = OpenAIGenerator(
    api_key=Secret.from_env_var("OPENAI_KEY"),
    api_base_url=os.environ["OPENAI_BASE_URL"],
    model=MODEL,
    generation_kwargs={"temperature": 0.2, "response_format": {"type": "json_schema", "json_schema": Product.model_json_schema()}},
)

pipeline = Pipeline()
pipeline.add_component("prompt_builder", prompt_builder)
pipeline.add_component("llm", llm)
pipeline.add_component("adapter", OutputAdapter(template="{{ replies[0] }}", output_type=dict))
pipeline.connect("prompt_builder.prompt", "llm.prompt")
pipeline.connect("llm.replies", "adapter.replies")


result = pipeline.run(
    {
        "prompt_builder": {
            "output_schema": Product.model_json_schema(),
            "text": "The new iPhone 15 Pro Max features a titanium design, the A17 Pro chip, and an advanced camera system with a 5x optical zoom telephoto lens. It comes with 256GB, 512GB, or 1TB of storage."
        }
    }
)
