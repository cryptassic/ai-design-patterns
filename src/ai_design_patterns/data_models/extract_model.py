from pydantic import BaseModel, Field


class ProcessedText(BaseModel):
    summary: str = Field(..., description="A concise summary of the text.")
    semantic_tags: list[str] = Field(..., description="A list of semantic tags extracted from the text.")
    named_entities: list[str] = Field(..., description="A list of named entities identified in the text.")
    original_content: str = Field(..., description="The original text content that was processed.")
    sentiment: str = Field(..., description="The sentiment of the text (e.g., positive, negative, neutral).")