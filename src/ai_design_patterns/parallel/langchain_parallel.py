import os
import asyncio

from typing import Any, Dict
from dotenv import load_dotenv; load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader

from ai_design_patterns.data_models.extract_model import ProcessedText

MODEL = "x-ai/grok-4.1-fast"

# Initialize the ChatOpenAI model (via OpenRouter) with API key and base URL from environment variables
llm = ChatOpenAI(
    model=MODEL,
    api_key=os.environ["OPENAI_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"]
)


def load_pdf(target_file: Dict[str, Any]) -> str:
    """
    Load content from a PDF file.

    Args:
        target_file: Dictionary containing 'file_path' key with the path to the PDF.

    Returns:
        The text content of the first page of the PDF, or empty string if loading fails.
    """
    doc = None
    if "file_path" in target_file:
        file_path = target_file["file_path"]
        try:
            loader = PyPDFLoader(file_path)
            doc = loader.load()
        except Exception:
            pass  # Silently fail and return empty string

    return doc[0].page_content if doc else ""


async def main(file_path: str) -> ProcessedText:
    """
    Process a PDF file using parallel LangChain runnables.

    1. Loads PDF content.
    2. Runs four parallel chains: summary, semantic tags, sentiment, named entities.
    3. Synthesizes results into a ProcessedText object using structured output.

    Args:
        file_path: Path to the PDF file.

    Returns:
        ProcessedText object with extracted and synthesized information.
    """
    # Runnable to load PDF content
    pdf_loader = RunnableLambda(load_pdf)

    # Summary chain: concise paragraph summary
    summarize_chain: Runnable = (
        ChatPromptTemplate.from_messages([
            ("system", "Summarize the given text into a concise paragraph."),
            ("user", "{content}")
        ])
        | llm
        | StrOutputParser()
    )

    # Semantic tags chain: extract 5 tags
    semantic_chain: Runnable = (
        ChatPromptTemplate.from_messages([
            ("system", "Extract 5 semantic tags from the given text, separated by commas."),
            ("user", "{content}")
        ])
        | llm
        | StrOutputParser()
    )

    # Sentiment chain: score from -1 to 1
    sentiment_chain: Runnable = (
        ChatPromptTemplate.from_messages([
            ("system", "Evaluate the sentiment of the given text and return a score from -1.0 (very negative) to 1.0 (very positive)."),
            ("user", "{content}")
        ])
        | llm
        | StrOutputParser()
    )

    # Named entities chain: org/location/person
    named_entities_chain: Runnable = (
        ChatPromptTemplate.from_messages([
            ("system", "Extract named entities (organization, location, person) from the given text. List them separated by commas."),
            ("user", "{content}")
        ])
        | llm
        | StrOutputParser()
    )

    # Synthesis prompt using parallel outputs
    synthesis_prompt = ChatPromptTemplate.from_messages([
        ("system", """Synthesize the following into a JSON object matching the ProcessedText schema:

Summary: {summary}
Semantic Tags: {semantic}
Sentiment: {sentiment}
Named Entities: {named_entities}"""),
        ("user", "Original text: {content}")
    ])

    # Parallel processing chain
    parallel_chain = (
        {"file_path": RunnablePassthrough()}
        | RunnablePassthrough.assign(content=pdf_loader)
        | RunnableParallel({
            "content": lambda x: x["content"],
            "summary": summarize_chain,
            "semantic": semantic_chain,
            "sentiment": sentiment_chain,
            "named_entities": named_entities_chain,
        })
    )

    # Full chain: parallel -> synthesis -> structured output
    full_chain = parallel_chain | synthesis_prompt | llm.with_structured_output(ProcessedText)

    # Uncomment to visualize the chain as Mermaid diagram
    # graph = full_chain.get_graph().draw_mermaid()
    # print(graph)

    return await full_chain.ainvoke(file_path)


if __name__ == "__main__":
    file_path = "src/ai_design_patterns/parallel/sample.pdf"
    result = asyncio.run(main(file_path))
    print(result)
