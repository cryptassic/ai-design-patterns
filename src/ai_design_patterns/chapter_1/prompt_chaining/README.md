# Prompt Chaining Pattern

**Chapter 1** from a guide on LLM agentic patterns by **Antonio Gulli**.

[![Prompt Chaining Visual](placeholder-for-fig2.png)](#)  
*Fig. 2: Prompt Chaining - Output of one prompt feeds into the next for complex tasks.*

## Overview

Prompt chaining (aka Pipeline pattern) breaks complex tasks into sequential, manageable sub-tasks. Each step uses a focused LLM prompt, with outputs feeding into the next. This improves reliability, reduces errors (e.g., hallucination, context drift), enables tool integration, and supports agentic workflows.

**Why use it?** Single prompts fail on multifaceted tasks due to high cognitive load. Chaining adds modularity, debugging ease, and structured outputs (e.g., JSON).

**Rule of thumb:** Apply for multi-step reasoning, tool use, or tasks too complex for one prompt.

## Key Benefits
- **Sequential decomposition:** Focused prompts minimize errors.
- **Structured outputs:** Use JSON/XML for reliable handoffs.
- **Roles per step:** Assign roles (e.g., "Market Analyst") for precision.
- **Tool integration:** Fetch external data/APIs between steps.
- **Frameworks:** LangChain, LangGraph, CrewAI, Google ADK.

## Limitations of Single Prompts
- Instruction neglect
- Contextual drift
- Error propagation
- Long context limits
- Increased hallucinations

**Example Chain (Market Report):**
1. Summarize report → JSON summary.
2. Extract trends + data → Structured JSON.
3. Draft email from trends.

## Practical Use Cases
1. **Info Processing:** Extract → Summarize → Entities → Query DB → Report.
2. **Complex QA:** Decompose query → Research sub-parts → Synthesize.
3. **Data Extraction:** Extract fields → Validate → Refine (iterative).
4. **Content Generation:** Ideas → Outline → Draft sections → Refine.
5. **Conversational Agents:** Intent extraction → State update → Response.
6. **Code Gen:** Pseudocode → Draft → Debug → Document.
7. **Multimodal:** Text extract → Link labels → Interpret table.

Supports parallel (e.g., multi-doc extraction) + sequential (synthesis).

## Context Engineering
Build rich context layers:
- System prompt (role/tone).
- Retrieved docs/tools.
- Implicit data (history/state).
Evolves prompt engineering for agentic performance. Use optimizers like Vertex AI.

## Hands-On Example (LangChain)

Install: `pip install langchain langchain-community langchain-openai langgraph`

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(temperature=0)  # Set OPENAI_API_KEY

# Prompt 1: Extract specs
prompt_extract = ChatPromptTemplate.from_template(
    "Extract the technical specifications from the following text:\n\n{text_input}"
)
extraction_chain = prompt_extract | llm | StrOutputParser()

# Prompt 2: To JSON
prompt_transform = ChatPromptTemplate.from_template(
    "Transform the following specifications into a JSON object with 'cpu', 'memory', and 'storage' as keys:\n\n{specifications}"
)

# Full chain
full_chain = (
    {"specifications": extraction_chain}
    | prompt_transform | llm | StrOutputParser()
)

# Run
input_text = "The new laptop model features a 3.5 GHz octa-core processor, 16GB of RAM, and a 1TB NVMe SSD."
result = full_chain.invoke({"text_input": input_text})
print(result)
# Output: {"cpu": "3.5 GHz octa-core processor", "memory": "16GB of RAM", "storage": "1TB NVMe SSD"}
```

## Key Takeaways
- Breaks tasks into focused LLM calls.
- Output → Input dependency chain.
- Boosts reliability for agents.
- Use frameworks for execution.

## References
- [LangChain LCEL](https://python.langchain.com/v0.2/docs/core_modules/expression_language/)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [Prompting Guide: Chaining](https://www.promptingguide.ai/techniques/chaining)
- [OpenAI Prompting](https://platform.openai.com/docs/guides/gpt/prompting)
- [CrewAI](https://docs.crewai.com/)
- [Google Prompt Engineering](https://cloud.google.com/discover/what-is-prompt-engineering?hl=en)
- [Vertex Prompt Optimizer](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/prompt-optimizer)

**Author:** Antonio Gulli  
**License:** MIT (assumed for demo purposes).