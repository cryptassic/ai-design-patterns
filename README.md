# AI Design Patterns

This project serves as a **personal dojo platform** for implementing AI design patterns, striving for **high-quality examples** drawn from **Antonio Gulli**'s guide on LLM agentic patterns.

## Implemented Patterns

- **[Prompt Chaining](src/ai_design_patterns/prompt_chaining/)**
  - [LangChain](src/ai_design_patterns/prompt_chaining/langchain_prompt_chaining.py)
  - [Haystack](src/ai_design_patterns/prompt_chaining/haystack_prompt_chaining.py)
- **[Parallel Processing](src/ai_design_patterns/parallel/)**
  - [LangChain](src/ai_design_patterns/parallel/langchain_parallel.py)
- **[Routing](src/ai_design_patterns/routing/)**
  - [LCEL LangChain](src/ai_design_patterns/routing/LCEL_langchain_routing.py)
- **[Reflection](src/ai_design_patterns/reflection/)**
  - [LangChain](src/ai_design_patterns/reflection/langchain_reflection.py)
- **[Tool Use](src/ai_design_patterns/tool_use/)**
  - [LangChain](src/ai_design_patterns/tool_use/langchain_tools.py)

## Setup

1. Install dependencies:
   ```bash
   uv sync
   ```
2. Copy `example.env` to `.env` and configure as needed.
