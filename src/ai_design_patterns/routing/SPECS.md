## Interesting Requirements/Specs for Building Routing

Here are **5 progressive specs** to implement custom routers. Start simple, add agents/tools/memory. Use LCEL/LangGraph. Test with LangSmith. **Goal**: Optimize for cost, speed, accuracy (e.g., cheap models for easy queries).

Each includes:
- **Description**
- **Core Components**
- **Twists** (advanced)
- **Success Metrics** (self-test)

### 1. **Sentiment-Based Customer Support Router** (LCEL Basics)
   - **Description**: Route support queries: Positive sentiment → Upsell chain (recommend products). Negative → Empathy/Support chain. Neutral → FAQ retriever.
   - **Core Components**:
     - Classifier: LLM → `Literal["positive", "negative", "neutral"]`.
     - Chains: Upsell (creative LLM), Support (RAG over docs), FAQ (vectorstore).
     - Use `RunnableBranch` + `RunnablePassthrough.assign(sentiment=classifier)`.
   - **Twists**:
     - Parallel: Run FAQ always, override with sentiment.
     - Fallback: If unsure (>0.8 confidence), route to human sim (prompt: "Escalate to human").
   - **Success Metrics**: 90% accuracy on 20 test queries (label manually). Upsell boosts "revenue score" (prompt-eval).