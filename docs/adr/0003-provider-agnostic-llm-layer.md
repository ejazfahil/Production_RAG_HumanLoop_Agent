# ADR-0003: Provider-Agnostic LLM Layer

**Date**: June 2026  
**Status**: Accepted

## Context

We use LLMs from multiple vendors (OpenAI, Anthropic, Mistral) and may switch between them for cost or regulatory reasons (e.g., Mistral for EU data residency).

Hardcoding OpenAI SDK calls throughout the codebase means:
- Vendor lock-in: Switching requires refactoring agent code.
- Duplication: Each node that calls the LLM imports vendor-specific logic.
- Testability: Tests can't run offline without mocking vendor APIs.

## Decision

**Define a minimal `LLM` protocol** (Python Protocol/ABC) that abstracts the `complete(system, prompt) -> Completion` contract.

Implement vendors behind this interface:
- `FakeLLM`: Deterministic offline (dev, CI, testing).
- `OpenAICompatLLM`: OpenAI, Mistral, vLLM (all use the same `/chat/completions` API).
- Stubs for Anthropic, others.

**Vendor selection via config**: `LLM_PROVIDER=openai|mistral|fake`.

## Consequences

### Positive

- **Vendor independence**: Swap `mistral` for `openai` in `.env`; agent code unchanged.
- **Testability**: `FakeLLM` runs tests offline without API keys.
- **Cost arbitrage**: Compare vendor pricing per-token; switch mid-deployment.
- **EU deployment**: Deploy with Mistral (EU co.), fulfill data-residency requirements.
- **Self-hosted**: Route to a self-hosted vLLM endpoint via `LLM_BASE_URL` config.

### Negative

- **Abstraction overhead**: Some vendor-specific features (function calling, vision) are not exposed; you lose vendor differentiation.
- **Least-common denominator**: The protocol is simple (`complete()` → text). Rich features (streaming, tool use) require protocol extension.

## Alternatives Considered

### 1. Hardcode OpenAI, use adapters
- **Why rejected**: Adapter pattern still couples nodes to vendor details; not cleaner.

### 2. Use LangChain's LLMChain (vendor abstraction built-in)
- **Why rejected**: LangChain's abstraction is heavier; we don't need its full feature set. A minimal protocol is cleaner.

### 3. Use LiteLLM (vendor-agnostic proxy)
- **Why rejected**: External dependency; we already achieve the same with a thin protocol + factory.

## Implementation Notes

### Adding a New Vendor

1. Implement a class matching the `LLM` protocol in `src/prag/llm/`.
2. Add factory logic in `src/prag/llm/factory.py`.
3. Add pricing to `src/prag/llm/base.py:PRICING`.
4. Update `src/prag/config.py:LLM_PROVIDER` enum.

Example: Anthropic

```python
class AnthropicLLM:
    def __init__(self, model: str, api_key: str) -> None:
        self.model = model
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)

    def complete(self, system: str, prompt: str, *, max_tokens: int = 512) -> Completion:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text
        usage = Usage(
            model=self.model,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
        )
        return Completion(text=text, usage=usage)
```

### Extending the Protocol

If you need streaming or tool calling:

```python
class LLM(Protocol):
    model: str
    def complete(self, system: str, prompt: str, *, max_tokens: int = 512) -> Completion: ...
    def stream(self, system: str, prompt: str) -> Iterator[str]: ...  # add this
```

Then implement in each vendor. Nodes opt-in to streaming.

## Cost Accounting

Pricing is per-1K tokens. Update `PRICING` as vendor rates change:

```python
PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.00015, 0.0006),  # (input, output) per 1K tokens
    "mistral-large-latest": (0.002, 0.006),
}
```

Actual cost per request is computed in `Usage.cost_usd` and aggregated to Prometheus.

## Testing

`FakeLLM` is used in all tests and CI. It's deterministic (no randomness) and runs offline, so tests are fast and repeatable.

For integration tests against a real vendor, set `LLM_PROVIDER` in the test env and skip them in CI (mark with `@pytest.mark.integration`).
