"""
llm_provider.py — Step 3: Swappable LLM backend interface.

Defines an abstract LLMProvider and implements AnthropicProvider.
The get_provider() factory returns None if no API key is set,
signaling the caller to use the template fallback.
"""

import os
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract interface for LLM generation backends.
    
    Subclass this to add new providers (OpenAI, local models, etc.).
    """
    
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response given system and user prompts.
        
        Args:
            system_prompt: System-level instructions (safety constraints,
                           audience, retrieved context).
            user_prompt: The specific request with payload data.
        
        Returns:
            Generated text string.
        """
        ...


class AnthropicProvider(LLMProvider):
    """LLM provider using the Anthropic Claude API.
    
    Reads the API key from the ANTHROPIC_API_KEY environment variable.
    """
    
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
    
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return message.content[0].text


class OpenAIProvider(LLMProvider):
    """Stub for future OpenAI-based provider.
    
    To implement:
        1. pip install openai
        2. Set OPENAI_API_KEY environment variable
        3. Fill in the generate() method using the OpenAI chat completions API
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        raise NotImplementedError(
            "OpenAIProvider is a stub. Implement it by filling in the "
            "generate() method with OpenAI API calls."
        )
    
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class LocalModelProvider(LLMProvider):
    """Stub for future local/self-hosted model provider.
    
    To implement:
        1. Load a local model (e.g., via transformers, llama.cpp, or Ollama)
        2. Fill in the generate() method
    """
    
    def __init__(self, model_path: str):
        raise NotImplementedError(
            "LocalModelProvider is a stub. Implement it by loading a "
            "local model and filling in the generate() method."
        )
    
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


def get_provider() -> LLMProvider | None:
    """Factory function returning the best available LLM provider.
    
    Returns:
        AnthropicProvider if ANTHROPIC_API_KEY is set, else None.
        None signals the caller to use the deterministic template fallback.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            return AnthropicProvider(api_key=api_key)
        except Exception:
            # If anthropic package is not installed or key is invalid,
            # fall back gracefully
            return None
    return None
