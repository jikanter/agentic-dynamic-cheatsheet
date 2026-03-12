"""
Text Generation: A unified Python interface for OpenAI, Claude, and Gemini APIs.

This package provides a simple, unified client for working with multiple LLM providers
including OpenAI, Claude, and Google Gemini, with built-in retry logic and web search integration.

Example:
    >>> from text_generation import UnifiedLLMClient
    >>> client = UnifiedLLMClient(provider="openai", model="gpt-4o")
    >>> response = client.generate("Hello, world!")
    >>> print(response)

    >>> # Using Gemini
    >>> client = UnifiedLLMClient(provider="gemini", model="gemini-2.5-flash")
    >>> response = client.generate("Explain quantum computing")
"""

__version__ = "0.1.0"
__author__ = "Mirac Suzgun"
__license__ = "MIT"

from text_generation.simple_unified_client import UnifiedLLMClient, Provider

__all__ = [
    "UnifiedLLMClient",
    "Provider",
]
