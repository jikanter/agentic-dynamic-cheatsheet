"""
Simple Unified Client for OpenAI, Claude, Gemini, and xAI (Grok) APIs

A straightforward, easy-to-use interface for working with OpenAI, Claude, Google Gemini, and xAI (Grok) models.

Features:
- Single class interface for multiple LLM providers
- Built-in web search integration via Tavily
- Native web search via provider's built-in tools:
  - OpenAI: Automatic web search with search-preview models (e.g., gpt-4o-mini-search-preview)
  - Claude: web_search_20250305 tool via Chat Completions API
  - Gemini: Google Search grounding via tools
  - xAI (Grok): web_search tool via Responses API
- Streaming support
- Conversation history management
- Automatic retry logic with 10-second wait between attempts
- Easy parameter customization (temperature, max_completion_tokens, etc.)
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional, Union, Iterator
from enum import Enum

# Provider SDKs — imported lazily so only the SDK for the chosen provider is required.
try:
    import openai as _openai_module
    OPENAI_AVAILABLE = True
except ImportError:
    _openai_module = None
    OPENAI_AVAILABLE = False

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    Anthropic = None
    ANTHROPIC_AVAILABLE = False

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TavilyClient = None
    TAVILY_AVAILABLE = False

try:
    from google import genai
    from google.genai import types as genai_types
    from google.genai import errors as genai_errors
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None
    genai_types = None
    genai_errors = None


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Provider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    CLAUDE = "claude"
    GEMINI = "gemini"
    XAI = "xai"  # Grok models hosted by xAI


class UnifiedLLMClient:
    """
    A unified interface for interacting with OpenAI, Claude, Gemini, and xAI (Grok) models.

    This client provides automatic retry logic, web search integration,
    and seamless conversation management across multiple LLM providers.

    Example:
        >>> from text_generation import UnifiedLLMClient
        >>> client = UnifiedLLMClient(provider="openai", model="gpt-4o")
        >>> response = client.generate("Hello, how are you?")
        >>> print(response)

        >>> # With Gemini
        >>> client = UnifiedLLMClient(provider="gemini", model="gemini-2.5-flash")
        >>> response = client.generate("Explain quantum computing")

        >>> # With custom parameters
        >>> response = client.generate(
        ...     "Explain AI",
        ...     temperature=0.9,
        ...     max_completion_tokens=500
        ... )
    """

    def __init__(
        self,
        provider: Union[str, Provider],
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        web_search_enabled: bool = False,
        tavily_api_key: Optional[str] = None,
        default_web_search: bool = False,
        web_search_config: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
        max_completion_tokens: Optional[int] = None,
        max_retries: int = 3,
        retry_delay: int = 10,
    ):
        """
        Initialize the unified LLM client.

        Args:
            provider: The LLM provider ("openai", "claude", "gemini", or "xai")
            model: Model name (e.g., "gpt-4o", "claude-sonnet-4-5-20250514", "gemini-2.5-flash", "grok-4-1")
            api_key: API key for the provider (defaults to env vars)
            base_url: Custom base URL for OpenAI-compatible APIs (e.g., Together AI, Ollama, DeepSeek)
            web_search_enabled: Enable web search capabilities via Tavily
            tavily_api_key: Tavily API key for web search
            default_web_search: Enable native web search via provider's built-in tool
            web_search_config: Configuration for default web search tool
                For OpenAI: {"search_context_size": "medium", "user_location": {...}}
                For Claude: {"max_uses": 5, "allowed_domains": [...], "blocked_domains": [...], "user_location": {...}}
                For Gemini: {"dynamic_retrieval_threshold": 0.3} (optional threshold for grounding)
            temperature: Default sampling temperature (0.0 to 2.0)
            max_completion_tokens: Default maximum tokens in response
            max_retries: Maximum number of retry attempts (default: 3)
            retry_delay: Delay in seconds between retries (default: 10)
        """
        normalized_provider = provider.lower() if isinstance(provider, str) else provider
        # Accept "grok" as an alias for xAI
        if isinstance(normalized_provider, str) and normalized_provider == "grok":
            normalized_provider = Provider.XAI.value
        self.provider = Provider(normalized_provider)
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.base_url = base_url
        self.web_search_enabled = web_search_enabled
        self.default_web_search = default_web_search
        self.web_search_config = web_search_config or {}
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Set default models
        if model is None:
            model = self._get_default_model()
        self.model = model

        # Initialize provider clients
        self._init_provider_client(api_key)

        # Initialize web search if enabled
        self.tavily_client = None
        if web_search_enabled:
            if not TAVILY_AVAILABLE:
                logger.warning("Tavily SDK not installed. Web search will be disabled. Install with: pip install tavily-python")
            else:
                tavily_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
                if tavily_key:
                    self.tavily_client = TavilyClient(api_key=tavily_key)
                else:
                    logger.warning("Tavily API key not provided. Web search will be disabled.")

        # Conversation history
        self.messages: List[Dict[str, str]] = []

    def _get_default_model(self) -> str:
        """Get the default model for the provider."""
        defaults = {
            Provider.OPENAI: "gpt-4o",
            Provider.CLAUDE: "claude-sonnet-4-5-20250514",
            Provider.GEMINI: "gemini-2.5-flash",
            Provider.XAI: "grok-4-1",
        }
        return defaults[self.provider]

    def _init_provider_client(self, api_key: Optional[str]) -> None:
        """Initialize the appropriate provider client."""
        if self.provider == Provider.OPENAI:
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI SDK not installed. Install it with: pip install openai")
            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key and not self.base_url:
                raise ValueError(
                    "OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter."
                )
            client_kwargs = {}
            if key:
                client_kwargs["api_key"] = key
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self.openai_client = _openai_module.OpenAI(**client_kwargs)

        elif self.provider == Provider.CLAUDE:
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic SDK not installed. Install it with: pip install anthropic")
            key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError(
                    "Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter."
                )
            self.anthropic_client = Anthropic(api_key=key)

        elif self.provider == Provider.GEMINI:
            if not GEMINI_AVAILABLE:
                raise ImportError(
                    "Google Gemini SDK not installed. Install it with: pip install google-genai"
                )
            key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not key:
                raise ValueError(
                    "Gemini API key is required. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable or pass api_key parameter."
                )
            self.gemini_client = genai.Client(api_key=key)

        elif self.provider == Provider.XAI:
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI SDK not installed (used for xAI). Install it with: pip install openai")
            key = api_key or os.getenv("XAI_API_KEY")
            if not key:
                raise ValueError(
                    "xAI API key is required. Set XAI_API_KEY environment variable or pass api_key parameter."
                )
            self.xai_client = _openai_module.OpenAI(api_key=key, base_url="https://api.x.ai/v1")

    def _get_openai_web_search_tool(self) -> Dict[str, Any]:
        """
        Build OpenAI web search tool configuration.
        
        Note: This is for the Responses API. For Chat Completions API,
        search-preview models (e.g., gpt-4o-mini-search-preview) have automatic
        web search and don't require tools.

        Returns:
            OpenAI web search tool dictionary
        """
        tool = {
            "type": "web_search_preview",
            "web_search_preview": {}
        }
        
        # Add search_context_size if provided
        if "search_context_size" in self.web_search_config:
            tool["web_search_preview"]["search_context_size"] = self.web_search_config["search_context_size"]
        else:
            tool["web_search_preview"]["search_context_size"] = "medium"
        
        # Add user_location if provided
        if "user_location" in self.web_search_config:
            user_loc = self.web_search_config["user_location"]
            # If user_location is already a dict with type, use it as-is
            # Otherwise, wrap it in the proper structure
            if isinstance(user_loc, dict) and "type" in user_loc:
                tool["web_search_preview"]["user_location"] = user_loc
            else:
                tool["web_search_preview"]["user_location"] = {
                    "type": "approximate",
                    "approximate": user_loc
                }
        
        return tool

    def _get_claude_web_search_tool(self) -> Dict[str, Any]:
        """
        Build Claude web search tool configuration.

        Returns:
            Claude web search tool dictionary
        """
        tool = {
            "type": "web_search_20250305",
            "name": "web_search"
        }
        
        # Add max_uses if provided
        if "max_uses" in self.web_search_config:
            tool["max_uses"] = self.web_search_config["max_uses"]
        else:
            tool["max_uses"] = 5
        
        # Add allowed_domains if provided
        if "allowed_domains" in self.web_search_config:
            tool["allowed_domains"] = self.web_search_config["allowed_domains"]
        
        # Add blocked_domains if provided
        if "blocked_domains" in self.web_search_config:
            tool["blocked_domains"] = self.web_search_config["blocked_domains"]
        
        # Add user_location if provided
        if "user_location" in self.web_search_config:
            user_loc = self.web_search_config["user_location"]
            # If user_location is already a dict with type, use it as-is
            # Otherwise, wrap it in the proper structure
            if isinstance(user_loc, dict) and "type" in user_loc:
                tool["user_location"] = user_loc
            else:
                tool["user_location"] = {
                    "type": "approximate",
                    **user_loc
                }
        
        return tool

    def _get_gemini_grounding_tool(self) -> Any:
        """
        Build Gemini Google Search grounding tool configuration.

        Returns:
            Gemini grounding tool for Google Search
        """
        if not GEMINI_AVAILABLE or genai_types is None:
            raise ImportError("Google Gemini SDK not available")

        # Build Google Search grounding tool following:
        # https://ai.google.dev/gemini-api/docs/google-search
        #
        # - For current Gemini 3 models, the recommended tool is `google_search`.
        # - Optionally, callers can configure a dynamic retrieval threshold via
        #   web_search_config["dynamic_retrieval_threshold"].

        google_search_kwargs: Dict[str, Any] = {}

        # Optional dynamic retrieval configuration (for Gemini 3+ models)
        if "dynamic_retrieval_threshold" in self.web_search_config:
            threshold = self.web_search_config["dynamic_retrieval_threshold"]
            google_search_kwargs["dynamic_retrieval_config"] = genai_types.DynamicRetrievalConfig(
                dynamic_threshold=threshold
            )

        return genai_types.Tool(
            google_search=genai_types.GoogleSearch(**google_search_kwargs)
        )

    def _get_xai_web_search_tool(self) -> Dict[str, Any]:
        """
        Build xAI (Grok) web search tool configuration.

        Returns:
            xAI web search tool dictionary
        """
        tool: Dict[str, Any] = {"type": "web_search"}

        filters: Dict[str, Any] = {}
        if "allowed_domains" in self.web_search_config:
            filters["allowed_domains"] = self.web_search_config["allowed_domains"]
        if "excluded_domains" in self.web_search_config:
            filters["excluded_domains"] = self.web_search_config["excluded_domains"]
        if self.web_search_config.get("enable_image_understanding"):
            filters["enable_image_understanding"] = True
        if filters:
            tool["filters"] = filters

        return tool

    def _web_search(self, query: str, max_results: int = 5) -> str:
        """
        Perform web search and return formatted results (Tavily-based).

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            Formatted search results as a string
        """
        if not self.tavily_client:
            logger.warning("Web search requested but Tavily client not initialized")
            return ""

        try:
            response = self.tavily_client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced"
            )

            results = []
            for idx, result in enumerate(response.get("results", []), 1):
                results.append(
                    f"{idx}. {result.get('title', 'No title')}\n"
                    f"   URL: {result.get('url', 'N/A')}\n"
                    f"   {result.get('content', 'No content')}\n"
                )

            return "\n".join(results) if results else "No results found."

        except Exception as e:
            logger.error(f"Web search failed: {str(e)}")
            return f"Web search failed: {str(e)}"

    def _retry_with_backoff(self, func, *args, **kwargs):
        """
        Execute a function with retry logic.

        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result from the function

        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries} failed: {str(e)}. "
                        f"Retrying in {self.retry_delay} seconds..."
                    )
                    time.sleep(self.retry_delay)
                else:
                    logger.error(
                        f"All {self.max_retries} attempts failed. Last error: {str(e)}"
                    )

        # If we get here, all retries failed
        raise RuntimeError(
            f"Failed after {self.max_retries} attempts. Last error: {str(last_exception)}"
        ) from last_exception

    def generate(
        self,
        prompt: str,
        use_web_search: bool = False,
        search_query: Optional[str] = None,
        use_default_web_search: Optional[bool] = None,
        system_message: Optional[str] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_completion_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> Union[str, Iterator[str]]:
        """
        Generate a response from the LLM with automatic retry logic.

        Args:
            prompt: The user prompt
            use_web_search: Whether to perform web search before generation (Tavily-based)
            search_query: Custom search query (defaults to prompt)
            use_default_web_search: Whether to use native web search tool (overrides default_web_search setting)
            system_message: System message to prepend
            stream: Enable streaming responses
            temperature: Sampling temperature (overrides default)
            max_completion_tokens: Maximum tokens (overrides default)
            **kwargs: Additional provider-specific parameters (e.g., top_p, frequency_penalty)

        Returns:
            Generated text response or iterator of response chunks

        Raises:
            RuntimeError: If all retry attempts fail
        """
        # Determine if we should use default web search
        use_native_web_search = use_default_web_search if use_default_web_search is not None else self.default_web_search

        # Handle Tavily-based web search (legacy)
        search_context = ""
        if use_web_search and self.web_search_enabled:
            query = search_query or prompt
            search_context = self._web_search(query)
            if search_context and not search_context.startswith("Web search failed"):
                prompt = f"Web Search Results:\n{search_context}\n\nUser Query: {prompt}"

        # Build messages
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})

        messages.extend(self.messages)
        messages.append({"role": "user", "content": prompt})

        # Use provided parameters or fall back to defaults
        final_temperature = temperature if temperature is not None else self.temperature
        final_max_completion_tokens = max_completion_tokens if max_completion_tokens is not None else self.max_completion_tokens

        # Generate based on provider with retry logic
        if self.provider == Provider.OPENAI:
            return self._retry_with_backoff(
                self._generate_openai,
                messages,
                stream,
                final_temperature,
                final_max_completion_tokens,
                use_native_web_search,
                **kwargs
            )
        elif self.provider == Provider.CLAUDE:
            return self._retry_with_backoff(
                self._generate_claude,
                messages,
                stream,
                system_message,
                final_temperature,
                final_max_completion_tokens,
                use_native_web_search,
                **kwargs
            )
        elif self.provider == Provider.GEMINI:
            return self._retry_with_backoff(
                self._generate_gemini,
                messages,
                stream,
                system_message,
                final_temperature,
                final_max_completion_tokens,
                use_native_web_search,
                **kwargs
            )
        elif self.provider == Provider.XAI:
            return self._retry_with_backoff(
                self._generate_xai,
                messages,
                stream,
                final_temperature,
                final_max_completion_tokens,
                use_native_web_search,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def generate_from_messages(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_completion_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> str:
        """
        Generate a response from a pre-built message list (no conversation history management).

        This is useful when the caller manages message history externally,
        e.g., for multi-turn code execution loops.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            temperature: Sampling temperature (overrides default).
            max_completion_tokens: Maximum tokens (overrides default).
            **kwargs: Additional provider-specific parameters.

        Returns:
            Generated text response.
        """
        final_temperature = temperature if temperature is not None else self.temperature
        final_max_tokens = max_completion_tokens if max_completion_tokens is not None else self.max_completion_tokens

        # Extract system message if present (needed for Claude and Gemini)
        system_message = None
        for m in messages:
            if m["role"] == "system":
                system_message = m["content"]
                break

        if self.provider == Provider.OPENAI:
            return self._retry_with_backoff(
                self._generate_openai, messages, False, final_temperature, final_max_tokens, False, **kwargs
            )
        elif self.provider == Provider.CLAUDE:
            return self._retry_with_backoff(
                self._generate_claude, messages, False, system_message, final_temperature, final_max_tokens, False, **kwargs
            )
        elif self.provider == Provider.GEMINI:
            return self._retry_with_backoff(
                self._generate_gemini, messages, False, system_message, final_temperature, final_max_tokens, False, **kwargs
            )
        elif self.provider == Provider.XAI:
            return self._retry_with_backoff(
                self._generate_xai, messages, False, final_temperature, final_max_tokens, False, **kwargs
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _is_search_preview_model(self, model: str) -> bool:
        """
        Check if the model is a search-preview model that has automatic web search.

        Args:
            model: Model name

        Returns:
            True if the model is a search-preview model
        """
        return model.endswith("-search-preview") or model.endswith("-search-api")

    @staticmethod
    def _is_openai_reasoning_model(model: str) -> bool:
        """
        Check if the model is an OpenAI reasoning model that doesn't support
        temperature, top_p, frequency_penalty, or presence_penalty.

        Covers o-series (o1, o3, o4, ...) and gpt-5 family.
        """
        name = model.lower()
        # o1, o3, o4-mini, o3-mini, etc.
        if name.startswith(("o1", "o3", "o4")):
            return True
        # gpt-5, gpt-5.2, gpt-5-mini, etc.
        if name.startswith("gpt-5"):
            return True
        return False

    def _extract_and_log_citations(self, response: Any) -> None:
        """
        Extract and log URLs from web search citations in the OpenAI response.
        
        According to OpenAI docs, citations are in message.annotations with type 'url_citation'.
        Reference: https://platform.openai.com/docs/guides/tools-web-search
        
        Args:
            response: OpenAI API response object
        """
        try:
            urls = []
            seen_urls = set()  # To avoid duplicates
            
            message = response.choices[0].message
            
            # Method 1: Check for annotations attribute (standard format)
            if hasattr(message, 'annotations') and message.annotations:
                for annotation in message.annotations:
                    # Handle both object and dict formats
                    if isinstance(annotation, dict):
                        if annotation.get('type') == 'url_citation' and 'url' in annotation:
                            url = annotation['url']
                            if url not in seen_urls:
                                seen_urls.add(url)
                                urls.append({
                                    'url': url,
                                    'title': annotation.get('title', 'No title'),
                                    'start_index': annotation.get('start_index'),
                                    'end_index': annotation.get('end_index')
                                })
                    else:
                        # Handle object format
                        annotation_type = getattr(annotation, 'type', None)
                        if annotation_type == 'url_citation':
                            url = getattr(annotation, 'url', None)
                            if url and url not in seen_urls:
                                seen_urls.add(url)
                                urls.append({
                                    'url': url,
                                    'title': getattr(annotation, 'title', 'No title'),
                                    'start_index': getattr(annotation, 'start_index', None),
                                    'end_index': getattr(annotation, 'end_index', None)
                                })
            
            # Method 2: Check response object directly (alternative location)
            if not urls and hasattr(response, 'citations'):
                citations = response.citations
                if citations:
                    for citation in citations:
                        if isinstance(citation, dict):
                            if citation.get('type') == 'url_citation' and 'url' in citation:
                                url = citation['url']
                                if url not in seen_urls:
                                    seen_urls.add(url)
                                    urls.append({
                                        'url': url,
                                        'title': citation.get('title', 'No title'),
                                        'start_index': citation.get('start_index'),
                                        'end_index': citation.get('end_index')
                                    })
            
            # Method 3: Try to access raw response dict if available
            if not urls and hasattr(response, 'model_dump'):
                try:
                    response_dict = response.model_dump()
                    # Navigate through the response structure
                    if 'choices' in response_dict and len(response_dict['choices']) > 0:
                        choice = response_dict['choices'][0]
                        if 'message' in choice:
                            msg = choice['message']
                            if 'annotations' in msg:
                                for annotation in msg['annotations']:
                                    if annotation.get('type') == 'url_citation' and 'url' in annotation:
                                        url = annotation['url']
                                        if url not in seen_urls:
                                            seen_urls.add(url)
                                            urls.append({
                                                'url': url,
                                                'title': annotation.get('title', 'No title'),
                                                'start_index': annotation.get('start_index'),
                                                'end_index': annotation.get('end_index')
                                            })
                except Exception:
                    pass  # If model_dump fails, continue
            
            # Log the URLs if found
            if urls:
                logger.info("=" * 60)
                logger.info("Web Search URLs Visited:")
                logger.info("=" * 60)
                for idx, url_info in enumerate(urls, 1):
                    logger.info(f"{idx}. {url_info['url']}")
                    if url_info['title'] and url_info['title'] != 'No title':
                        logger.info(f"   Title: {url_info['title']}")
                logger.info("=" * 60)
            else:
                logger.debug("No URL citations found in response (this is normal if no web search was performed)")
                
        except Exception as e:
            logger.warning(f"Error extracting citations: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    def _generate_openai(
        self,
        messages: List[Dict[str, str]],
        stream: bool,
        temperature: float,
        max_completion_tokens: Optional[int],
        use_native_web_search: bool = False,
        **kwargs: Any
    ) -> Union[str, Iterator[str]]:
        """Generate response using OpenAI."""
        params = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }
        
        # Handle web search for OpenAI
        # Note: For search-preview models (e.g., gpt-4o-mini-search-preview),
        # web search is automatic and no tools are needed.
        # For regular models, web search requires the Responses API (not Chat Completions),
        # so we only enable it for search-preview models.
        if use_native_web_search:
            if self._is_search_preview_model(self.model):
                # Search-preview models have automatic web search, no tools needed
                logger.info(f"Using search-preview model {self.model} - web search is automatic")
            else:
                # For regular models, web search via tools requires Responses API, not Chat Completions
                # We'll log a warning and proceed without tools
                logger.warning(
                    f"Web search requested for model {self.model}, but web search tools "
                    f"require the Responses API (not Chat Completions). "
                    f"To use web search, either use a search-preview model (e.g., gpt-4o-mini-search-preview) "
                    f"or use the Responses API directly."
                )
                # Don't add tools for regular models in Chat Completions API

        # Add max_completion_tokens if specified
        if max_completion_tokens is not None:
            params["max_completion_tokens"] = max_completion_tokens

        # Handle temperature and other parameters based on model type.
        # Some models (search-preview, reasoning/o-series, gpt-5) don't support
        # temperature, top_p, frequency_penalty, or presence_penalty.
        unsupported_sampling = {'temperature', 'top_p', 'frequency_penalty', 'presence_penalty'}
        if self._is_search_preview_model(self.model) or self._is_openai_reasoning_model(self.model):
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in unsupported_sampling}
            if filtered_kwargs != kwargs:
                removed = set(kwargs.keys()) - set(filtered_kwargs.keys())
                if removed:
                    logger.debug(f"Removed unsupported parameters for model {self.model}: {removed}")
            params.update(filtered_kwargs)
        else:
            # For regular models, add temperature (explicit parameter takes precedence over kwargs)
            params["temperature"] = temperature
            # Remove temperature from kwargs if present to avoid duplication
            kwargs_without_temp = {k: v for k, v in kwargs.items() if k != 'temperature'}
            # Add any additional kwargs (e.g., top_p, frequency_penalty, presence_penalty)
            params.update(kwargs_without_temp)

        logger.info(f"Calling OpenAI API with model: {self.model}")
        response = self.openai_client.chat.completions.create(**params)

        # Extract and log URLs from citations if web search was used
        if use_native_web_search or self._is_search_preview_model(self.model):
            self._extract_and_log_citations(response)

        if stream:
            def stream_generator() -> Iterator[str]:
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            return stream_generator()
        else:
            # Handle response - for server-side tools like web_search_preview,
            # the API executes them automatically and returns the final content
            message = response.choices[0].message
            if message.content:
                return message.content
            else:
                # If no content, return empty string (shouldn't happen with server-side tools)
                logger.warning("Received response with no content")
                return ""

    def _extract_xai_text(self, response: Any) -> str:
        """Extract text from xAI Responses API response."""
        try:
            if hasattr(response, "output_text") and response.output_text:
                return response.output_text

            if hasattr(response, "output") and response.output:
                parts = []
                for item in response.output:
                    for content in getattr(item, "content", []):
                        if getattr(content, "type", None) == "output_text":
                            text = getattr(content, "text", None)
                            if text:
                                parts.append(text)
                if parts:
                    return "".join(parts)

            if hasattr(response, "model_dump"):
                data = response.model_dump()
                outputs = data.get("output", [])
                parts = []
                for item in outputs:
                    for content in item.get("content", []):
                        if content.get("type") == "output_text":
                            text = content.get("text")
                            if text:
                                parts.append(text)
                if parts:
                    return "".join(parts)
        except Exception as err:  # pragma: no cover - best-effort parsing
            logger.debug(f"Failed to extract xAI text: {err}")

        logger.warning("Received xAI response with no content")
        return ""

    def _generate_xai(
        self,
        messages: List[Dict[str, str]],
        stream: bool,
        temperature: float,
        max_completion_tokens: Optional[int],
        use_native_web_search: bool = False,
        **kwargs: Any,
    ) -> Union[str, Iterator[str]]:
        """Generate response using xAI (Grok) Responses API."""
        params: Dict[str, Any] = {
            "model": self.model,
            "input": messages,
            "stream": stream,
        }

        tools = []

        if use_native_web_search:
            tools.append(self._get_xai_web_search_tool())
            logger.info("xAI web search tool enabled")

        if tools:
            params["tools"] = tools

        params["temperature"] = temperature
        if max_completion_tokens is not None:
            params["max_output_tokens"] = max_completion_tokens

        # Pass through additional kwargs supported by Responses API
        passthrough_keys = {
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "response_format",
        }
        for key in passthrough_keys:
            if key in kwargs:
                params[key] = kwargs[key]

        logger.info(f"Calling xAI Responses API with model: {self.model}")
        response = self.xai_client.responses.create(**params)

        if stream:
            def stream_generator() -> Iterator[str]:
                for event in response:
                    # Stream events may expose output_text or delta content
                    if hasattr(event, "output_text") and event.output_text:
                        yield event.output_text
                        continue
                    if hasattr(event, "delta") and event.delta:
                        delta_output = getattr(event.delta, "output_text", None)
                        if isinstance(delta_output, str):
                            if delta_output:
                                yield delta_output
                            continue
                        if isinstance(delta_output, list):
                            for delta_part in delta_output:
                                if delta_part:
                                    yield delta_part
                            continue
                    if hasattr(event, "output") and event.output:
                        for item in event.output:
                            for content in getattr(item, "content", []):
                                if getattr(content, "type", None) == "output_text":
                                    text = getattr(content, "text", None)
                                    if text:
                                        yield text
            return stream_generator()

        return self._extract_xai_text(response)

    def _generate_claude(
        self,
        messages: List[Dict[str, str]],
        stream: bool,
        system_message: Optional[str],
        temperature: float,
        max_completion_tokens: Optional[int],
        use_native_web_search: bool = False,
        **kwargs: Any
    ) -> Union[str, Iterator[str]]:
        """Generate response using Claude."""
        # Claude separates system message from messages
        filtered_messages = [m for m in messages if m["role"] != "system"]

        params = {
            "model": self.model,
            "messages": filtered_messages,
            "temperature": temperature,
            "max_tokens": max_completion_tokens or 4096,
        }

        # Add system message if provided
        if system_message:
            params["system"] = system_message

        # Add web search tool if enabled
        if use_native_web_search:
            web_search_tool = self._get_claude_web_search_tool()
            # Check if tools already exist in kwargs or params
            if "tools" in kwargs:
                # Create a new list to avoid mutating the original
                tools = list(kwargs["tools"]) if isinstance(kwargs["tools"], list) else [kwargs["tools"]]
                tools.append(web_search_tool)
                kwargs["tools"] = tools
            elif "tools" in params:
                tools = list(params["tools"]) if isinstance(params["tools"], list) else [params["tools"]]
                tools.append(web_search_tool)
                params["tools"] = tools
            else:
                params["tools"] = [web_search_tool]
            logger.info("Claude web search tool enabled")

        # Add any additional kwargs (e.g., top_p, top_k)
        params.update(kwargs)

        logger.info(f"Calling Claude API with model: {self.model}")

        if stream:
            params["stream"] = True
            response = self.anthropic_client.messages.create(**params)

            def stream_generator() -> Iterator[str]:
                for event in response:
                    # Handle content block deltas with text
                    if event.type == 'content_block_delta':
                        if hasattr(event.delta, 'text') and event.delta.text:
                            yield event.delta.text
            return stream_generator()
        else:
            response = self.anthropic_client.messages.create(**params)
            # Extract text from content blocks
            text_parts = []
            for content_block in response.content:
                if content_block.type == "text":
                    text_parts.append(content_block.text)
            return "".join(text_parts) if text_parts else ""

    def _generate_gemini(
        self,
        messages: List[Dict[str, str]],
        stream: bool,
        system_message: Optional[str],
        temperature: float,
        max_completion_tokens: Optional[int],
        use_native_web_search: bool = False,
        **kwargs: Any
    ) -> Union[str, Iterator[str]]:
        """Generate response using Google Gemini."""
        if not GEMINI_AVAILABLE or genai_types is None:
            raise ImportError("Google Gemini SDK not available")

        # Build contents for Gemini API
        # Gemini uses "user" and "model" roles (not "assistant")
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                # System messages are handled separately via system_instruction
                continue
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append(genai_types.Content(role=role, parts=[genai_types.Part(text=msg["content"])]))

        # Build generation config
        config_params: Dict[str, Any] = {}

        # Add system instruction if provided
        if system_message:
            config_params["system_instruction"] = system_message

        # Add temperature (Gemini 2.5+ defaults to 1.0)
        config_params["temperature"] = temperature

        # Add max_output_tokens if specified
        if max_completion_tokens is not None:
            config_params["max_output_tokens"] = max_completion_tokens

        # Add other supported parameters from kwargs
        supported_params = {"top_p", "top_k", "candidate_count", "stop_sequences", "presence_penalty", "frequency_penalty", "seed"}
        for param in supported_params:
            if param in kwargs:
                config_params[param] = kwargs[param]

        # Build tools list if web search is enabled
        tools = None
        if use_native_web_search:
            grounding_tool = self._get_gemini_grounding_tool()
            tools = [grounding_tool]
            logger.info("Gemini Google Search grounding enabled")

        # Attach tools directly to the GenerateContentConfig, as in:
        # https://ai.google.dev/gemini-api/docs/google-search
        if tools is not None:
            config_params["tools"] = tools

        # Create generation config
        config = genai_types.GenerateContentConfig(**config_params)

        logger.info(f"Calling Gemini API with model: {self.model}")

        if stream:
            # Streaming response
            response_stream = self.gemini_client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config,
            )

            def stream_generator() -> Iterator[str]:
                for chunk in response_stream:
                    if chunk.text:
                        yield chunk.text

            return stream_generator()
        else:
            # Non-streaming response
            response = self.gemini_client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )

            # Extract text from response
            if response.text:
                return response.text
            else:
                # Check for candidates if text property is empty
                if response.candidates:
                    text_parts = []
                    for candidate in response.candidates:
                        if candidate.content and candidate.content.parts:
                            for part in candidate.content.parts:
                                if hasattr(part, "text") and part.text:
                                    text_parts.append(part.text)
                    if text_parts:
                        return "".join(text_parts)

                logger.warning("Received Gemini response with no content")
                return ""

    def create_container(self, name: str = "dc-session", memory_limit: str = "4g") -> str:
        """
        Create an OpenAI code interpreter container and return its ID.

        Args:
            name: Container name.
            memory_limit: Memory limit for the container (e.g., "4g").

        Returns:
            The container ID string.
        """
        if self.provider != Provider.OPENAI:
            raise ValueError("Code interpreter containers are only supported with the OpenAI provider.")
        container = self.openai_client.containers.create(name=name, memory_limit=memory_limit)
        logger.info(f"Created code interpreter container: {container.id}")
        return container.id

    def generate_with_code_interpreter(
        self,
        messages: List[Dict[str, str]],
        container_id: str,
        temperature: Optional[float] = None,
        max_completion_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate a response using OpenAI's Responses API with Code Interpreter.

        Code execution happens server-side — no local subprocess needed.

        Args:
            messages: Conversation messages (system/user/assistant dicts).
            container_id: The container ID for code interpreter.
            temperature: Sampling temperature (overrides default).
            max_completion_tokens: Max tokens (overrides default).
            **kwargs: Additional parameters (e.g., reasoning_effort).

        Returns:
            The model's text output.
        """
        if self.provider != Provider.OPENAI:
            raise ValueError("Code interpreter is only supported with the OpenAI provider.")

        final_temperature = temperature if temperature is not None else self.temperature
        final_max_tokens = max_completion_tokens if max_completion_tokens is not None else self.max_completion_tokens

        # Remap Chat Completions kwargs to Responses API format
        # reasoning_effort="medium" -> reasoning={"effort": "medium"}
        if "reasoning_effort" in kwargs:
            kwargs["reasoning"] = {"effort": kwargs.pop("reasoning_effort")}

        params: Dict[str, Any] = {
            "model": self.model,
            "input": messages,
            "tools": [{"type": "code_interpreter", "container": container_id}],
        }

        # Reasoning models don't support temperature
        if self._is_openai_reasoning_model(self.model):
            filtered_kwargs = {k: v for k, v in kwargs.items()
                               if k not in {'temperature', 'top_p', 'frequency_penalty', 'presence_penalty'}}
            params.update(filtered_kwargs)
        else:
            params["temperature"] = final_temperature
            kwargs_filtered = {k: v for k, v in kwargs.items() if k != 'temperature'}
            params.update(kwargs_filtered)

        if final_max_tokens is not None:
            params["max_output_tokens"] = final_max_tokens

        logger.info(f"Calling OpenAI Responses API (code interpreter) with model: {self.model}")

        def _call():
            response = self.openai_client.responses.create(**params)
            return response.output_text

        return self._retry_with_backoff(_call)

    def generate_with_code_interpreter_claude(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_completion_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate a response using Claude's native code execution tool.

        Code execution happens server-side in Anthropic's sandboxed environment —
        no local subprocess or container management needed.

        Pricing note: 1,550 free hours/month per organisation; additional usage
        at $0.05/hour per container.  Execution is **free** when web_search or
        web_fetch tools are also included in the same request.

        Args:
            messages: Conversation messages (system/user/assistant dicts).
            temperature: Sampling temperature (overrides default).
            max_completion_tokens: Max tokens (overrides default).
            **kwargs: Additional Claude parameters.  OpenAI-specific keys such as
                ``reasoning_effort`` are silently ignored.

        Returns:
            The model's text output (all text content blocks concatenated).
        """
        if self.provider != Provider.CLAUDE:
            raise ValueError("Claude code execution is only supported with the Claude provider.")

        # Drop OpenAI-specific params that Claude doesn't accept
        kwargs.pop("reasoning_effort", None)

        final_temperature = temperature if temperature is not None else self.temperature
        final_max_tokens = max_completion_tokens if max_completion_tokens is not None else self.max_completion_tokens

        code_execution_tool = {"type": "code_execution_20250825", "name": "code_execution"}

        # Extract system message (if any) to pass separately, matching _generate_claude's convention
        system_message = next((m["content"] for m in messages if m["role"] == "system"), None)

        logger.info(f"Calling Claude API (code execution tool) with model: {self.model}")

        def _call():
            return self._generate_claude(
                messages=messages,
                stream=False,
                system_message=system_message,
                temperature=final_temperature,
                max_completion_tokens=final_max_tokens,
                use_native_web_search=False,
                tools=[code_execution_tool],
                **kwargs,
            )

        return self._retry_with_backoff(_call)

    def chat(
        self,
        message: str,
        use_web_search: bool = False,
        use_default_web_search: Optional[bool] = None,
        **kwargs: Any
    ) -> str:
        """
        Send a message in a conversational context (maintains history).

        Args:
            message: The user message
            use_web_search: Whether to use Tavily-based web search
            use_default_web_search: Whether to use native web search tool (overrides default_web_search setting)
            **kwargs: Additional generation parameters (temperature, max_completion_tokens, etc.)

        Returns:
            Assistant's response

        Example:
            >>> client.chat("My name is Alice")
            >>> client.chat("What's my name?")  # Remembers the previous context
        """
        response = self.generate(
            message,
            use_web_search=use_web_search,
            use_default_web_search=use_default_web_search,
            **kwargs
        )

        # Handle streaming response
        if isinstance(response, str):
            full_response = response
        else:
            full_response = "".join(response)

        # Update history
        self.messages.append({"role": "user", "content": message})
        self.messages.append({"role": "assistant", "content": full_response})

        return full_response

    def reset_history(self) -> None:
        """Clear the conversation history."""
        self.messages = []
        logger.info("Conversation history cleared")

    def get_history(self) -> List[Dict[str, str]]:
        """
        Get the current conversation history.

        Returns:
            List of message dictionaries with 'role' and 'content'
        """
        return self.messages.copy()

    def set_history(self, messages: List[Dict[str, str]]) -> None:
        """
        Set the conversation history.

        Args:
            messages: List of message dictionaries with 'role' and 'content'

        Example:
            >>> history = [
            ...     {"role": "user", "content": "Hello"},
            ...     {"role": "assistant", "content": "Hi there!"}
            ... ]
            >>> client.set_history(history)
        """
        self.messages = messages.copy()
        logger.info(f"Conversation history set with {len(messages)} messages")

    @classmethod
    def from_env(
        cls,
        provider: Union[str, Provider],
        web_search_enabled: bool = False,
        **kwargs: Any
    ) -> "UnifiedLLMClient":
        """
        Create a client using environment variables for API keys.

        Expected environment variables:
        - OPENAI_API_KEY (for OpenAI)
        - ANTHROPIC_API_KEY (for Claude)
        - GEMINI_API_KEY or GOOGLE_API_KEY (for Gemini)
        - XAI_API_KEY (for xAI / Grok)
        - TAVILY_API_KEY (for Tavily-based web search, optional)

        Args:
            provider: The LLM provider ("openai", "claude", "gemini", or "xai")
            web_search_enabled: Enable Tavily-based web search
            **kwargs: Additional client parameters (model, temperature, max_completion_tokens,
                     default_web_search, web_search_config, etc.)

        Returns:
            Configured UnifiedLLMClient instance

        Example:
            >>> client = UnifiedLLMClient.from_env(
            ...     provider="openai",
            ...     temperature=0.8,
            ...     max_completion_tokens=1000,
            ...     default_web_search=True
            ... )
        """
        return cls(
            provider=provider,
            web_search_enabled=web_search_enabled,
            **kwargs
        )
