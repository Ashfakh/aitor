"""
LLM interface for ReAct agents.
Provides a unified interface to call different LLM providers using their official clients.
"""

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union

# Structured response models
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    MOCK = "mock"


class Message:
    """LLM message format."""

    def __init__(self, role: str, content: str):
        """
        Initialize message.

        Args:
            role: Message role (system, user, assistant)
            content: Message content
        """
        self.role = role
        self.content = content

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "Message":
        """Create from dictionary."""
        return cls(role=data["role"], content=data["content"])


class LLMResponse:
    """Response from LLM."""

    def __init__(
        self,
        content: str,
        model: str,
        usage: Optional[Dict[str, int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize LLM response.

        Args:
            content: Response content
            model: Model used
            usage: Token usage information
            metadata: Additional metadata
        """
        self.content = content
        self.model = model
        self.usage = usage or {}
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "model": self.model,
            "usage": self.usage,
            "metadata": self.metadata,
        }


class BaseLLM(ABC):
    """Base class for LLM implementations."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM.

        Args:
            config: LLM configuration
        """
        self.config = config
        self.model = config.get("model", "default")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 2000)
        self.timeout = config.get("timeout", 30.0)

    @abstractmethod
    async def generate(self, messages: List[Message], **kwargs) -> LLMResponse:
        """
        Generate response from messages.

        Args:
            messages: List of messages
            **kwargs: Additional parameters

        Returns:
            LLM response
        """
        pass

    @abstractmethod
    async def generate_json(
        self, messages: List[Message], schema: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Generate JSON response from messages.

        Args:
            messages: List of messages
            schema: Expected JSON schema
            **kwargs: Additional parameters

        Returns:
            Parsed JSON response
        """
        pass

    @abstractmethod
    async def generate_structured(
        self, messages: List[Message], response_model: type[BaseModel], **kwargs
    ) -> BaseModel:
        """
        Generate structured response using Pydantic models.

        Args:
            messages: List of messages
            response_model: Pydantic model class for response structure
            **kwargs: Additional parameters

        Returns:
            Pydantic model instance
        """
        pass


class OpenAILLM(BaseLLM):
    """OpenAI LLM implementation using official client."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OpenAI LLM.

        Args:
            config: LLM configuration
        """
        super().__init__(config)
        self.api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        self.base_url = config.get("base_url")  # For Azure or custom endpoints
        self.model = config.get("model", "gpt-4")
        self.organization = config.get("organization") or os.getenv("OPENAI_ORG_ID")

        if not self.api_key:
            raise ValueError("OpenAI API key not provided")

        self._client = None
        self._instructor_client = None

    @property
    def client(self):
        """Get or create OpenAI client with connection pooling."""
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError("Please install openai: pip install openai")

            # Client configuration
            client_kwargs = {"api_key": self.api_key, "timeout": self.timeout}

            if self.organization:
                client_kwargs["organization"] = self.organization

            if self.base_url:
                client_kwargs["base_url"] = self.base_url

            # Create async client with connection pooling built-in
            self._client = openai.AsyncOpenAI(**client_kwargs)

        return self._client

    @property
    def instructor_client(self):
        """Get or create instructor-patched OpenAI client."""
        if self._instructor_client is None:
            try:
                import instructor
            except ImportError:
                raise ImportError("Please install instructor: pip install instructor")

            # Patch the regular client with instructor
            self._instructor_client = instructor.from_openai(self.client)

        return self._instructor_client

    async def generate(self, messages: List[Message], **kwargs) -> LLMResponse:
        """Generate response using OpenAI API."""
        try:
            # Merge kwargs with defaults
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)

            # Additional OpenAI-specific parameters
            params = {
                "model": self.model,
                "messages": [msg.to_dict() for msg in messages],
            }

            # Use max_completion_tokens for newer models, max_tokens for older ones
            if "o3" in self.model.lower() or "gpt-4o" in self.model.lower():
                params["max_completion_tokens"] = max_tokens
                # o3 models don't support temperature
                if "o3" not in self.model.lower():
                    params["temperature"] = temperature
            else:
                params["max_tokens"] = max_tokens
                params["temperature"] = temperature

            # Add optional parameters if provided
            for key in [
                "top_p",
                "n",
                "stop",
                "presence_penalty",
                "frequency_penalty",
                "logit_bias",
                "user",
            ]:
                if key in kwargs:
                    params[key] = kwargs[key]

            # Add response format if specified
            if "response_format" in kwargs:
                params["response_format"] = kwargs["response_format"]

            # Make API call
            response = await self.client.chat.completions.create(**params)

            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                if response.usage
                else {},
            )

        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise

    async def generate_json(
        self, messages: List[Message], schema: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Dict[str, Any]:
        """Generate JSON response using OpenAI API."""
        # Use JSON mode if available (GPT-4 Turbo and later)
        if (
            "gpt-4" in self.model
            and "1106" in self.model
            or "gpt-4-turbo" in self.model
        ):
            kwargs["response_format"] = {"type": "json_object"}

            # Add instruction to system message
            json_messages = messages.copy()
            json_messages.insert(
                0,
                Message(
                    role="system",
                    content="You must respond with valid JSON that follows the specified format.",
                ),
            )
        else:
            # For older models, add JSON instruction
            json_messages = messages.copy()
            json_messages.append(
                Message(
                    role="system",
                    content="Respond with valid JSON only. Do not include any text outside the JSON.",
                )
            )

        response = await self.generate(json_messages, **kwargs)

        try:
            return json.loads(response.content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            logger.error(f"Response content: {response.content}")
            raise

    async def generate_structured(
        self, messages: List[Message], response_model: type[BaseModel], **kwargs
    ) -> BaseModel:
        """Generate structured response using instructor and Pydantic models."""
        try:
            # Merge kwargs with defaults
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)

            # Build parameters
            params = {
                "model": self.model,
                "messages": [msg.to_dict() for msg in messages],
                "response_model": response_model,
            }

            # Use max_completion_tokens for newer models, max_tokens for older ones
            if "o3" in self.model.lower() or "gpt-4o" in self.model.lower():
                params["max_completion_tokens"] = max_tokens
                # o3 models don't support temperature
                if "o3" not in self.model.lower():
                    params["temperature"] = temperature
            else:
                params["max_tokens"] = max_tokens
                params["temperature"] = temperature

            # Add optional parameters if provided
            for key in [
                "top_p",
                "n",
                "stop",
                "presence_penalty",
                "frequency_penalty",
                "logit_bias",
                "user",
            ]:
                if key in kwargs:
                    params[key] = kwargs[key]

            # Make API call with instructor
            response = await self.instructor_client.chat.completions.create(**params)

            return response

        except Exception as e:
            logger.error(f"OpenAI structured generation error: {str(e)}")
            raise


class AnthropicLLM(BaseLLM):
    """Anthropic Claude LLM implementation using official client."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Anthropic LLM.

        Args:
            config: LLM configuration
        """
        super().__init__(config)
        self.api_key = config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        self.model = config.get("model", "claude-3-opus-20240229")

        if not self.api_key:
            raise ValueError("Anthropic API key not provided")

        self._client = None

    @property
    def client(self):
        """Get or create Anthropic client with connection pooling."""
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError("Please install anthropic: pip install anthropic")

            # Create async client with connection pooling built-in
            self._client = anthropic.AsyncAnthropic(
                api_key=self.api_key, timeout=self.timeout
            )

        return self._client

    async def generate(self, messages: List[Message], **kwargs) -> LLMResponse:
        """Generate response using Anthropic API."""
        try:
            # Convert messages to Anthropic format
            system_message = None
            chat_messages = []

            for msg in messages:
                if msg.role == "system":
                    # Combine multiple system messages
                    if system_message:
                        system_message += "\n\n" + msg.content
                    else:
                        system_message = msg.content
                else:
                    chat_messages.append(msg.to_dict())

            # Merge kwargs with defaults
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)

            # Build parameters
            params = {
                "model": self.model,
                "messages": chat_messages,
            }

            # Use max_completion_tokens for newer models, max_tokens for older ones
            if "o3" in self.model.lower() or "gpt-4o" in self.model.lower():
                params["max_completion_tokens"] = max_tokens
                # o3 models don't support temperature
                if "o3" not in self.model.lower():
                    params["temperature"] = temperature
            else:
                params["max_tokens"] = max_tokens
                params["temperature"] = temperature

            if system_message:
                params["system"] = system_message

            # Add optional parameters if provided
            for key in ["top_p", "top_k", "stop_sequences"]:
                if key in kwargs:
                    params[key] = kwargs[key]

            # Make API call
            response = await self.client.messages.create(**params)

            # Extract text content
            content = ""
            for content_block in response.content:
                if content_block.type == "text":
                    content += content_block.text

            return LLMResponse(
                content=content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens
                    + response.usage.output_tokens,
                }
                if hasattr(response, "usage")
                else {},
            )

        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise

    async def generate_json(
        self, messages: List[Message], schema: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Dict[str, Any]:
        """Generate JSON response using Anthropic API."""
        # Add JSON instruction
        json_messages = messages.copy()
        json_messages.append(
            Message(
                role="user",
                content="Please respond with valid JSON only. Do not include any explanatory text outside the JSON.",
            )
        )

        response = await self.generate(json_messages, **kwargs)

        try:
            # Clean response content (Claude sometimes adds explanatory text)
            content = response.content.strip()

            # Find JSON in the response
            json_start = content.find("{")
            json_end = content.rfind("}") + 1

            if json_start != -1 and json_end > json_start:
                json_str = content[json_start:json_end]
                return json.loads(json_str)
            else:
                return json.loads(content)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            logger.error(f"Response content: {response.content}")
            raise

    async def generate_structured(
        self, messages: List[Message], response_model: type[BaseModel], **kwargs
    ) -> BaseModel:
        """Generate structured response - fallback to JSON parsing for Anthropic."""
        # For now, fallback to generate_json and parse manually
        # TODO: Implement instructor support for Anthropic when available
        response_json = await self.generate_json(messages, **kwargs)
        return response_model.model_validate(response_json)


class MockLLM(BaseLLM):
    """Mock LLM for testing and development."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Mock LLM.

        Args:
            config: LLM configuration
        """
        super().__init__(config)
        self.responses = config.get("responses", {})
        self.default_response = config.get(
            "default_response", "This is a mock response."
        )
        self.delay = config.get("delay", 0.1)

    async def generate(self, messages: List[Message], **kwargs) -> LLMResponse:
        """Generate mock response."""
        # Get last user message
        last_user_msg = None
        for msg in reversed(messages):
            if msg.role == "user":
                last_user_msg = msg.content
                break

        # Check for predefined responses
        content = self.default_response
        if last_user_msg:
            for pattern, response in self.responses.items():
                if pattern.lower() in last_user_msg.lower():
                    content = response
                    break

        # Simulate API delay
        await asyncio.sleep(self.delay)

        return LLMResponse(
            content=content,
            model="mock",
            usage={
                "prompt_tokens": sum(len(msg.content.split()) for msg in messages),
                "completion_tokens": len(content.split()),
                "total_tokens": sum(len(msg.content.split()) for msg in messages)
                + len(content.split()),
            },
        )

    async def generate_json(
        self, messages: List[Message], schema: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Dict[str, Any]:
        """Generate mock JSON response."""
        response = await self.generate(messages, **kwargs)

        # Try to parse as JSON, otherwise return default
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {"response": response.content, "status": "mock"}

    async def generate_structured(
        self, messages: List[Message], response_model: type[BaseModel], **kwargs
    ) -> BaseModel:
        """Generate mock structured response."""
        # Create a simple mock response that matches the model structure
        if hasattr(response_model, "model_fields"):
            mock_data = {}
            for field_name, field_info in response_model.model_fields.items():
                if field_name == "action":
                    mock_data[field_name] = "think"  # Default action
                elif field_name == "content":
                    mock_data[field_name] = "This is a mock response"
                elif field_name == "tool_name":
                    mock_data[field_name] = "mock_tool"
                elif field_name == "parameters":
                    mock_data[field_name] = "{}"
                else:
                    mock_data[field_name] = "mock_value"

            return response_model.model_validate(mock_data)
        else:
            # Fallback for basic models
            return response_model()


class LLMFactory:
    """Factory for creating LLM instances."""

    _providers = {
        LLMProvider.OPENAI: OpenAILLM,
        LLMProvider.ANTHROPIC: AnthropicLLM,
        LLMProvider.MOCK: MockLLM,
    }

    @classmethod
    def create(
        cls, provider: Union[str, LLMProvider], config: Dict[str, Any]
    ) -> BaseLLM:
        """
        Create LLM instance.

        Args:
            provider: LLM provider name or enum
            config: LLM configuration

        Returns:
            LLM instance
        """
        if isinstance(provider, str):
            provider = LLMProvider(provider.lower())

        if provider not in cls._providers:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        llm_class = cls._providers[provider]
        return llm_class(config)  # type: ignore

    @classmethod
    def register_provider(cls, provider: LLMProvider, llm_class: type):
        """
        Register a new LLM provider.

        Args:
            provider: Provider enum
            llm_class: LLM class
        """
        cls._providers[provider] = llm_class  # type: ignore


class LLMManager:
    """
    Manager for handling multiple LLM configurations and instances.
    """

    def __init__(self, default_provider: Optional[str] = None):
        """
        Initialize LLM manager.

        Args:
            default_provider: Default provider to use
        """
        self.default_provider = default_provider or "mock"
        self._llms: Dict[str, BaseLLM] = {}
        self._configs: Dict[str, Dict[str, Any]] = {}

    def add_provider(self, name: str, provider: str, config: Dict[str, Any]) -> None:
        """
        Add a new LLM provider configuration.

        Args:
            name: Configuration name
            provider: Provider type
            config: Provider configuration
        """
        self._configs[name] = {"provider": provider, "config": config}

        # Create LLM instance
        llm = LLMFactory.create(provider, config)
        self._llms[name] = llm

        logger.info(f"Added LLM provider '{name}' using {provider}")

    def get_llm(self, name: Optional[str] = None) -> BaseLLM:
        """
        Get LLM instance by name.

        Args:
            name: Configuration name (uses default if None)

        Returns:
            LLM instance
        """
        if name is None:
            name = self.default_provider

        if name not in self._llms:
            raise ValueError(f"LLM configuration '{name}' not found")

        return self._llms[name]

    async def generate(
        self, messages: List[Message], provider: Optional[str] = None, **kwargs
    ) -> LLMResponse:
        """
        Generate response using specified provider.

        Args:
            messages: List of messages
            provider: Provider name (uses default if None)
            **kwargs: Additional parameters

        Returns:
            LLM response
        """
        llm = self.get_llm(provider)
        return await llm.generate(messages, **kwargs)

    async def generate_json(
        self,
        messages: List[Message],
        schema: Optional[Dict[str, Any]] = None,
        provider: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate JSON response using specified provider.

        Args:
            messages: List of messages
            schema: Expected JSON schema
            provider: Provider name (uses default if None)
            **kwargs: Additional parameters

        Returns:
            Parsed JSON response
        """
        llm = self.get_llm(provider)
        return await llm.generate_json(messages, schema, **kwargs)

    async def generate_structured(
        self,
        messages: List[Message],
        response_model: type[BaseModel],
        provider: Optional[str] = None,
        **kwargs,
    ) -> BaseModel:
        """
        Generate structured response using specified provider.

        Args:
            messages: List of messages
            response_model: Pydantic model class for response structure
            provider: Provider name (uses default if None)
            **kwargs: Additional parameters

        Returns:
            Pydantic model instance
        """
        llm = self.get_llm(provider)
        return await llm.generate_structured(messages, response_model, **kwargs)

    def list_providers(self) -> List[str]:
        """List available provider configurations."""
        return list(self._configs.keys())

    def get_config(self, name: str) -> Dict[str, Any]:
        """Get provider configuration."""
        if name not in self._configs:
            raise ValueError(f"Configuration '{name}' not found")
        return self._configs[name].copy()

    def set_default(self, name: str) -> None:
        """Set default provider."""
        if name not in self._llms:
            raise ValueError(f"Provider '{name}' not configured")
        self.default_provider = name
        logger.info(f"Set default LLM provider to '{name}'")
