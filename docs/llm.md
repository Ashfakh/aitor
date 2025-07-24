# LLM Integration Guide

Aitor provides a unified interface for integrating with multiple Large Language Model providers, including OpenAI, Anthropic, and custom providers.

## Overview

The LLM system in Aitor consists of:
- **LLMManager**: Central manager for multiple providers
- **LLMProvider**: Abstract interface for different providers
- **Built-in Providers**: OpenAI, Anthropic, and Mock providers
- **Configuration Management**: Provider-specific settings
- **Error Handling**: Automatic failover and retry logic

## Quick Start

### Basic Setup

```python
from aitor.llm import LLMManager

# Create LLM manager
llm_manager = LLMManager()

# Add OpenAI provider
llm_manager.add_provider(
    name="openai",
    provider="openai",
    config={
        "api_key": "sk-your-openai-api-key",
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 500
    }
)

# Add Anthropic provider
llm_manager.add_provider(
    name="claude",
    provider="anthropic", 
    config={
        "api_key": "sk-ant-your-anthropic-key",
        "model": "claude-3-opus-20240229",
        "max_tokens": 500
    }
)

# Set default provider
llm_manager.set_default_provider("openai")
```

### Using with Agents

```python
from aitor import create_react_agent

# Create agent with LLM manager
agent = await create_react_agent(
    name="SmartAgent",
    llm_provider="openai",  # Uses provider from manager
    llm_config={
        "api_key": "sk-your-key",
        "model": "gpt-4",
        "temperature": 0.7
    },
    tools=[your_tools]
)

# Or use existing LLM manager
agent = ReactAgent(
    name="SmartAgent",
    llm_manager=llm_manager
)
```

## Supported Providers

### OpenAI

Supports GPT-3.5, GPT-4, and o3-mini models:

```python
# Standard GPT-4 configuration
openai_config = {
    "api_key": "sk-your-key",
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 500,
    "top_p": 0.9,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}

# o3-mini specific configuration (no temperature support)
o3_mini_config = {
    "api_key": "sk-your-key", 
    "model": "o3-mini",
    "max_completion_tokens": 500,  # Use max_completion_tokens instead of max_tokens
    # Note: o3-mini doesn't support temperature parameter
}

llm_manager.add_provider("openai", "openai", openai_config)
llm_manager.add_provider("o3", "openai", o3_mini_config)
```

### Anthropic

Supports Claude 3.x series models:

```python
# Claude 3 Opus (most capable)
claude_opus_config = {
    "api_key": "sk-ant-your-key",
    "model": "claude-3-opus-20240229",
    "max_tokens": 500,
    "temperature": 0.7
}

# Claude 3 Sonnet (balanced)
claude_sonnet_config = {
    "api_key": "sk-ant-your-key",
    "model": "claude-3-sonnet-20240229", 
    "max_tokens": 500,
    "temperature": 0.7
}

# Claude 3 Haiku (fastest)
claude_haiku_config = {
    "api_key": "sk-ant-your-key",
    "model": "claude-3-haiku-20240307",
    "max_tokens": 500,
    "temperature": 0.7
}

llm_manager.add_provider("claude-opus", "anthropic", claude_opus_config)
llm_manager.add_provider("claude-sonnet", "anthropic", claude_sonnet_config)
llm_manager.add_provider("claude-haiku", "anthropic", claude_haiku_config)
```

### Mock Provider

For testing and development:

```python
# Mock provider with predefined responses
mock_config = {
    "responses": {
        "calculator": "ACT: calculator(expression='2+2')",
        "search": "ACT: google_web_search(query='test search')",
        "default": "THINK: I need to analyze this problem step by step."
    },
    "default_response": "I understand. Let me help you with that.",
    "delay": 0.1  # Simulate API latency
}

llm_manager.add_provider("mock", "mock", mock_config)
```

## Advanced Configuration

### Provider Switching

```python
# Switch providers dynamically
agent.set_llm_provider("claude-opus")  # Switch to Claude
response1 = await agent.solve("Complex reasoning task")

agent.set_llm_provider("openai")  # Switch to OpenAI
response2 = await agent.solve("Code generation task")

agent.set_llm_provider("claude-haiku")  # Switch to fastest Claude
response3 = await agent.solve("Quick question")
```

### Failover Configuration

```python
# Configure automatic failover
llm_manager.set_failover_sequence([
    "openai",      # Try OpenAI first
    "claude-sonnet", # Fall back to Claude Sonnet
    "claude-haiku"   # Finally try Claude Haiku
])

# The agent will automatically try providers in order if one fails
agent = ReactAgent(name="RobustAgent", llm_manager=llm_manager)
```

### Custom Provider

Create custom providers for other LLM services:

```python
from aitor.llm import LLMProvider
from typing import List, Dict, Any
import httpx

class CustomLLMProvider(LLMProvider):
    """Custom provider for a different LLM service."""
    
    def __init__(self, config: Dict[str, Any]):
        self.api_key = config["api_key"]
        self.model = config.get("model", "default-model")
        self.base_url = config.get("base_url", "https://api.custom-llm.com")
        self.max_tokens = config.get("max_tokens", 500)
        self.temperature = config.get("temperature", 0.7)
    
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Generate response from custom LLM service."""
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    **kwargs
                },
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise Exception(f"Custom LLM API error: {response.status_code}")
            
            data = response.json()
            return data["choices"][0]["message"]["content"]
    
    async def generate_structured_response(
        self,
        messages: List[Dict[str, str]],
        response_format: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate structured JSON response."""
        
        # Add format instructions to the last message
        format_instruction = f"\nPlease respond in the following JSON format: {response_format}"
        messages[-1]["content"] += format_instruction
        
        response_text = await self.generate_response(messages, **kwargs)
        
        try:
            import json
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {"response": response_text}

# Register custom provider
custom_config = {
    "api_key": "your-custom-api-key",
    "model": "custom-model-v1",
    "base_url": "https://api.custom-llm.com",
    "max_tokens": 1000,
    "temperature": 0.8
}

# Register the custom provider class
llm_manager.register_provider_class("custom", CustomLLMProvider)
llm_manager.add_provider("my-custom", "custom", custom_config)
```

## Environment Configuration

### Environment Variables

```bash
# OpenAI
export OPENAI_API_KEY="sk-your-openai-key"

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-key"

# Custom provider
export CUSTOM_LLM_API_KEY="your-custom-key"
export CUSTOM_LLM_BASE_URL="https://api.custom-llm.com"
```

### Configuration from Environment

```python
import os
from aitor.llm import LLMManager

def create_llm_manager_from_env() -> LLMManager:
    """Create LLM manager with configuration from environment variables."""
    
    llm_manager = LLMManager()
    
    # OpenAI configuration
    if openai_key := os.getenv("OPENAI_API_KEY"):
        llm_manager.add_provider("openai", "openai", {
            "api_key": openai_key,
            "model": os.getenv("OPENAI_MODEL", "gpt-4"),
            "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
            "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "500"))
        })
    
    # Anthropic configuration
    if anthropic_key := os.getenv("ANTHROPIC_API_KEY"):
        llm_manager.add_provider("claude", "anthropic", {
            "api_key": anthropic_key,
            "model": os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"),
            "max_tokens": int(os.getenv("ANTHROPIC_MAX_TOKENS", "500"))
        })
    
    # Set default provider
    default_provider = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
    if default_provider in llm_manager.get_available_providers():
        llm_manager.set_default_provider(default_provider)
    
    return llm_manager

# Usage
llm_manager = create_llm_manager_from_env()
```

## Model-Specific Considerations

### OpenAI o3-mini

The o3-mini model has specific requirements:

```python
o3_mini_config = {
    "api_key": "sk-your-key",
    "model": "o3-mini",
    "max_completion_tokens": 500,  # Use this instead of max_tokens
    # Do not include temperature - o3-mini doesn't support it
}

# When using o3-mini, responses may be more deterministic
agent = await create_react_agent(
    name="O3Agent",
    llm_provider="openai",
    llm_config=o3_mini_config,
    max_reasoning_steps=15  # o3-mini may need more steps for complex reasoning
)
```

### Claude Models Comparison

```python
# Choose Claude model based on your needs:

# Claude Opus - Most capable, best for complex reasoning
claude_opus = {
    "model": "claude-3-opus-20240229",
    "max_tokens": 1000,  # Higher token limit for complex responses
    "temperature": 0.7
}

# Claude Sonnet - Balanced performance and speed
claude_sonnet = {
    "model": "claude-3-sonnet-20240229", 
    "max_tokens": 500,
    "temperature": 0.7
}

# Claude Haiku - Fastest, best for quick responses
claude_haiku = {
    "model": "claude-3-haiku-20240307",
    "max_tokens": 300,
    "temperature": 0.5
}
```

## Error Handling and Retry Logic

### Built-in Retry Logic

```python
# LLM providers include automatic retry logic
llm_config = {
    "api_key": "sk-your-key",
    "model": "gpt-4",
    "max_retries": 3,          # Retry failed requests
    "retry_delay": 1.0,        # Delay between retries (seconds)
    "timeout": 30.0,           # Request timeout
    "backoff_factor": 2.0      # Exponential backoff multiplier
}
```

### Custom Error Handling

```python
from aitor.llm import LLMError, RateLimitError, APIError

class RobustAgent(ReactAgent):
    """Agent with enhanced error handling."""
    
    async def solve(self, problem: str) -> str:
        """Solve with enhanced error handling."""
        
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                return await super().solve(problem)
                
            except RateLimitError as e:
                if attempt < max_attempts - 1:
                    wait_time = 60 * (2 ** attempt)  # Exponential backoff
                    print(f"Rate limited, waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                    continue
                raise e
                
            except APIError as e:
                if "model_overloaded" in str(e) and attempt < max_attempts - 1:
                    print(f"Model overloaded, switching provider...")
                    # Switch to backup provider
                    self.set_llm_provider("claude-sonnet")
                    continue
                raise e
                
            except Exception as e:
                if attempt < max_attempts - 1:
                    print(f"Unexpected error: {e}, retrying...")
                    await asyncio.sleep(5)
                    continue
                raise e
        
        raise Exception("All retry attempts failed")
```

## Performance Optimization

### Connection Pooling

```python
# Configure connection pooling for high-throughput applications
openai_config = {
    "api_key": "sk-your-key",
    "model": "gpt-4",
    "connection_pool_size": 10,    # Number of concurrent connections
    "connection_pool_maxsize": 20,  # Maximum pool size
    "keep_alive": True,             # Keep connections alive
    "timeout": 30.0
}
```

### Caching Responses

```python
from functools import lru_cache
import hashlib

class CachedLLMProvider(LLMProvider):
    """LLM provider with response caching."""
    
    def __init__(self, base_provider: LLMProvider, cache_size: int = 128):
        self.base_provider = base_provider
        self.cache_size = cache_size
        
        # Create cached version of generate_response
        self._cached_generate = lru_cache(maxsize=cache_size)(
            self._generate_response_cacheable
        )
    
    def _generate_response_cacheable(self, messages_hash: str, **kwargs) -> str:
        """Cacheable version that uses hash instead of messages directly."""
        # This is a simplified version - you'd need to store original messages
        return self.base_provider.generate_response(self.messages, **kwargs)
    
    async def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response with caching."""
        
        # Create hash of messages for cache key
        messages_str = str(messages) + str(sorted(kwargs.items()))
        messages_hash = hashlib.md5(messages_str.encode()).hexdigest()
        
        # Store messages for actual API call
        self.messages = messages
        
        try:
            return self._cached_generate(messages_hash, **kwargs)
        except Exception:
            # If caching fails, call directly
            return await self.base_provider.generate_response(messages, **kwargs)

# Usage
base_provider = openai_provider
cached_provider = CachedLLMProvider(base_provider, cache_size=256)
```

## Monitoring and Logging

### Usage Tracking

```python
class MonitoredLLMManager(LLMManager):
    """LLM manager with usage tracking."""
    
    def __init__(self):
        super().__init__()
        self.usage_stats = {
            "requests": 0,
            "tokens_used": 0,
            "errors": 0,
            "by_provider": {}
        }
    
    async def generate_response(self, provider_name: str, messages: List[Dict], **kwargs) -> str:
        """Generate response with usage tracking."""
        
        start_time = time.time()
        
        try:
            response = await super().generate_response(provider_name, messages, **kwargs)
            
            # Track successful request
            self.usage_stats["requests"] += 1
            self.usage_stats["by_provider"].setdefault(provider_name, {"requests": 0, "errors": 0})
            self.usage_stats["by_provider"][provider_name]["requests"] += 1
            
            # Estimate token usage (rough approximation)
            estimated_tokens = sum(len(msg["content"].split()) for msg in messages) * 1.3
            self.usage_stats["tokens_used"] += estimated_tokens
            
            execution_time = time.time() - start_time
            logger.info(f"LLM request to {provider_name} completed in {execution_time:.2f}s")
            
            return response
            
        except Exception as e:
            # Track error
            self.usage_stats["errors"] += 1
            self.usage_stats["by_provider"].setdefault(provider_name, {"requests": 0, "errors": 0})
            self.usage_stats["by_provider"][provider_name]["errors"] += 1
            
            logger.error(f"LLM request to {provider_name} failed: {e}")
            raise
    
    def get_usage_report(self) -> Dict[str, Any]:
        """Get detailed usage report."""
        return {
            "total_requests": self.usage_stats["requests"],
            "total_errors": self.usage_stats["errors"],
            "success_rate": (self.usage_stats["requests"] - self.usage_stats["errors"]) / max(1, self.usage_stats["requests"]),
            "estimated_tokens": self.usage_stats["tokens_used"],
            "by_provider": self.usage_stats["by_provider"]
        }

# Usage
monitored_llm = MonitoredLLMManager()
# ... add providers ...

# Get usage report
report = monitored_llm.get_usage_report()
print(f"Success rate: {report['success_rate']:.2%}")
print(f"Total tokens: {report['estimated_tokens']}")
```

## Best Practices

### 1. Provider Selection
- Use GPT-4 for complex reasoning and code generation
- Use Claude for longer context and nuanced conversations  
- Use o3-mini for cost-effective applications
- Use Claude Haiku for quick responses

### 2. Configuration Management
- Store API keys in environment variables
- Use different models for different tasks
- Configure appropriate timeout values
- Set up failover between providers

### 3. Error Handling
- Implement retry logic with exponential backoff
- Handle rate limiting gracefully
- Have backup providers configured
- Log errors for monitoring

### 4. Performance
- Use connection pooling for high-throughput applications
- Cache responses when appropriate
- Monitor token usage and costs
- Choose the right model for each task

### 5. Security
- Never hardcode API keys
- Use secure credential storage
- Rotate API keys regularly
- Monitor for unusual usage patterns

The LLM integration system provides a robust foundation for building intelligent agents that can work with multiple language models while handling errors gracefully and optimizing for performance.