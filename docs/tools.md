# Tools Development Guide

This guide covers creating custom tools for Aitor agents, from basic functions to advanced patterns for multi-tenant applications.

## Overview

Tools are the primary way agents interact with external systems and perform actions. The Aitor framework provides a flexible tool system that supports:

- Automatic parameter validation
- Async execution with timeouts
- JSON schema generation for LLM integration
- Thread-safe execution
- Comprehensive error handling

## Basic Tool Creation

### Using the @tool Decorator

The simplest way to create a tool is using the `@tool` decorator:

```python
from aitor.tools import tool

@tool(
    name="calculator",
    description="Perform mathematical calculations",
    timeout=10.0
)
def calculate(expression: str) -> float:
    """Calculate the result of a mathematical expression."""
    try:
        # Safe evaluation with restricted context
        allowed_names = {
            k: v for k, v in math.__dict__.items() 
            if not k.startswith("__")
        }
        allowed_names.update({
            "abs": abs, "round": round, "min": min, "max": max
        })
        
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return float(result)
    except Exception as e:
        raise ValueError(f"Calculation error: {e}")
```

### Tool Parameters

The `@tool` decorator accepts several parameters:

- **`name`**: Tool identifier (defaults to function name)
- **`description`**: Human-readable description for LLMs
- **`parameters`**: Custom JSON schema (auto-generated if not provided)
- **`timeout`**: Execution timeout in seconds (default: 30.0)
- **`async_execution`**: Whether to run in thread pool (default: True)

### Parameter Validation

Tools automatically validate parameters based on function signatures:

```python
@tool(name="text_analyzer", description="Analyze text properties")
def analyze_text(
    text: str,
    include_stats: bool = True,
    max_length: Optional[int] = None
) -> dict:
    """Analyze various properties of text."""
    if max_length and len(text) > max_length:
        text = text[:max_length]
    
    result = {
        "length": len(text),
        "word_count": len(text.split())
    }
    
    if include_stats:
        result.update({
            "sentence_count": text.count('.') + text.count('!') + text.count('?'),
            "avg_word_length": sum(len(word) for word in text.split()) / len(text.split()) if text.split() else 0
        })
    
    return result
```

## Advanced Tool Patterns

### 1. Factory Pattern for Context-Specific Tools

For applications serving multiple tenants or contexts, create tools dynamically:

```python
def create_database_tool(tenant_id: str, connection_string: str):
    """Factory function to create tenant-specific database tool."""
    
    @tool(
        name=f"database_query_{tenant_id}",
        description=f"Query database for tenant {tenant_id}"
    )
    def query_database(query: str, limit: int = 100) -> dict:
        """Execute database query with pre-configured tenant context."""
        # tenant_id and connection_string are captured from closure
        return {
            "tenant": tenant_id,
            "query": query,
            "results": execute_query(connection_string, tenant_id, query, limit)
        }
    
    return query_database

# Usage
tenant_db_tool = create_database_tool("acme_corp", "postgresql://...")
await agent.register_tool(tenant_db_tool)
```

### 2. Using functools.partial for Parameter Binding

Bind static parameters using `functools.partial`:

```python
from functools import partial
from aitor.tools import tool

def api_request(base_url: str, api_key: str, endpoint: str, method: str = "GET") -> dict:
    """Base API request function with all parameters."""
    url = f"{base_url}/{endpoint}"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    response = requests.request(method, url, headers=headers)
    return response.json()

# Create tenant-specific API tool using partial
tenant_api = partial(
    api_request,
    base_url="https://tenant123.api.com",
    api_key="secret_key_123"
)

@tool(name="tenant_api", description="Make API calls for current tenant")
def api_tool(endpoint: str, method: str = "GET") -> dict:
    return tenant_api(endpoint=endpoint, method=method)
```

### 3. Custom Tool Classes with Context

For complex tools that need to maintain state:

```python
from aitor.tools import Tool
from typing import Any, Dict

class TenantToolFactory:
    """Tool factory with pre-configured tenant parameters."""
    
    def __init__(self, tenant_id: str, api_key: str, base_url: str):
        self.tenant_id = tenant_id
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def search_documents(self, query: str, limit: int = 10) -> dict:
        """Search documents for this tenant."""
        response = self.session.get(
            f"{self.base_url}/search",
            params={"q": query, "limit": limit, "tenant": self.tenant_id}
        )
        return response.json()
    
    def create_document(self, title: str, content: str, tags: List[str] = None) -> dict:
        """Create a new document."""
        payload = {
            "title": title,
            "content": content,
            "tags": tags or [],
            "tenant": self.tenant_id
        }
        response = self.session.post(f"{self.base_url}/documents", json=payload)
        return response.json()
    
    def create_tools(self) -> List[Tool]:
        """Create all tools for this tenant."""
        return [
            Tool(
                name=f"search_docs_{self.tenant_id}",
                func=self.search_documents,
                description=f"Search documents for tenant {self.tenant_id}",
                timeout=30.0
            ),
            Tool(
                name=f"create_doc_{self.tenant_id}",
                func=self.create_document,
                description=f"Create document for tenant {self.tenant_id}",
                timeout=30.0
            )
        ]

# Usage
tenant_factory = TenantToolFactory("acme_corp", "api_key", "https://api.example.com")
tools = tenant_factory.create_tools()

for tool in tools:
    await agent.register_tool(tool)
```

### 4. Environment-Based Configuration

Use dataclasses for structured configuration:

```python
from dataclasses import dataclass
from aitor.tools import tool

@dataclass
class ServiceConfig:
    tenant_id: str
    database_url: str
    api_key: str
    storage_bucket: str
    redis_url: str

def create_service_tools(config: ServiceConfig) -> List[Tool]:
    """Create all tools for a specific service configuration."""
    
    @tool(name="fetch_user_data", description="Fetch user data from database")
    def fetch_user_data(user_id: str) -> dict:
        # config is captured in closure
        return query_database(config.database_url, config.tenant_id, user_id)
    
    @tool(name="cache_data", description="Cache data in Redis")
    def cache_data(key: str, value: str, ttl: int = 3600) -> bool:
        return redis_set(config.redis_url, f"{config.tenant_id}:{key}", value, ttl)
    
    @tool(name="upload_file", description="Upload file to storage")
    def upload_file(file_name: str, content: bytes) -> str:
        return upload_to_s3(config.storage_bucket, f"{config.tenant_id}/{file_name}", content)
    
    @tool(name="send_notification", description="Send notification via API")
    def send_notification(message: str, recipient: str) -> bool:
        return send_via_api(config.api_key, config.tenant_id, message, recipient)
    
    return [fetch_user_data, cache_data, upload_file, send_notification]

# Usage
config = ServiceConfig(
    tenant_id="acme_corp",
    database_url="postgresql://...",
    api_key="secret_123",
    storage_bucket="acme-files",
    redis_url="redis://..."
)

tools = create_service_tools(config)
```

## Built-in Tools

Aitor includes several pre-built tools for common tasks:

### Web Search Tools

```python
from tools import google_web_search, tavily_web_search

# Configure environment variables:
# GOOGLE_SEARCH_API_KEY, GOOGLE_SEARCH_ENGINE_ID, TAVILY_API_KEY

await agent.register_tool(google_web_search)
await agent.register_tool(tavily_web_search)

# Agent can now search the web
response = await agent.solve("Search for recent AI developments")
```

### Slack Integration Tools

```python
from tools import (
    slack_send_message, 
    slack_send_file, 
    slack_send_dm,
    slack_get_channels
)

# Configure SLACK_BOT_TOKEN environment variable

await agent.register_tool(slack_send_message)
await agent.register_tool(slack_send_file)

# Agent can now interact with Slack
response = await agent.solve("Send a status update to #general channel")
```

## Tool Registry Management

### Avoiding Registry Conflicts

When creating multiple agents, ensure each has its own tool registry:

```python
from aitor.tools import ToolRegistry

# Method 1: Explicit tool registry per agent
def create_agent_with_tools(tenant_id: str):
    tool_registry = ToolRegistry()
    
    @tool(name="lookup_info", description=f"Look up info for {tenant_id}")
    def lookup_info(query: str) -> dict:
        return get_tenant_info(tenant_id, query)
    
    agent = ReactAgent(
        name=f"Agent_{tenant_id}",
        llm_manager=llm_manager,
        tool_registry=tool_registry
    )
    
    await agent.register_tool(lookup_info)
    return agent

# Method 2: Using unique tool names
def create_unique_tools(tenant_id: str):
    @tool(
        name=f"lookup_info_{tenant_id}",  # Unique name per tenant
        description=f"Look up info for {tenant_id}"
    )
    def lookup_info(query: str) -> dict:
        return get_tenant_info(tenant_id, query)
    
    return lookup_info
```

### Tool Registration Best Practices

```python
class CustomerServiceAgent:
    """Complete example of tenant-specific agent with tools."""
    
    def __init__(self, tenant_id: str, config: TenantConfig):
        self.tenant_id = tenant_id
        self.config = config
    
    def create_tools(self) -> List[Tool]:
        """Create all tenant-specific tools."""
        
        @tool(name="lookup_customer", description="Look up customer information")
        def lookup_customer(customer_id: str) -> dict:
            return query_customer_db(
                self.config.database_url, 
                self.tenant_id, 
                customer_id
            )
        
        @tool(name="create_ticket", description="Create support ticket")
        def create_ticket(title: str, description: str, priority: str = "medium") -> str:
            return create_support_ticket(
                self.config.support_api_key,
                self.tenant_id,
                title,
                description,
                priority
            )
        
        @tool(name="get_billing_info", description="Get customer billing information")
        def get_billing_info(customer_id: str) -> dict:
            return get_billing_data(
                self.config.billing_api_url,
                self.tenant_id,
                customer_id
            )
        
        return [lookup_customer, create_ticket, get_billing_info]
    
    async def create_agent(self, llm_manager: LLMManager) -> ReactAgent:
        """Create the complete agent with tenant-specific tools."""
        tools = self.create_tools()
        
        agent = await create_react_agent(
            name=f"CustomerService_{self.tenant_id}",
            llm_manager=llm_manager,
            tools=tools,
            agent_role="customer service representative",
            additional_instructions=f"You are helping customers for {self.tenant_id}"
        )
        
        return agent

# Usage
agent_factory = CustomerServiceAgent("acme_corp", tenant_config)
agent = await agent_factory.create_agent(llm_manager)
```

## Error Handling in Tools

### Comprehensive Error Handling

```python
@tool(name="file_processor", description="Process uploaded files")
def process_file(file_path: str, operation: str) -> dict:
    """Process a file with specified operation."""
    try:
        # Validate file exists and is readable
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"Cannot read file: {file_path}")
        
        # Validate operation
        valid_operations = ["analyze", "convert", "validate"]
        if operation not in valid_operations:
            raise ValueError(f"Invalid operation: {operation}. Must be one of {valid_operations}")
        
        # Perform operation
        with open(file_path, 'r') as f:
            content = f.read()
        
        if operation == "analyze":
            result = analyze_file_content(content)
        elif operation == "convert":
            result = convert_file_format(content)
        elif operation == "validate":
            result = validate_file_structure(content)
        
        return {
            "success": True,
            "operation": operation,
            "file_path": file_path,
            "result": result
        }
    
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File processing failed: {e}")
    except PermissionError as e:
        raise PermissionError(f"File access denied: {e}")
    except ValueError as e:
        raise ValueError(f"Invalid parameters: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error processing file: {e}")
```

### Timeout and Resource Management

```python
@tool(
    name="external_api_call",
    description="Call external API with proper resource management",
    timeout=60.0  # 60 second timeout
)
def call_external_api(endpoint: str, params: dict = None) -> dict:
    """Call external API with timeout and retry logic."""
    import requests
    from requests.adapters import HTTPAdapter
    from requests.packages.urllib3.util.retry import Retry
    
    # Create session with retry strategy
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    try:
        response = session.get(
            endpoint,
            params=params or {},
            timeout=(10, 30)  # (connect timeout, read timeout)
        )
        response.raise_for_status()
        
        return {
            "success": True,
            "status_code": response.status_code,
            "data": response.json()
        }
    
    except requests.exceptions.Timeout:
        raise TimeoutError(f"API call to {endpoint} timed out")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"API call failed: {e}")
    finally:
        session.close()
```

## Testing Tools

### Unit Testing Tools

```python
import pytest
from aitor.tools import Tool

@pytest.mark.asyncio
async def test_calculator_tool():
    # Create tool instance
    calc_tool = Tool(
        name="test_calculator",
        func=calculate,
        description="Test calculator",
        timeout=5.0
    )
    
    # Test successful calculation
    result = await calc_tool.execute(expression="2 + 3 * 4")
    assert result.success is True
    assert result.result == 14.0
    
    # Test error handling
    result = await calc_tool.execute(expression="invalid")
    assert result.success is False
    assert "error" in result.error.lower()

@pytest.mark.asyncio
async def test_tool_registry():
    from aitor.tools import ToolRegistry
    
    registry = ToolRegistry()
    
    # Test tool registration
    calc_tool = create_calculator_tool()
    await registry.register(calc_tool)
    
    assert "calculator" in registry
    assert len(registry) == 1
    
    # Test tool execution through registry
    result = await registry.execute_tool("calculator", expression="5 * 5")
    assert result.success is True
    assert result.result == 25.0
```

### Integration Testing with Mock LLM

```python
from aitor import create_react_agent
from aitor.llm import LLMManager

@pytest.mark.asyncio
async def test_agent_with_tools():
    # Setup mock LLM
    llm_manager = LLMManager()
    llm_manager.add_provider("mock", "mock", {
        "responses": {
            "calculator": "ACT: calculator(expression='2+2')",
            "default": "THINK: Let me solve this step by step."
        }
    })
    
    # Create agent with test tools
    agent = await create_react_agent(
        name="TestAgent",
        llm_provider="mock",
        llm_config={},
        tools=[create_calculator_tool()]
    )
    
    # Test agent response
    response = await agent.solve("What is 2 + 2?")
    assert "4" in response or "four" in response.lower()
    
    await agent.shutdown()
```

## Performance Optimization

### Async Tool Execution

Tools run asynchronously by default, but you can optimize for specific use cases:

```python
# CPU-intensive tool - run in thread pool (default)
@tool(name="heavy_computation", async_execution=True)
def heavy_computation(data: list) -> dict:
    # This will run in thread pool to avoid blocking event loop
    result = perform_heavy_calculation(data)
    return {"result": result}

# I/O bound tool - run directly in event loop
@tool(name="async_api_call", async_execution=False)
async def async_api_call(url: str) -> dict:
    # This runs directly in event loop
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
```

### Connection Pooling

For tools that make frequent external calls:

```python
import aiohttp
from typing import Optional

class APIToolFactory:
    """Factory for API tools with connection pooling."""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session with connection pooling."""
        if self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(
                limit=100,  # Total connection pool size
                limit_per_host=30,  # Connections per host
                ttl_dns_cache=300,  # DNS cache TTL
                use_dns_cache=True,
            )
            
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
        
        return self.session
    
    @tool(name="api_search", description="Search via API")
    async def search(self, query: str, limit: int = 10) -> dict:
        """Search using pooled HTTP connection."""
        session = await self.get_session()
        
        async with session.get(
            f"{self.base_url}/search",
            params={"q": query, "limit": limit}
        ) as response:
            return await response.json()
    
    async def cleanup(self):
        """Close HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()

# Usage with proper cleanup
api_factory = APIToolFactory("https://api.example.com", "api-key")
search_tool = api_factory.search

await agent.register_tool(search_tool)

# Don't forget cleanup
await api_factory.cleanup()
```

This comprehensive guide covers all aspects of tool development in the Aitor framework, from basic creation to advanced patterns for enterprise applications.