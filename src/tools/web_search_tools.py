"""
Web search tools for ReAct agents.
Provides Google Custom Search and Tavily search capabilities.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

import requests

from aitor.tools import tool


@tool(
    name="google_web_search",
    description="Search the web using Google Custom Search API. Returns search results with titles, snippets, and URLs.",
    timeout=30.0,
)
def google_web_search(
    query: str,
    num_results: Optional[int] = 10,
    site_search: Optional[str] = None,
    date_restrict: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Search the web using Google Custom Search API.

    Args:
        query: The search query string
        num_results: Number of results to return (1-10, default: 10)
        site_search: Restrict search to specific site (e.g., 'reddit.com')
        date_restrict: Date restriction (e.g., 'd1' for past day, 'w1' for past week, 'm1' for past month)

    Returns:
        Dictionary containing search results with titles, snippets, URLs, and metadata

    Raises:
        ValueError: If API key or search engine ID not configured
        requests.RequestException: If API request fails
    """
    # Get API credentials from environment
    api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

    if not api_key:
        raise ValueError("GOOGLE_SEARCH_API_KEY environment variable not set")
    if not search_engine_id:
        raise ValueError("GOOGLE_SEARCH_ENGINE_ID environment variable not set")

    # Validate parameters
    if not query.strip():
        raise ValueError("Query cannot be empty")

    num_results = max(1, min(num_results or 10, 10))

    # Build API request
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": search_engine_id,
        "q": query,
        "num": num_results,
    }

    # Add optional parameters
    if site_search:
        params["siteSearch"] = site_search
    if date_restrict:
        params["dateRestrict"] = date_restrict

    try:
        # Make API request
        response = requests.get(url, params=params, timeout=25)
        response.raise_for_status()

        data = response.json()

        # Extract search results
        items = data.get("items", [])
        search_info = data.get("searchInformation", {})

        results = []
        for item in items:
            result = {
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "url": item.get("link", ""),
                "display_url": item.get("displayLink", ""),
            }

            # Add additional metadata if available
            if "pagemap" in item:
                pagemap = item["pagemap"]
                if "metatags" in pagemap and pagemap["metatags"]:
                    meta = pagemap["metatags"][0]
                    result["description"] = meta.get(
                        "og:description", meta.get("description", "")
                    )
                    result["image"] = meta.get("og:image", "")

            results.append(result)

        return {
            "query": query,
            "results": results,
            "total_results": search_info.get("totalResults", "0"),
            "search_time": search_info.get("searchTime", 0),
            "num_results": len(results),
            "timestamp": time.time(),
        }

    except requests.RequestException as e:
        raise requests.RequestException(f"Google Search API request failed: {str(e)}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse Google Search API response: {str(e)}")


@tool(
    name="tavily_web_search",
    description="Search the web using Tavily API. Provides comprehensive search results with content extraction and AI-powered summaries.",
    timeout=30.0,
)
def tavily_web_search(
    query: str,
    search_depth: Optional[str] = "basic",
    max_results: Optional[int] = 10,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    include_answer: Optional[bool] = True,
    include_raw_content: Optional[bool] = False,
) -> Dict[str, Any]:
    """
    Search the web using Tavily API with AI-powered content extraction.

    Args:
        query: The search query string
        search_depth: Search depth ("basic" or "advanced", default: "basic")
        max_results: Maximum number of results to return (1-20, default: 10)
        include_domains: List of domains to include in search
        exclude_domains: List of domains to exclude from search
        include_answer: Whether to include AI-generated answer (default: True)
        include_raw_content: Whether to include raw content extraction (default: False)

    Returns:
        Dictionary containing search results, AI answer, and metadata

    Raises:
        ValueError: If API key not configured or invalid parameters
        requests.RequestException: If API request fails
    """
    # Get API key from environment
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable not set")

    # Validate parameters
    if not query.strip():
        raise ValueError("Query cannot be empty")

    if search_depth not in ["basic", "advanced"]:
        search_depth = "basic"

    max_results = max(1, min(max_results or 10, 20))

    # Build API request
    url = "https://api.tavily.com/search"
    headers = {
        "Content-Type": "application/json",
    }

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": search_depth,
        "max_results": max_results,
        "include_answer": include_answer,
        "include_raw_content": include_raw_content,
    }

    # Add optional domain filters
    if include_domains:
        payload["include_domains"] = include_domains
    if exclude_domains:
        payload["exclude_domains"] = exclude_domains

    try:
        # Make API request
        response = requests.post(url, headers=headers, json=payload, timeout=25)
        response.raise_for_status()

        data = response.json()

        # Extract and format results
        results = []
        for item in data.get("results", []):
            result = {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "score": item.get("score", 0.0),
                "published_date": item.get("published_date", ""),
            }

            # Add raw content if requested
            if include_raw_content and "raw_content" in item:
                result["raw_content"] = item["raw_content"]

            results.append(result)

        response_data = {
            "query": query,
            "results": results,
            "num_results": len(results),
            "search_depth": search_depth,
            "timestamp": time.time(),
        }

        # Add AI-generated answer if available
        if include_answer and "answer" in data:
            response_data["answer"] = data["answer"]

        # Add additional metadata
        if "follow_up_questions" in data:
            response_data["follow_up_questions"] = data["follow_up_questions"]

        if "images" in data:
            response_data["images"] = data["images"]

        return response_data

    except requests.RequestException as e:
        raise requests.RequestException(f"Tavily API request failed: {str(e)}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse Tavily API response: {str(e)}")
