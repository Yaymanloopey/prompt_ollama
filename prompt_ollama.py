#!/usr/bin/env python3
"""
Template script for prompting a local Ollama instance.
Requires: pip install requests
"""

import requests
import json
# from typing import Optional
import sys
import os

# Load configuration from config.json
def load_config():
    """Load configuration from config.json file."""
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: config.json not found at {config_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: config.json is not valid JSON", file=sys.stderr)
        sys.exit(1)

# UPDATE THE json.config to determine url, model & generation parameters
config = load_config()
OLLAMA_BASE_URL = config["ollama"]["base_url"]
DEFAULT_MODEL = config["ollama"]["default_model"]
DEFAULT_TIMEOUT = config["ollama"]["timeout"]
DEFAULT_TEMPERATURE = config["generation"]["temperature"]
DEFAULT_TOP_P = config["generation"]["top_p"]


def prompt_ollama(
    prompt: str,
    model: str = None,
    stream: bool = False,
    temperature: float = None,
    top_p: float = None,
) -> str:
    """
    Send a prompt to a local Ollama instance and get a response.

    Args:
        prompt: The input prompt text
        model: Model name to use (default: from config)
        stream: Whether to stream the response (default: False)
        temperature: Sampling temperature (0-1, higher = more creative)
        top_p: Nucleus sampling parameter

    Returns:
        The model's response text
    """
    model = model or DEFAULT_MODEL
    temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE
    top_p = top_p if top_p is not None else DEFAULT_TOP_P
    
    url = f"{OLLAMA_BASE_URL}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "temperature": temperature,
        "top_p": top_p,
    }

    try:
        response = requests.post(url, json=payload, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()

        if stream:
            # Handle streaming response
            full_response = ""
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        full_response += data["response"]
                        print(data["response"], end="", flush=True)
            print()  # New line after streaming
            return full_response
        else:
            # Handle non-streaming response
            data = response.json()
            return data.get("response", "")

    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to Ollama. Is it running on localhost:11434?", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("Error: Request timed out. The model may be taking too long to generate.", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def prompt_ollama_chat(
    messages: list[dict],
    model: str = None,
    stream: bool = False,
    temperature: float = None,
) -> str:
    """
    Send a multi-turn conversation to Ollama using the chat API.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
                  e.g., [{"role": "user", "content": "Hello"}]
        model: Model name to use (default: from config)
        stream: Whether to stream the response
        temperature: Sampling temperature (default: from config)

    Returns:
        The model's response text
    """
    model = model or DEFAULT_MODEL
    temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE
    
    url = f"{OLLAMA_BASE_URL}/api/chat"

    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "temperature": temperature,
    }

    try:
        response = requests.post(url, json=payload, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()

        if stream:
            full_response = ""
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "message" in data:
                        content = data["message"].get("content", "")
                        full_response += content
                        print(content, end="", flush=True)
            print()
            return full_response
        else:
            data = response.json()
            return data.get("message", {}).get("content", "")

    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to Ollama. Is it running on localhost:11434?", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("Error: Request timed out.", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def list_available_models() -> list[str]:
    """
    Get list of available models from Ollama.

    Returns:
        List of model names
    """
    url = f"{OLLAMA_BASE_URL}/api/tags"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        models = [model["name"] for model in data.get("models", [])]
        return models
    except Exception as e:
        print(f"Error fetching models: {e}", file=sys.stderr)
        return []


def main():
    """Example usage of the Ollama prompt functions."""
    
    # Check available models
    print("Available models:")
    models = list_available_models()
    for model in models:
        print(f"  - {model}")

    if not models:
        print("No models found. Make sure Ollama is running and has models installed.")
        return

    # Example 1: Simple prompt
    # print("\n=== Simple Prompt Example ===")
    # response = prompt_ollama("What is the capital of France?", stream=False)
    # print(response)

    # Example 2: Streaming prompt
    # print("\n=== Streaming Prompt Example ===")
    # print("Response: ", end="")
    # prompt_ollama("Tell me a short joke.", stream=True)

    # Example 3: Chat-style conversation
    # print("\n=== Chat Example ===")
    # messages = [
    #     {"role": "user", "content": "Hello, how are you?"},
    # ]
    # response = prompt_ollama_chat(messages, stream=False)
    # print(response)


if __name__ == "__main__":
    main()
