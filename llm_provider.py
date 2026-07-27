"""
LLM Provider Module
Unified interface for interacting with different LLM providers (DeepSeek, Ollama, OpenAI)
"""

import os
from typing import Optional
from dotenv import load_dotenv
import requests
import json

# Load environment variables
load_dotenv()


class DeepSeek:
    """
    DeepSeek LLM Provider - Cost-effective LLM API client
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        """
        Initialize DeepSeek client
        
        Args:
            api_key: DeepSeek API key (defaults to DEEPSEEK_API_KEY env var)
            api_url: DeepSeek API URL (defaults to DEEPSEEK_API_URL env var)
            model: Model name to use
            temperature: Temperature for response generation (0-2)
            max_tokens: Maximum tokens in response
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.api_url = api_url or os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables or parameters")
    
    def chat(self, message: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Send a chat message to DeepSeek
        
        Args:
            message: User message
            system_prompt: Optional system prompt to guide the model
            **kwargs: Additional parameters to override defaults
            
        Returns:
            Model response
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": message})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
        
        except requests.exceptions.RequestException as e:
            raise Exception(f"DeepSeek API Error: {str(e)}")
        except (KeyError, IndexError) as e:
            raise Exception(f"Unexpected response format: {str(e)}")
    
    def stream(self, message: str, system_prompt: Optional[str] = None, **kwargs):
        """
        Stream a chat message from DeepSeek
        
        Args:
            message: User message
            system_prompt: Optional system prompt
            **kwargs: Additional parameters
            
        Yields:
            Streamed response chunks
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": message})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": True
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers=headers,
                json=payload,
                stream=True,
                timeout=30
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith("data: "):
                        data = line_str[6:]
                        if data != "[DONE]":
                            try:
                                chunk = json.loads(data)
                                content = chunk["choices"][0]["delta"].get("content", "")
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                continue
        
        except requests.exceptions.RequestException as e:
            raise Exception(f"DeepSeek Streaming Error: {str(e)}")


class Ollama:
    """
    Ollama LLM Provider - Local LLM (Free, runs locally)
    """
    
    def __init__(
        self,
        api_url: Optional[str] = None,
        model: str = "mistral",
        temperature: float = 0.7,
    ):
        """
        Initialize Ollama client
        
        Args:
            api_url: Ollama API URL (defaults to OLLAMA_API_URL env var)
            model: Model name to use
            temperature: Temperature for response generation
        """
        self.api_url = api_url or os.getenv("OLLAMA_API_URL", "http://localhost:11434")
        self.model = model
        self.temperature = temperature
    
    def chat(self, message: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Send a chat message to Ollama
        
        Args:
            message: User message
            system_prompt: Optional system prompt
            **kwargs: Additional parameters
            
        Returns:
            Model response
        """
        # Combine system prompt and message
        full_prompt = ""
        if system_prompt:
            full_prompt = f"{system_prompt}\n"
        full_prompt += message
        
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "temperature": kwargs.get("temperature", self.temperature)
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/api/generate",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result["response"]
        
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama Connection Error: {str(e)}. Make sure Ollama is running.")


class LLMFactory:
    """
    Factory for creating LLM provider instances
    """
    
    @staticmethod
    def create_deepseek(**kwargs) -> DeepSeek:
        """Create a DeepSeek client"""
        return DeepSeek(**kwargs)
    
    @staticmethod
    def create_ollama(**kwargs) -> Ollama:
        """Create an Ollama client"""
        return Ollama(**kwargs)
