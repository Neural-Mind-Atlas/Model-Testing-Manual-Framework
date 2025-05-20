# src/clients/mistral_client.py
import os
import time
from mistralai.client import MistralClient as MistralAPIClient
from mistralai.models.chat_completion import ChatMessage
from typing import Dict, Optional, Any
from .base_client import BaseClient

class MistralClient(BaseClient):
    """Client for Mistral API."""

    def __init__(self, api_key: str = None, model_config: Dict[str, Any] = None):
        """Initialize the Mistral client."""
        super().__init__(api_key=api_key, model_config=model_config)
        self.name = "mistral"
        
        # Use provided API key or get from environment
        if not api_key:
            api_key = os.environ.get("MISTRAL_API_KEY")
        
        # Initialize the Mistral client
        self.client = MistralAPIClient(api_key=api_key)

    def generate(self, prompt: str, config: Optional[Dict[str, Any]] = None) -> str:
        """Generate a response using Mistral API."""
        if not config:
            config = {}
            
        # Merge defaults with passed config
        use_config = self.defaults.copy() if hasattr(self, 'defaults') and self.defaults else {}
        if config:
            use_config.update(config)
        
        try:
            # Record start time
            start_time = time.time()
            
            # Call the API
            messages = [ChatMessage(role="user", content=prompt)]
            
            # Add system message if provided
            if use_config.get("system_prompt"):
                messages.insert(0, ChatMessage(role="system", content=use_config["system_prompt"]))
            
            response = self.client.chat(
                model=self.version,
                messages=messages,
                temperature=use_config.get("temperature", 0.7),
                max_tokens=use_config.get("max_tokens", 4096),
                top_p=use_config.get("top_p", 0.9)
            )
            
            # Record completion time
            end_time = time.time()
            
            # Save usage data
            self.last_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            # Save timing data
            self.last_timing = {
                "total_time": end_time - start_time,
                "time_to_first_token": None  # Mistral doesn't provide this
            }
            
            # Calculate cost
            self.last_cost = self.calculate_cost(
                self.last_usage["prompt_tokens"],
                self.last_usage["completion_tokens"]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"Mistral API error: {str(e)}")