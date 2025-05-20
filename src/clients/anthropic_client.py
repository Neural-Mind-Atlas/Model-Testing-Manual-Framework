# src/clients/anthropic_client.py
import os
import time
import anthropic
from typing import Dict, Optional, Any
from .base_client import BaseClient

class AnthropicClient(BaseClient):
    """Client for Anthropic API."""

    def __init__(self, api_key: str = None, model_config: Dict[str, Any] = None):
        """Initialize the Anthropic client."""
        super().__init__(api_key=api_key, model_config=model_config)
        self.name = "anthropic"
        
        # Use provided API key or get from environment
        if not api_key:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        
        # Initialize the Anthropic client
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str, config: Optional[Dict[str, Any]] = None) -> str:
        """Generate a response using Anthropic API."""
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
            response = self.client.messages.create(
                model=self.version,
                max_tokens=use_config.get("max_output_tokens", 4096),
                temperature=use_config.get("temperature", 0.7),
                messages=[
                    {"role": "user", "content": prompt}
                ],
                system=use_config.get("system_prompt", "You are a helpful AI assistant.")
            )
            
            # Record completion time
            end_time = time.time()
            
            # Save usage data
            self.last_usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }
            
            # Save timing data
            self.last_timing = {
                "total_time": end_time - start_time,
                "time_to_first_token": None  # Anthropic doesn't provide this
            }
            
            # Calculate cost
            self.last_cost = self.calculate_cost(
                self.last_usage["prompt_tokens"],
                self.last_usage["completion_tokens"]
            )
            
            # Extract text content from the response
            content = ""
            for block in response.content:
                if block.type == "text":
                    content += block.text
            
            return content
            
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")