# src/clients/openai_client.py
import os
import time
import openai
from typing import Dict, Optional, Any
from .base_client import BaseClient

class OpenAIClient(BaseClient):
    """Client for OpenAI API."""

    def __init__(self, api_key: str = None, model_config: Dict[str, Any] = None):
        """Initialize the OpenAI client."""
        super().__init__(api_key=api_key, model_config=model_config)
        self.name = "openai"
        
        # Use provided API key or get from environment
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
        
        # Initialize the OpenAI client
        self.client = openai.OpenAI(api_key=api_key)
        
        # Set organization ID if available
        if model_config and "org_id_env" in model_config:
            org_id = os.environ.get(model_config["org_id_env"])
            if org_id:
                self.client.organization = org_id

    def generate(self, prompt: str, config: Optional[Dict[str, Any]] = None) -> str:
        """Generate a response using OpenAI API."""
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
            response = self.client.chat.completions.create(
                model=self.version,
                messages=[{"role": "user", "content": prompt}],
                temperature=use_config.get("temperature", 0.7),
                max_tokens=use_config.get("max_tokens", 4096),
                top_p=use_config.get("top_p", 1.0)
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
                "time_to_first_token": None  # OpenAI doesn't provide this
            }
            
            # Calculate cost
            self.last_cost = self.calculate_cost(
                self.last_usage["prompt_tokens"],
                self.last_usage["completion_tokens"]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")