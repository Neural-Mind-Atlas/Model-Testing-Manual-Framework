# src/clients/others.py
import os
import time
import cohere
from typing import Dict, Optional, Any
from .base_client import BaseClient
from src.utils.tokenizers import count_tokens

class CohereClient(BaseClient):
    """Client for Cohere API."""

    def __init__(self, api_key: str = None, model_config: Dict[str, Any] = None):
        """Initialize the Cohere client."""
        super().__init__(api_key=api_key, model_config=model_config)
        self.name = "cohere"
        
        # Use provided API key or get from environment
        if not api_key:
            api_key = os.environ.get("COHERE_API_KEY")
        
        # Initialize the Cohere client
        self.client = cohere.Client(api_key)

    def generate(self, prompt: str, config: Optional[Dict[str, Any]] = None) -> str:
        """Generate a response using Cohere API."""
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
            response = self.client.chat(
                message=prompt,
                model=self.version,
                temperature=use_config.get("temperature", 0.7),
                max_tokens=use_config.get("max_tokens", 4096),
                p=use_config.get("top_p", 0.9)
            )
            
            # Record completion time
            end_time = time.time()
            
            # Extract usage information if available
            tokens_used = getattr(response, 'token_count', None)
            if tokens_used:
                prompt_tokens = tokens_used.get('prompt_tokens', 0)
                completion_tokens = tokens_used.get('response_tokens', 0)
                total_tokens = tokens_used.get('total_tokens', prompt_tokens + completion_tokens)
            else:
                # Estimate token usage
                prompt_tokens = count_tokens(prompt, self.model_name)
                completion_tokens = count_tokens(response.text, self.model_name)
                total_tokens = prompt_tokens + completion_tokens
            
            self.last_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
            
            # Save timing data
            self.last_timing = {
                "total_time": end_time - start_time,
                "time_to_first_token": None  # Cohere doesn't provide this
            }
            
            # Calculate cost
            self.last_cost = self.calculate_cost(
                self.last_usage["prompt_tokens"],
                self.last_usage["completion_tokens"]
            )
            
            return response.text
            
        except Exception as e:
            raise Exception(f"Cohere API error: {str(e)}")

# # Keep DatabricksClient implementation as is
# class DatabricksClient(BaseClient):
#     """Client for Databricks API."""

#     def __init__(self, api_key: str = None, model_config: Dict[str, Any] = None):
#         """Initialize the Databricks client."""
#         super().__init__(api_key=api_key, model_config=model_config)
#         self.name = "databricks"

#     def generate(self, prompt: str, config: Optional[Dict[str, Any]] = None) -> str:
#         """Generate a response using Databricks API."""
#         # In a real implementation, this would call the Databricks API
#         # For testing purposes, we'll return a mock response
#         return f"This is a mock response from Databricks for prompt: {prompt[:50]}..."