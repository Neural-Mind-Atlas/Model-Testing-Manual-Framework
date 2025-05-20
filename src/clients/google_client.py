# src/clients/google_client.py
import os
import time
import google.generativeai as genai
from typing import Dict, Optional, Any
from .base_client import BaseClient
from src.utils.tokenizers import count_tokens

class GoogleClient(BaseClient):
    """Client for Google API."""

    def __init__(self, api_key: str = None, model_config: Dict[str, Any] = None):
        """Initialize the Google client."""
        super().__init__(api_key=api_key, model_config=model_config)
        self.name = "google"
        
        # Use provided API key or get from environment
        if not api_key:
            api_key = os.environ.get("GOOGLE_API_KEY")
        
        # Initialize the Google API
        genai.configure(api_key=api_key)

    def generate(self, prompt: str, config: Optional[Dict[str, Any]] = None) -> str:
        """Generate a response using Google API."""
        if not config:
            config = {}
            
        # Merge defaults with passed config
        use_config = self.defaults.copy() if hasattr(self, 'defaults') and self.defaults else {}
        if config:
            use_config.update(config)
        
        try:
            # Record start time
            start_time = time.time()
            
            # Get model
            model = genai.GenerativeModel(self.version)
            
            # Call the API
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": use_config.get("temperature", 0.7),
                    "top_p": use_config.get("top_p", 0.95),
                    "max_output_tokens": use_config.get("max_output_tokens", 8192)
                }
            )
            
            # Record completion time
            end_time = time.time()
            
            # Estimate token usage since Google doesn't always provide it
            prompt_tokens = count_tokens(prompt, self.model_name)
            completion = response.text
            completion_tokens = count_tokens(completion, self.model_name)
            
            # Save usage data
            self.last_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
            
            # Save timing data
            self.last_timing = {
                "total_time": end_time - start_time,
                "time_to_first_token": None  # Google doesn't provide this
            }
            
            # Calculate cost
            self.last_cost = self.calculate_cost(
                self.last_usage["prompt_tokens"],
                self.last_usage["completion_tokens"]
            )
            
            return response.text
            
        except Exception as e:
            raise Exception(f"Google API error: {str(e)}")