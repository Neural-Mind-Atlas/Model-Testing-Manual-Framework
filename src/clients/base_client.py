# src/clients/base_client.py
"""Base client class for interacting with LLM APIs."""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

class BaseClient(ABC):
    """Abstract base class for all model API clients."""

    def __init__(self, api_key: str = None, model_config: Dict[str, Any] = None):
        """
        Initialize the client with model configuration.

        Args:
            api_key: API key for authentication
            model_config: Dictionary containing model configuration
        """
        self.name = "base"
        self.api_key = api_key
        self.model_config = model_config

        if model_config:
            self.model_name = model_config.get("name", "unknown")
            self.display_name = model_config.get("display_name", "Unknown Model")
            self.version = model_config.get("version", "1.0")
            self.max_tokens = model_config.get("max_tokens", 4096)
            self.context_window = model_config.get("context_window", 4096)
            self.defaults = model_config.get("defaults", {})
            self.cost_config = model_config.get("cost", {})
        else:
            # Set default values if no config provided
            self.model_name = "unknown"
            self.display_name = "Unknown Model"
            self.version = "1.0"
            self.max_tokens = 4096
            self.context_window = 4096
            self.defaults = {}
            self.cost_config = {}

    @abstractmethod
    def generate(self, prompt: str, config: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a response from the model.

        Args:
            prompt: User prompt/input text
            config: Additional configuration parameters

        Returns:
            Generated response text
        """
        pass

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate the cost of a request.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD
        """
        input_cost = (input_tokens / 1000) * self.cost_config.get("input_per_1k", 0)
        output_cost = (output_tokens / 1000) * self.cost_config.get("output_per_1k", 0)
        return input_cost + output_cost

    def _create_timing_info(self, start_time: float, first_token_time: Optional[float] = None) -> Dict[str, float]:
        """
        Create timing information for a request.

        Args:
            start_time: Request start time
            first_token_time: Time when first token was received

        Returns:
            Dictionary of timing metrics
        """
        end_time = time.time()
        timing = {
            "total_time": end_time - start_time,
            "time_to_first_token": None if not first_token_time else first_token_time - start_time
        }
        return timing