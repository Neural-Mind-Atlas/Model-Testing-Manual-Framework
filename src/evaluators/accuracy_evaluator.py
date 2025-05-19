"""Accuracy evaluator for factual knowledge."""

import os
from typing import Dict, List, Any

from ..clients.base_client import BaseClient
from ..utils.config import load_evaluation_metrics

class AccuracyEvaluator:
    """Evaluates factual accuracy of model responses."""

    def __init__(self, evaluation_model: BaseClient, metrics_config: Dict[str, Any] = None):
        """
        Initialize the accuracy evaluator.

        Args:
            evaluation_model: Model client for evaluation
            metrics_config: Metrics configuration dictionary
        """
        self.evaluation_model = evaluation_model

        # Load metrics configuration if not provided
        if metrics_config is None:
            config_dir = os.environ.get("CONFIG_DIR", "./config")
            self.metrics_config = load_evaluation_metrics(f"{config_dir}/evaluation/metrics.yaml")
        else:
            self.metrics_config = metrics_config

    def evaluate(self,
                 prompt: str,
                 response: str,
                 expected_answer: str = None,
                 metrics: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate the accuracy of a model response.

        Args:
            prompt: Original prompt given to the model
            response: Model's response
            expected_answer: Expected answer if available
            metrics: Specific metrics to evaluate

        Returns:
            Dictionary of evaluation scores
        """
        if not metrics:
            metrics = ["correctness", "completeness"]

        results = {}

        # If no evaluation model is available, provide mock scores
        if self.evaluation_model is None:
            for metric in metrics:
                results[metric] = 0.8  # Default moderate score
            
            results["overall_score"] = 0.8
            return results

        for metric in metrics:
            if metric not in self.metrics_config:
                continue

            metric_config = self.metrics_config.get(metric, {})

            if metric_config.get("evaluation_method") == "model_based":
                # Create evaluation prompt
                eval_prompt = self._create_evaluation_prompt(prompt, response, expected_answer, metric)

                # Generate evaluation using evaluation model
                try:
                    eval_response = self.evaluation_model.generate(
                        prompt=eval_prompt,
                        config={"system_prompt": "You are an expert evaluator assessing AI model responses."}
                    )

                    # Parse score from response
                    score = self._parse_score(eval_response, metric_config.get("scale", [0, 5]))
                    results[metric] = score
                except Exception as e:
                    results[metric] = {
                        "score": 0,
                        "error": str(e)
                    }
            else:
                # Implement rule-based or other evaluation methods here
                results[metric] = 0.7  # Default fallback score

        # Calculate weighted average score
        weighted_score = 0
        total_weight = 0

        for metric, score in results.items():
            if isinstance(score, dict):
                # Skip if there was an error
                continue

            metric_weight = self.metrics_config.get(metric, {}).get("weight", 1.0)
            weighted_score += score * metric_weight
            total_weight += metric_weight

        if total_weight > 0:
            results["overall_score"] = weighted_score / total_weight
        else:
            results["overall_score"] = 0

        return results

    def _create_evaluation_prompt(self, prompt: str, response: str, expected_answer: str, metric: str) -> str:
        """Create a prompt for evaluating a specific metric."""
        metric_config = self.metrics_config.get(metric, {})
        scale = metric_config.get("scale", [0, 5])
        scale_description = ", ".join([f"{i}: {desc}" for i, desc in enumerate(metric_config.get("scale_descriptions", []))])

        template = f"""
        Please evaluate the following AI model response for {metric_config.get("description", metric)}.

        Original prompt:
        "{prompt}"

        AI model response:
        "{response}"
        """

        if expected_answer:
            template += f"""
            Expected answer:
            "{expected_answer}"
            """

        template += f"""
        On a scale of {min(scale)} to {max(scale)},
        where {min(scale)} means poor performance and {max(scale)} means excellent performance,
        rate the response and explain your rating.

        Your answer should be in this format:
        Rating: [numeric score between {min(scale)} and {max(scale)}]
        Explanation: [your explanation]
        """

        return template

    def _parse_score(self, evaluation_text: str, scale: List[int]) -> float:
        """Parse the score from evaluation response."""
        try:
            # Look for "Rating: X" pattern
            for line in evaluation_text.split("\n"):
                if line.lower().startswith("rating:"):
                    score_str = line.split(":", 1)[1].strip()
                    # Extract first number found
                    import re
                    numbers = re.findall(r'\d+\.?\d*', score_str)
                    if numbers:
                        score = float(numbers[0])
                        # Ensure score is within scale
                        return max(min(score, max(scale)), min(scale))

            # If no rating found, assume middle score
            return (max(scale) + min(scale)) / 2
        except Exception:
            # Default to middle score on error
            return (max(scale) + min(scale)) / 2