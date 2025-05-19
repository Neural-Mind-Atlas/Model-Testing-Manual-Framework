"""Context utilization evaluator."""

import os
from typing import Dict, List, Any

from ..clients.base_client import BaseModelClient
from ..utils.config import load_evaluation_metrics

class ContextEvaluator:
    """Evaluates context utilization in model responses."""

    def __init__(self, evaluation_model: BaseModelClient, metrics_config: Dict[str, Any] = None):
        """
        Initialize the context evaluator.

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

    async def evaluate(self,
                 prompt: str,
                 response: str,
                 context: str,
                 context_questions: List[str] = None,
                 metrics: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate the context utilization in a model response.

        Args:
            prompt: Original prompt given to the model
            response: Model's response
            context: Context provided to the model
            context_questions: Specific questions to check if they can be answered from the context
            metrics: Specific metrics to evaluate

        Returns:
            Dictionary of evaluation scores
        """
        if not metrics:
            metrics = ["relevance", "accuracy", "completeness"]

        results = {}

        for metric in metrics:
            if metric not in self.metrics_config["metrics"]:
                continue

            metric_config = self.metrics_config["metrics"][metric]

            if metric_config["evaluation_method"] == "model_based":
                # Create evaluation prompt
                eval_prompt = self._create_evaluation_prompt(
                    prompt, response, context, context_questions, metric
                )

                # Generate evaluation using evaluation model
                eval_response = await self.evaluation_model.generate_response(
                    prompt=eval_prompt,
                    system_prompt="You are an expert evaluator assessing AI model context utilization.",
                    temperature=0.1
                )

                # Parse score from response
                try:
                    score = self._parse_score(eval_response["text"], metric_config["scale"])
                    results[metric] = score
                except Exception as e:
                    results[metric] = {
                        "score": 0,
                        "error": str(e)
                    }
            else:
                # Implement rule-based or other evaluation methods here
                results[metric] = 0

        # Calculate weighted average score
        weighted_score = 0
        total_weight = 0

        for metric, score in results.items():
            if isinstance(score, dict):
                # Skip if there was an error
                continue

            metric_weight = self.metrics_config["metrics"][metric].get("weight", 1.0)
            weighted_score += score * metric_weight
            total_weight += metric_weight

        if total_weight > 0:
            results["overall_score"] = weighted_score / total_weight
        else:
            results["overall_score"] = 0

        return results

    def _create_evaluation_prompt(self, prompt: str, response: str,
                                  context: str, context_questions: List[str],
                                  metric: str) -> str:
        """Create a prompt for evaluating context utilization."""
        metric_config = self.metrics_config["metrics"][metric]
        scale_description = ", ".join([f"{i}: {desc}" for i, desc in enumerate(metric_config.get("scale_descriptions", []))])

        template = f"""
        Please evaluate the following AI model response for {metric_config["description"]}.

        Original prompt:
        "{prompt}"

        Context provided to the model:
        "{context}"

        AI model response:
        "{response}"
        """

        if context_questions:
            questions_str = "\n".join([f"- {question}" for question in context_questions])
            template += f"""
            Specific questions to check if they are answered from the context:
            {questions_str}
            """

        template += f"""
        On a scale of {min(metric_config["scale"])} to {max(metric_config["scale"])},
        where {min(metric_config["scale"])} means poor context utilization and {max(metric_config["scale"])} means excellent context utilization,
        rate the response and explain your rating.

        Consider:
        1. Did the response use information from the context accurately?
        2. Did the response include key information from the context relevant to the prompt?
        3. Did the response avoid adding information not present in the context?

        Your answer should be in this format:
        Rating: [numeric score between {min(metric_config["scale"])} and {max(metric_config["scale"])}]
        Explanation: [your explanation]
        Context information used: [list specific information from the context that was used]
        Context information missed: [list relevant information from the context that was not used]
        Information added outside of context: [list information in the response not found in the context or "None"]
        """

        return template

    def _parse_score(self, evaluation_text: str, scale: List[int]) -> int:
        """Parse the score from evaluation response."""
        try:
            # Look for "Rating: X" pattern
            for line in evaluation_text.split("\n"):
                if line.lower().startswith("rating:"):
                    score_str = line.split(":", 1)[1].strip()
                    # Extract first number found
                    import re
                    numbers = re.findall(r'\d+', score_str)
                    if numbers:
                        score = int(numbers[0])
                        # Ensure score is within scale
                        return max(min(score, max(scale)), min(scale))

            # If no rating found, assume minimum score
            return min(scale)
        except Exception:
            # Default to minimum score on error
            return min(scale)