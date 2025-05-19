"""Efficiency evaluator for token usage and response time."""

from typing import Dict, Any

class EfficiencyEvaluator:
    """Evaluates efficiency metrics of model responses."""

    def evaluate(self,
                response_data: Dict[str, Any],
                quality_score: float,
                cost_per_token: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Evaluate the efficiency of a model response.

        Args:
            response_data: Response data including usage and timing information
            quality_score: Quality score from other evaluators
            cost_per_token: Cost per token information

        Returns:
            Dictionary of efficiency metrics
        """
        results = {}

        # Extract usage information
        usage = response_data.get("usage", {})
        timing = response_data.get("timing", {})
        cost = response_data.get("cost", 0)

        # Calculate token efficiency
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0) or (prompt_tokens + completion_tokens)

        if total_tokens > 0 and quality_score > 0:
            # Calculate token efficiency (quality per 1000 tokens)
            results["token_efficiency"] = (quality_score * 1000) / total_tokens

            # Calculate quality per dollar
            if cost > 0:
                results["cost_per_quality_point"] = cost / quality_score
                results["quality_points_per_dollar"] = quality_score / cost
        else:
            results["token_efficiency"] = 0
            results["cost_per_quality_point"] = 0
            results["quality_points_per_dollar"] = 0

        # Response time metrics
        results["response_time"] = timing.get("total_time", 0)
        results["time_to_first_token"] = timing.get("time_to_first_token", 0)

        if completion_tokens > 0 and timing.get("total_time", 0) > 0:
            # Calculate tokens per second
            results["tokens_per_second"] = completion_tokens / timing.get("total_time", 1)
        else:
            results["tokens_per_second"] = 0

        # Calculate cost metrics
        results["cost"] = cost
        if total_tokens > 0:
            results["cost_per_1k_tokens"] = (cost * 1000) / total_tokens
        else:
            results["cost_per_1k_tokens"] = 0

        return results