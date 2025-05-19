"""Model recommendation engine."""

import os
import yaml
import json
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
from .base_reporter import BaseReporter

logger = logging.getLogger(__name__)

class RecommendationEngine(BaseReporter):
    """Generates model recommendations based on specific criteria."""

    def __init__(self, results_dir: str = "./results"):
        """
        Initialize the recommendation engine.

        Args:
            results_dir: Directory containing results
        """
        super().__init__()
        self.name = "recommendations"
        self.results_dir = results_dir
        self.comparisons_dir = os.path.join(results_dir, "comparisons")

        # Ensure output directory exists
        os.makedirs(self.comparisons_dir, exist_ok=True)

    def generate_report(self, results: Dict[str, Any], output_path: str) -> bool:
        """
        Generate a recommendations report from test results.
        
        Args:
            results: Dictionary containing test results
            output_path: Path where the report should be saved
            
        Returns:
            bool: True if report generation was successful, False otherwise
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Generate standard recommendations
            recommendations = self.generate_standard_recommendations(results, output_path)
            
            logger.info(f"Recommendations report generated at {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error generating recommendations report: {e}", exc_info=True)
            return False

    def generate_recommendations(self,
                           results: Dict[str, Any],
                           scenarios: List[Dict[str, Any]],
                           output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate model recommendations for different scenarios.

        Args:
            results: Dictionary containing test results
            scenarios: List of scenario dictionaries with weights
            output_file: Output file path, or None to use default

        Returns:
            Dictionary containing recommendations
        """
        # Default output file
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.comparisons_dir, f"recommendations_{timestamp}.yaml")

        # Generate recommendations for each scenario
        recommendations = {
            "timestamp": self._get_current_timestamp(),
            "scenarios": []
        }

        for scenario in scenarios:
            scenario_name = scenario.get("name", "unnamed_scenario")
            description = scenario.get("description", "")
            weights = scenario.get("weights", {})

            # Score each model based on weighted metrics
            scores = {}

            for model_id, result in results.items():
                # Skip models with errors
                if "error" in result:
                    continue
                    
                score = 0

                # Apply weights to metrics
                for metric, weight in weights.items():
                    # Handle different metric locations
                    value = None
                    
                    # Check if it's a top-level metric
                    if metric in result:
                        value = result[metric]
                    # Check if it's in the metrics dictionary
                    elif "metrics" in result and metric in result["metrics"]:
                        metric_value = result["metrics"][metric]
                        if isinstance(metric_value, dict) and "overall_score" in metric_value:
                            value = metric_value["overall_score"]
                        else:
                            value = metric_value
                    # Check if it's a specific test category
                    elif "metrics" in result:
                        for category, category_data in result["metrics"].items():
                            if metric == category and isinstance(category_data, dict) and "overall_score" in category_data:
                                value = category_data["overall_score"]
                    
                    if value is not None:
                        # Invert cost-related metrics (lower is better)
                        if (metric.startswith("cost_") or 
                            metric.endswith("_cost") or 
                            "cost" in metric.lower()):
                            # Find max value to normalize
                            max_val = 0
                            for m_id, m_result in results.items():
                                if "error" in m_result:
                                    continue
                                m_value = 0
                                if metric in m_result:
                                    m_value = m_result[metric]
                                elif "metrics" in m_result and metric in m_result["metrics"]:
                                    m_metric = m_result["metrics"][metric]
                                    if isinstance(m_metric, dict) and "overall_score" in m_metric:
                                        m_value = m_metric["overall_score"]
                                    else:
                                        m_value = m_metric
                                if m_value > max_val:
                                    max_val = m_value

                            if max_val > 0:
                                normalized = 1 - (value / max_val)
                                score += normalized * weight
                        else:
                            # For other metrics, higher is better
                            # Find max value to normalize
                            max_val = 0
                            for m_id, m_result in results.items():
                                if "error" in m_result:
                                    continue
                                m_value = 0
                                if metric in m_result:
                                    m_value = m_result[metric]
                                elif "metrics" in m_result and metric in m_result["metrics"]:
                                    m_metric = m_result["metrics"][metric]
                                    if isinstance(m_metric, dict) and "overall_score" in m_metric:
                                        m_value = m_metric["overall_score"]
                                    else:
                                        m_value = m_metric
                                if m_value > max_val:
                                    max_val = m_value

                            if max_val > 0:
                                normalized = value / max_val
                                score += normalized * weight

                scores[model_id] = score

            # Sort models by score
            ranked_models = sorted([(model, score) for model, score in scores.items()],
                                key=lambda x: x[1], reverse=True)

            scenario_recommendation = {
                "name": scenario_name,
                "description": description,
                "weights": weights,
                "recommended_models": [
                    {"rank": i+1, "model": model, "score": score}
                    for i, (model, score) in enumerate(ranked_models)
                ]
            }

            # Add top 3 models with explanations
            if ranked_models:
                scenario_recommendation["top_recommendations"] = []

                for i, (model, score) in enumerate(ranked_models[:min(3, len(ranked_models))]):
                    # Generate explanation based on strengths
                    strengths = []

                    for metric, weight in weights.items():
                        metric_value = None
                        # Check different locations for the metric
                        if metric in results[model]:
                            metric_value = results[model][metric]
                        elif "metrics" in results[model] and metric in results[model]["metrics"]:
                            m_value = results[model]["metrics"][metric]
                            if isinstance(m_value, dict) and "overall_score" in m_value:
                                metric_value = m_value["overall_score"]
                            else:
                                metric_value = m_value
                                
                        if metric_value is not None:
                            # Check if this is a top performer in this metric
                            all_values = []
                            for m_id, m_result in results.items():
                                if "error" in m_result:
                                    continue
                                m_value = None
                                if metric in m_result:
                                    m_value = m_result[metric]
                                elif "metrics" in m_result and metric in m_result["metrics"]:
                                    metric_data = m_result["metrics"][metric]
                                    if isinstance(metric_data, dict) and "overall_score" in metric_data:
                                        m_value = metric_data["overall_score"]
                                    else:
                                        m_value = metric_data
                                if m_value is not None:
                                    all_values.append(m_value)

                            if all_values:
                                is_cost_metric = (metric.startswith("cost_") or 
                                                metric.endswith("_cost") or 
                                                "cost" in metric.lower())
                                all_values.sort(reverse=not is_cost_metric)
                                
                                if (is_cost_metric and metric_value == all_values[0]) or \
                                   (not is_cost_metric and metric_value == all_values[0]):
                                    metric_display = metric.replace("_", " ").title()
                                    if is_cost_metric:
                                        strengths.append(f"Lowest {metric_display}")
                                    else:
                                        strengths.append(f"Best {metric_display}")
                                elif len(all_values) > 1 and metric_value == all_values[1]:
                                    metric_display = metric.replace("_", " ").title()
                                    strengths.append(f"Strong {metric_display}")

                    scenario_recommendation["top_recommendations"].append({
                        "rank": i+1,
                        "model": model,
                        "score": score,
                        "strengths": strengths
                    })

            recommendations["scenarios"].append(scenario_recommendation)

        # Write recommendations to YAML file if output_file is provided
        if output_file:
            with open(output_file, 'w') as f:
                yaml.dump(recommendations, f, default_flow_style=False, sort_keys=False)

        return recommendations

    def generate_standard_recommendations(self, results: Dict[str, Any], output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate standard recommendations for common use cases.

        Args:
            results: Dictionary containing test results
            output_file: Output file path, or None to use default

        Returns:
            Dictionary containing standard recommendations
        """
        # Define standard scenarios
        standard_scenarios = [
            {
                "name": "best_overall",
                "description": "Best overall performer",
                "weights": {
                    "overall_score": 1.0,
                    "reasoning": 1.0,
                    "factual": 1.0,
                    "hallucination": 1.0,
                    "instruction": 1.0,
                    "context": 1.0,
                    "creative": 0.8,
                    "code": 0.7,
                    "ppt_writing": 1.2,
                    "meta_prompting": 1.0,
                    "image_prompts": 0.9
                }
            },
            {
                "name": "most_cost_effective",
                "description": "Most cost-effective model",
                "weights": {
                    "cost": 2.0,
                    "overall_score": 1.0,
                    "reasoning": 0.8,
                    "factual": 0.8,
                    "hallucination": 0.8,
                    "instruction": 0.8,
                    "context": 0.8,
                    "ppt_writing": 1.0
                }
            },
            {
                "name": "best_for_ppt",
                "description": "Best model for PPT generation",
                "weights": {
                    "ppt_writing": 2.0,
                    "context": 1.2,
                    "hallucination": 1.2,
                    "instruction": 1.0,
                    "creative": 0.9,
                    "image_prompts": 1.2
                }
            },
            {
                "name": "fastest",
                "description": "Fastest model with acceptable quality",
                "weights": {
                    "overall_score": 0.8,
                    "response_time": 2.0,
                    "reasoning": 0.7,
                    "factual": 0.7,
                    "instruction": 0.7,
                    "ppt_writing": 0.8
                }
            },
            {
                "name": "best_reasoning",
                "description": "Best model for complex reasoning",
                "weights": {
                    "reasoning": 2.0,
                    "factual": 1.2,
                    "context": 1.0,
                    "hallucination": 1.0
                }
            }
        ]

        return self.generate_recommendations(results, standard_scenarios, output_file)