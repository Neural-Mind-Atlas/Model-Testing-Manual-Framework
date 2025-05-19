"""Cost analysis for model evaluation."""

import os
import yaml
import json
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
from .base_reporter import BaseReporter
import logging

logger = logging.getLogger(__name__)

class CostAnalyzer(BaseReporter):
    """Analyzes cost metrics and cost-effectiveness of models."""

    def __init__(self, results_dir: str = "./results"):
        """
        Initialize the cost analyzer.

        Args:
            results_dir: Directory containing results
        """
        super().__init__()
        self.name = "cost_analyzer"
        self.results_dir = results_dir
        self.comparisons_dir = os.path.join(results_dir, "comparisons")

        # Ensure output directory exists
        os.makedirs(self.comparisons_dir, exist_ok=True)

    def generate_report(self, results: Dict[str, Any], output_path: str) -> bool:
        """
        Generate a cost analysis report from test results.
        
        Args:
            results: Dictionary containing test results
            output_path: Path where the report should be saved
            
        Returns:
            bool: True if report generation was successful, False otherwise
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Generate cost-effectiveness report
            cost_report = self.generate_cost_effectiveness_report(results)
            
            # Write to output file
            with open(output_path, 'w') as file:
                yaml.dump(cost_report, file, default_flow_style=False, sort_keys=False)
                
            logger.info(f"Cost analysis report generated at {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error generating cost analysis report: {e}", exc_info=True)
            return False

    def generate_cost_effectiveness_report(self,
                                    results: Dict[str, Any],
                                    performance_metrics: List[str] = None,
                                    output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a cost-effectiveness report comparing all models.

        Args:
            results: Dictionary containing test results
            performance_metrics: List of metrics to use for performance evaluation
            output_file: Output file path, or None to use default

        Returns:
            Dictionary containing the cost analysis report
        """
        # Default performance metrics if not provided
        if performance_metrics is None:
            performance_metrics = ["overall_score"]

        # Default output file
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.comparisons_dir, f"cost_effectiveness_{timestamp}.yaml")

        # Extract cost and performance data
        cost_data = {}

        for model_id, result in results.items():
            # Skip models with errors
            if "error" in result:
                continue

            # Initialize model data
            model_metrics = {
                "total_cost_usd": 0,
                "average_cost_per_request": 0,
                "total_tokens": 0,
                "overall_score": 0
            }
            
            # Extract cost metrics
            if "cost" in result:
                model_metrics["total_cost_usd"] = result["cost"]
                model_metrics["average_cost_per_request"] = result["cost"]  # For a single request
            
            # Extract token usage
            if "usage" in result:
                usage = result["usage"]
                if "total_tokens" in usage:
                    model_metrics["total_tokens"] = usage["total_tokens"]
            
            # Extract performance metrics
            for metric in performance_metrics:
                if metric == "overall_score" and "overall_score" in result:
                    model_metrics[metric] = result["overall_score"]
                elif "metrics" in result and metric in result["metrics"]:
                    # Handle nested metrics
                    metric_value = result["metrics"][metric]
                    if isinstance(metric_value, dict) and "overall_score" in metric_value:
                        model_metrics[metric] = metric_value["overall_score"]
                    else:
                        model_metrics[metric] = metric_value
            
            # Calculate derived metrics
            if model_metrics["total_tokens"] > 0:
                model_metrics["cost_per_1k_tokens"] = (model_metrics["total_cost_usd"] * 1000) / model_metrics["total_tokens"]
            else:
                model_metrics["cost_per_1k_tokens"] = 0
            
            # Calculate cost-effectiveness ratios
            for metric in performance_metrics:
                if metric in model_metrics and model_metrics["average_cost_per_request"] > 0:
                    ratio_name = f"{metric}_per_dollar"
                    model_metrics[ratio_name] = model_metrics[metric] / model_metrics["average_cost_per_request"]
            
            cost_data[model_id] = model_metrics

        # Create report structure
        report = {
            "timestamp": self._get_current_timestamp(),
            "cost_metrics": {
                "models": list(cost_data.keys()),
                "metrics": list(next(iter(cost_data.values())).keys()) if cost_data else [],
                "data": cost_data
            },
            "rankings": {}
        }

        # Add rankings for each metric
        if cost_data:
            all_metrics = list(next(iter(cost_data.values())).keys())
            
            for metric in all_metrics:
                try:
                    # Create a list of (model_name, metric_value) tuples
                    metric_values = []
                    for model_name, model_metrics in cost_data.items():
                        if metric in model_metrics:
                            metric_values.append((model_name, model_metrics[metric]))

                    if not metric_values:
                        continue

                    # Determine if higher or lower is better
                    reverse = not (metric.startswith("cost_") or 
                                  "cost" in metric or 
                                  metric.endswith("_cost_usd"))

                    # Sort by metric value
                    metric_values.sort(key=lambda x: x[1], reverse=reverse)

                    # Create ranking
                    report["rankings"][metric] = [
                        {"rank": i+1, "model": model, "value": value}
                        for i, (model, value) in enumerate(metric_values)
                    ]
                except Exception as e:
                    logger.error(f"Error ranking models for metric {metric}: {e}", exc_info=True)

        # Write report to YAML file if output_file is provided
        if output_file:
            with open(output_file, 'w') as f:
                yaml.dump(report, f, default_flow_style=False, sort_keys=False)

        return report

    def calculate_quality_adjusted_cost(self,
                                results: Dict[str, Any],
                                quality_metric: str = "overall_score",
                                output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate quality-adjusted cost for all models.

        Args:
            results: Dictionary containing test results
            quality_metric: Metric to use for quality
            output_file: Output file path, or None to use default

        Returns:
            Dictionary containing the quality-adjusted cost analysis
        """
        # Default output file
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.comparisons_dir, f"quality_adjusted_cost_{quality_metric}_{timestamp}.yaml")

        # Extract cost and quality data
        qa_data = {}

        for model_id, result in results.items():
            # Skip models with errors
            if "error" in result:
                continue

            # Extract cost
            cost = 0
            if "cost" in result:
                cost = result["cost"]
            
            # Extract quality score
            quality = 0
            if quality_metric == "overall_score" and "overall_score" in result:
                quality = result["overall_score"]
            elif "metrics" in result and quality_metric in result["metrics"]:
                metric_value = result["metrics"][quality_metric]
                if isinstance(metric_value, dict) and "overall_score" in metric_value:
                    quality = metric_value["overall_score"]
                else:
                    quality = metric_value
            
            # Calculate quality-adjusted metrics
            if quality > 0:
                qa_data[model_id] = {
                    "cost": cost,
                    "quality": quality,
                    "cost_per_quality_point": cost / quality if quality > 0 else float('inf'),
                    "quality_points_per_dollar": quality / cost if cost > 0 else 0
                }

        # Create report structure
        report = {
            "timestamp": self._get_current_timestamp(),
            "quality_metric": quality_metric,
            "models": list(qa_data.keys()),
            "data": qa_data,
            "rankings": {}
        }

        # Rank by cost per quality point (lower is better)
        cost_per_quality = [(model, data["cost_per_quality_point"])
                          for model, data in qa_data.items()]
        cost_per_quality.sort(key=lambda x: x[1])

        report["rankings"]["cost_per_quality_point"] = [
            {"rank": i+1, "model": model, "value": value}
            for i, (model, value) in enumerate(cost_per_quality)
        ]

        # Rank by quality points per dollar (higher is better)
        quality_per_dollar = [(model, data["quality_points_per_dollar"])
                            for model, data in qa_data.items()]
        quality_per_dollar.sort(key=lambda x: x[1], reverse=True)

        report["rankings"]["quality_points_per_dollar"] = [
            {"rank": i+1, "model": model, "value": value}
            for i, (model, value) in enumerate(quality_per_dollar)
        ]

        # Write report to YAML file if output_file is provided
        if output_file:
            with open(output_file, 'w') as f:
                yaml.dump(report, f, default_flow_style=False, sort_keys=False)

        return report