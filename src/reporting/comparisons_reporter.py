"""Model comparison matrix generator."""

import os
import yaml
import json
from typing import Dict, List, Any, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class ModelComparison:
    """Generates comparative analyses across models."""

    def __init__(self, results_dir: str = "./results"):
        """
        Initialize the model comparison tool.

        Args:
            results_dir: Directory containing results
        """
        self.results_dir = results_dir
        self.yaml_dir = os.path.join(results_dir, "yaml")
        self.comparisons_dir = os.path.join(results_dir, "comparisons")

        # Ensure output directory exists
        os.makedirs(self.comparisons_dir, exist_ok=True)

    def generate_comparison(self, results: Dict[str, Any], output_file: Optional[str] = None) -> str:
        """
        Generate a comparison matrix for the provided results.

        Args:
            results: Dictionary of model results
            output_file: Output file path, or None to use default

        Returns:
            Path to the generated comparison file
        """
        # Default output file
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.comparisons_dir, f"comparison_{timestamp}.yaml")

        # Create comparison matrix
        comparison = {
            "timestamp": self._get_current_timestamp(),
            "models_compared": list(results.keys()),
            "metrics_compared": self._get_common_metrics(results),
            "comparison_data": {},
            "rankings": {}
        }

        # Extract comparison data
        for model_id, model_data in results.items():
            comparison["comparison_data"][model_id] = self._extract_metrics(model_data)

        # Add rankings for each metric
        common_metrics = self._get_common_metrics(results)
        for metric in common_metrics:
            try:
                # Create a list of (model_name, metric_value) tuples
                metric_values = []
                for model_name, model_data in results.items():
                    # Handle metrics in different locations
                    value = None
                    if "aggregate_metrics" in model_data and metric in model_data["aggregate_metrics"]:
                        value = model_data["aggregate_metrics"][metric]
                    elif metric in model_data:
                        value = model_data[metric]
                    elif "metrics" in model_data and metric in model_data["metrics"]:
                        value = model_data["metrics"][metric]
                    
                    if value is not None:
                        metric_values.append((model_name, value))

                # Determine if higher or lower is better
                reverse = not (metric.startswith("cost_") or 
                              metric.endswith("_cost") or 
                              metric.startswith("time_") or 
                              "error" in metric.lower())

                # Sort by metric value
                metric_values.sort(key=lambda x: x[1], reverse=reverse)

                # Create ranking
                comparison["rankings"][metric] = [
                    {"rank": i+1, "model": model, "value": value}
                    for i, (model, value) in enumerate(metric_values)
                ]
            except Exception as e:
                logger.error(f"Error ranking models for metric {metric}: {e}", exc_info=True)

        # Write comparison to YAML file
        with open(output_file, 'w') as f:
            yaml.dump(comparison, f, default_flow_style=False, sort_keys=False)

        return output_file

    def generate_comparison_matrix(self,
                             metrics: List[str],
                             output_file: Optional[str] = None) -> str:
        """
        Generate a comparison matrix for specified metrics across all models.

        Args:
            metrics: List of metrics to compare
            output_file: Output file path, or None to use default

        Returns:
            Path to the generated comparison file
        """
        # Default output file
        if output_file is None:
            metrics_str = "_".join(metrics[:3]) + (f"_plus_{len(metrics)-3}" if len(metrics) > 3 else "")
            output_file = os.path.join(self.comparisons_dir, f"comparison_{metrics_str}.yaml")

        # Get all YAML result files
        yaml_files = []
        if os.path.exists(self.yaml_dir):
            yaml_files = [f for f in os.listdir(self.yaml_dir) if f.endswith('.yaml')]
        else:
            # Try to find YAML files directly in results directory
            for root, _, files in os.walk(self.results_dir):
                for file in files:
                    if file.endswith('.yaml') and "test_results" in file:
                        yaml_files.append(os.path.join(root, file))
        
        if not yaml_files:
            raise FileNotFoundError(f"No YAML result files found in {self.yaml_dir} or subdirectories of {self.results_dir}")

        # Load data from all models
        model_data = {}
        for yaml_file in yaml_files:
            try:
                file_path = yaml_file if os.path.isabs(yaml_file) else os.path.join(self.yaml_dir, yaml_file)
                with open(file_path, 'r') as f:
                    data = yaml.safe_load(f)
                    
                    # Handle different result formats
                    if "results" in data:
                        # Multiple models in one file
                        for model_name, model_results in data["results"].items():
                            model_data[model_name] = model_results
                    else:
                        # Single model result
                        model_name = data.get("model", os.path.basename(yaml_file).replace(".yaml", ""))
                        model_data[model_name] = data
            except Exception as e:
                logger.error(f"Error loading {yaml_file}: {e}", exc_info=True)

        # Extract specified metrics for each model
        comparison_data = {}
        for model_name, data in model_data.items():
            comparison_data[model_name] = self._extract_metrics(data, metrics)

        # Create comparison matrix
        matrix = {
            "timestamp": self._get_current_timestamp(),
            "metrics": metrics,
            "models": list(comparison_data.keys()),
            "comparison_data": comparison_data,
            "rankings": {}
        }

        # Add rankings for each metric
        for metric in metrics:
            try:
                # Create a list of (model_name, metric_value) tuples
                metric_values = []
                for model_name, model_metrics in comparison_data.items():
                    if metric in model_metrics:
                        value = model_metrics[metric]
                        if isinstance(value, (int, float)):
                            metric_values.append((model_name, value))

                if not metric_values:
                    continue

                # Determine if higher or lower is better
                reverse = not (metric.startswith("cost_") or 
                              metric.endswith("_cost") or 
                              metric.startswith("time_") or 
                              "error" in metric.lower())

                # Sort by metric value
                metric_values.sort(key=lambda x: x[1], reverse=reverse)

                # Create ranking
                matrix["rankings"][metric] = [
                    {"rank": i+1, "model": model, "value": value}
                    for i, (model, value) in enumerate(metric_values)
                ]
            except Exception as e:
                logger.error(f"Error ranking models for metric {metric}: {e}", exc_info=True)

        # Write comparison to YAML file
        with open(output_file, 'w') as f:
            yaml.dump(matrix, f, default_flow_style=False, sort_keys=False)

        return output_file

    def _extract_metrics(self, data: Dict[str, Any], metrics: List[str] = None) -> Dict[str, Any]:
        """
        Extract specified metrics from model data.
        
        Args:
            data: Model result data
            metrics: List of metrics to extract, or None for all
            
        Returns:
            Dictionary of extracted metrics
        """
        result = {}

        # Handle different result formats
        if "metrics" in data:
            # Extract metrics from test result
            metrics_data = data["metrics"]
            for metric, value in metrics_data.items():
                if metrics is None or metric in metrics:
                    if isinstance(value, dict) and "overall_score" in value:
                        # If it's a nested metric with overall score, use that
                        result[metric] = value["overall_score"]
                    else:
                        result[metric] = value
        
        # Extract top-level aggregate metrics
        if "aggregate_metrics" in data:
            for metric, value in data["aggregate_metrics"].items():
                if metrics is None or metric in metrics:
                    result[metric] = value
        
        # Extract other relevant metrics
        for field in ["overall_score", "cost", "usage", "timing"]:
            if field in data:
                value = data[field]
                if isinstance(value, dict):
                    # For dictionaries like usage and timing, flatten
                    for subfield, subvalue in value.items():
                        metric_name = f"{field}_{subfield}"
                        if metrics is None or metric_name in metrics:
                            result[metric_name] = subvalue
                else:
                    # For scalar values like overall_score and cost
                    if metrics is None or field in metrics:
                        result[field] = value

        # Calculate additional metrics if needed
        if "usage" in data and "cost" in data:
            total_tokens = data["usage"].get("total_tokens", 0)
            cost = data["cost"]
            
            if total_tokens > 0:
                result["cost_per_1k_tokens"] = (cost * 1000) / total_tokens

        return result
    
    def _get_common_metrics(self, results: Dict[str, Any]) -> List[str]:
        """
        Get a list of metrics common to all results.
        
        Args:
            results: Dictionary of model results
            
        Returns:
            List of common metrics
        """
        common_metrics = set()
        first = True
        
        for model_id, data in results.items():
            # Skip models with errors
            if "error" in data:
                continue
                
            # Extract metrics from this model
            model_metrics = set()
            
            # Check aggregate metrics
            if "aggregate_metrics" in data:
                for metric in data["aggregate_metrics"].keys():
                    model_metrics.add(metric)
            
            # Check direct metrics
            if "metrics" in data:
                for metric in data["metrics"].keys():
                    model_metrics.add(metric)
            
            # Check other common fields
            for field in ["overall_score", "cost"]:
                if field in data:
                    model_metrics.add(field)
            
            # Handle nested usage and timing
            for field in ["usage", "timing"]:
                if field in data and isinstance(data[field], dict):
                    for subfield in data[field].keys():
                        model_metrics.add(f"{field}_{subfield}")
            
            # Initialize common_metrics with the first model's metrics
            if first:
                common_metrics = model_metrics
                first = False
            else:
                # Keep only metrics present in all models
                common_metrics = common_metrics.intersection(model_metrics)
        
        return sorted(list(common_metrics))
    
    def _get_current_timestamp(self):
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()