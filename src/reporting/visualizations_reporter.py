"""Visualization generator for test results."""

import os
import logging
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import math
from .base_reporter import BaseReporter

logger = logging.getLogger(__name__)

class Visualizer(BaseReporter):
    """Generates visualizations from test results."""

    def __init__(self, results_dir: str = "./results"):
        """
        Initialize the visualizer.

        Args:
            results_dir: Directory containing results
        """
        super().__init__()
        self.name = "visualizer"
        self.results_dir = results_dir
        self.visualizations_dir = os.path.join(results_dir, "visualizations")

        # Ensure output directory exists
        os.makedirs(self.visualizations_dir, exist_ok=True)

        # Set default style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["font.size"] = 10

    def generate_report(self, results: Dict[str, Any], output_path: str) -> bool:
        """
        Generate visualization reports from test results.
        
        Args:
            results: Dictionary containing test results
            output_path: Directory where visualizations should be saved
            
        Returns:
            bool: True if report generation was successful, False otherwise
        """
        try:
            # Ensure output directory exists
            if os.path.isfile(output_path):
                # If output_path is a file, use its directory
                output_dir = os.path.dirname(output_path)
            else:
                # If output_path is a directory, use it directly
                output_dir = output_path
                
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create various visualizations
            visualizations = []
            
            # Only create visualizations if we have more than one model
            if len(results) > 1:
                # Extract valid results (no errors)
                valid_results = {model_id: result for model_id, result in results.items() if "error" not in result}
                
                if valid_results:
                    # Create bar chart of overall scores
                    bar_chart_path = os.path.join(output_dir, f"overall_scores_{timestamp}.png")
                    self.create_bar_chart(valid_results, "overall_score", bar_chart_path)
                    visualizations.append(bar_chart_path)
                    
                    # Create radar chart of key metrics if available
                    metrics = self._get_common_metrics(valid_results)
                    if len(metrics) >= 3:
                        radar_chart_path = os.path.join(output_dir, f"metrics_radar_{timestamp}.png")
                        self.create_radar_chart(valid_results, metrics[:5], radar_chart_path)  # Use top 5 metrics
                        visualizations.append(radar_chart_path)
                    
                    # Create cost vs performance scatter plot if cost data is available
                    if all("cost" in result for result in valid_results.values()):
                        scatter_path = os.path.join(output_dir, f"cost_vs_performance_{timestamp}.png")
                        self.create_cost_vs_performance_scatter(valid_results, scatter_path)
                        visualizations.append(scatter_path)
            
            logger.info(f"Generated {len(visualizations)} visualizations in {output_dir}")
            return True
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}", exc_info=True)
            return False

    def create_radar_chart(self,
                      results: Dict[str, Any],
                      metrics: List[str],
                      output_file: Optional[str] = None) -> str:
        """
        Create a radar chart comparing multiple models across metrics.

        Args:
            results: Dictionary containing test results
            metrics: List of metrics to compare
            output_file: Output file path, or None to use default

        Returns:
            Path to the generated visualization
        """
        # Default output file
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_str = "_".join(metrics[:3]) + (f"_plus_{len(metrics)-3}" if len(metrics) > 3 else "")
            output_file = os.path.join(self.visualizations_dir, f"radar_chart_{metrics_str}_{timestamp}.png")

        # Extract data for radar chart
        chart_data = {}
        
        for model_id, result in results.items():
            values = []
            for metric in metrics:
                value = self._extract_metric_value(result, metric)
                values.append(value if value is not None else 0)
            
            if values and any(v > 0 for v in values):
                chart_data[model_id] = values

        if not chart_data:
            logger.warning(f"No valid data for radar chart for metrics: {metrics}")
            return None

        # Normalize values for radar chart (0-1 scale)
        normalized_data = {}
        for model, values in chart_data.items():
            normalized = []
            for i, val in enumerate(values):
                # Find max value for this metric across all models
                max_val = max(chart_data[m][i] for m in chart_data)
                if max_val > 0:
                    normalized.append(val / max_val)
                else:
                    normalized.append(0)
            normalized_data[model] = normalized

        # Create radar chart
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, polar=True)

        # Set the angle of each axis
        angles = [n / float(len(metrics)) * 2 * math.pi for n in range(len(metrics))]
        angles += angles[:1]  # Close the loop

        # Add labels for each axis
        plt.xticks(angles[:-1], [m.replace("_", " ").title() for m in metrics])

        # Draw axis lines
        ax.set_theta_offset(math.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw y-axis grid lines
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"])
        
        # Plot each model
        for i, (model, values) in enumerate(normalized_data.items()):
            values += values[:1]  # Close the loop
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, marker='o')
            ax.fill(angles, values, alpha=0.1)

        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

        plt.title('Model Comparison Across Metrics', size=15)
        plt.tight_layout()

        # Save the chart
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        return output_file

    def create_bar_chart(self,
                    results: Dict[str, Any],
                    metric: str,
                    output_file: Optional[str] = None) -> str:
        """
        Create a bar chart for a specific metric across models.

        Args:
            results: Dictionary containing test results
            metric: Metric to compare
            output_file: Output file path, or None to use default

        Returns:
            Path to the generated visualization
        """
        # Default output file
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.visualizations_dir, f"bar_chart_{metric}_{timestamp}.png")

        # Extract data for bar chart
        chart_data = []

        for model_id, result in results.items():
            value = self._extract_metric_value(result, metric)
            if value is not None:
                chart_data.append({
                    "model": model_id,
                    "value": value
                })

        if not chart_data:
            logger.warning(f"No valid data for bar chart for metric: {metric}")
            return None

        # Sort by value
        chart_data.sort(key=lambda x: x["value"], reverse=True)
        
        # Create bar chart
        plt.figure(figsize=(12, 8))
        
        # Plot bars
        models = [item["model"] for item in chart_data]
        values = [item["value"] for item in chart_data]
        
        # Create color map based on values
        colors = plt.cm.YlGnBu(np.linspace(0.3, 0.9, len(values)))
        
        bars = plt.bar(models, values, color=colors)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', rotation=0)

        # Set labels and title
        plt.xlabel("Model")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"{metric.replace('_', ' ').title()} Comparison")

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha="right")
        
        plt.tight_layout()

        # Save the chart
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        return output_file

    def create_cost_vs_performance_scatter(self,
                                     results: Dict[str, Any],
                                     output_file: Optional[str] = None) -> str:
        """
        Create a scatter plot of cost vs. performance.

        Args:
            results: Dictionary containing test results
            output_file: Output file path, or None to use default

        Returns:
            Path to the generated visualization
        """
        # Default output file
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.visualizations_dir, f"cost_vs_performance_{timestamp}.png")

        # Extract data for scatter plot
        scatter_data = []

        for model_id, result in results.items():
            cost = self._extract_metric_value(result, "cost")
            performance = self._extract_metric_value(result, "overall_score")
            
            if cost is not None and performance is not None:
                scatter_data.append({
                    "model": model_id,
                    "cost": cost,
                    "performance": performance
                })

        if not scatter_data:
            logger.warning("No valid data for cost vs performance scatter plot")
            return None

        # Create scatter plot
        plt.figure(figsize=(12, 8))
        
        # Extract data
        models = [item["model"] for item in scatter_data]
        costs = [item["cost"] for item in scatter_data]
        performances = [item["performance"] for item in scatter_data]
        
        # Create scatter plot with varying sizes based on performance
        sizes = [p * 100 + 50 for p in performances]  # Scale performance for marker size
        
        plt.scatter(costs, performances, s=sizes, alpha=0.7)
        
        # Add labels for each point
        for i, model in enumerate(models):
            plt.annotate(model, (costs[i], performances[i]), 
                        xytext=(7, 0), textcoords='offset points',
                        fontsize=10)

        # Add cost-efficiency line (diagonal lines from origin)
        max_cost = max(costs) * 1.1
        max_perf = max(performances) * 1.1
        
        # Add reference lines for different cost-efficiency ratios
        for ratio in [0.5, 1.0, 2.0]:
            x = np.linspace(0, max_cost, 100)
            y = ratio * x
            mask = y <= max_perf
            plt.plot(x[mask], y[mask], '--', color='gray', alpha=0.5, 
                    label=f'Performance/Cost = {ratio}' if ratio == 1.0 else None)
        
        # Set labels and title
        plt.xlabel("Cost ($)")
        plt.ylabel("Performance Score")
        plt.title("Cost vs. Performance Comparison")
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend if needed
        plt.legend()
        
        plt.tight_layout()

        # Save the chart
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        return output_file
    
    def _extract_metric_value(self, result: Dict[str, Any], metric: str) -> Optional[float]:
        """
        Extract a metric value from a result dictionary.
        
        Args:
            result: Result dictionary
            metric: Metric to extract
            
        Returns:
            Metric value or None if not found
        """
        # Check direct top-level metric
        if metric in result:
            value = result[metric]
            if isinstance(value, (int, float)):
                return float(value)
        
        # Check metrics dictionary
        if "metrics" in result and metric in result["metrics"]:
            value = result["metrics"][metric]
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, dict) and "overall_score" in value:
                return float(value["overall_score"])
        
        # Check nested dictionaries
        for field in ["usage", "timing"]:
            if field in result and isinstance(result[field], dict):
                nested_metric = f"{field}_{metric}" if not metric.startswith(f"{field}_") else metric
                field_metric = metric[len(field)+1:] if metric.startswith(f"{field}_") else None
                
                if nested_metric in result[field]:
                    value = result[field][nested_metric]
                    if isinstance(value, (int, float)):
                        return float(value)
                elif field_metric and field_metric in result[field]:
                    value = result[field][field_metric]
                    if isinstance(value, (int, float)):
                        return float(value)
        
        return None
    
    def _get_common_metrics(self, results: Dict[str, Any]) -> List[str]:
        """
        Get a list of metrics common to all results.
        
        Args:
            results: Dictionary of model results
            
        Returns:
            List of common metrics
        """
        # Start with commonly expected metrics
        priority_metrics = [
            "overall_score", 
            "reasoning", 
            "factual", 
            "hallucination",
            "instruction",
            "context",
            "creative",
            "ppt_writing",
            "cost"
        ]
        
        # Check which priority metrics are available
        available_metrics = []
        for metric in priority_metrics:
            if all(self._extract_metric_value(result, metric) is not None for result in results.values()):
                available_metrics.append(metric)
        
        # If we don't have enough priority metrics, look for others
        if len(available_metrics) < 3:
            # Collect all possible metrics
            all_metrics = set()
            for result in results.values():
                # Add direct metrics
                for key in result.keys():
                    if isinstance(result[key], (int, float)):
                        all_metrics.add(key)
                
                # Add metrics from metrics dictionary
                if "metrics" in result:
                    for key, value in result["metrics"].items():
                        if isinstance(value, (int, float)) or (isinstance(value, dict) and "overall_score" in value):
                            all_metrics.add(key)
            
            # Find metrics available in all results
            common_metrics = []
            for metric in all_metrics:
                if all(self._extract_metric_value(result, metric) is not None for result in results.values()):
                    common_metrics.append(metric)
            
            # Add any common metrics not already in available_metrics
            for metric in common_metrics:
                if metric not in available_metrics:
                    available_metrics.append(metric)
        
        return available_metrics