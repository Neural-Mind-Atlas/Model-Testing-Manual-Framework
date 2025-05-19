# src/reporting/yaml_generator.py
import yaml
import os
import logging
from .base_reporter import BaseReporter

logger = logging.getLogger(__name__)

class YAMLReporter(BaseReporter):
    """Generates YAML reports from test results."""
    
    def __init__(self):
        """Initialize the YAML reporter."""
        super().__init__()
        self.name = "yaml"

    def generate_report(self, results, output_path):
        """
        Generate a YAML report from test results.
        
        Args:
            results: Dictionary containing test results
            output_path: Path where the report should be saved
            
        Returns:
            bool: True if report generation was successful, False otherwise
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Format results for YAML output
            formatted_results = self._format_results(results)

            # Write results to YAML file
            with open(output_path, 'w') as file:
                yaml.dump(formatted_results, file, default_flow_style=False, sort_keys=False)

            logger.info(f"YAML report generated at {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error generating YAML report: {e}", exc_info=True)
            return False
            
    def _format_results(self, results):
        """
        Format results for YAML output.
        
        Args:
            results: Dictionary containing test results
            
        Returns:
            dict: Formatted results for YAML output
        """
        formatted = {
            "test_date": self._get_current_timestamp(),
            "framework_version": "1.0.0",
            "models_tested": list(results.keys()),
            "results": {}
        }
        
        for model_id, result in results.items():
            if "error" in result:
                formatted["results"][model_id] = {
                    "error": result["error"],
                    "status": "failed"
                }
            else:
                model_result = {
                    "status": "success",
                    "test_category": result.get("test_category", "unknown"),
                    "context_length": result.get("context_length", "unknown"),
                    "overall_score": result.get("overall_score", 0),
                    "metrics": result.get("metrics", {}),
                    "usage": result.get("usage", {}),
                    "timing": result.get("timing", {}),
                    "cost": result.get("cost", 0)
                }
                
                # Add response sample if available (truncated for readability)
                if "response_sample" in result:
                    sample = result["response_sample"]
                    # Truncate very long samples
                    if len(sample) > 1000:
                        sample = sample[:1000] + "... [truncated]"
                    model_result["response_sample"] = sample
                
                formatted["results"][model_id] = model_result
        
        return formatted