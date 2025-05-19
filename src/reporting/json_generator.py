"""JSON report generator for test results."""

import json
import os
import logging
from datetime import datetime
from .base_reporter import BaseReporter

logger = logging.getLogger(__name__)

class JSONReporter(BaseReporter):
    """Generates JSON reports from test results."""
    
    def __init__(self):
        """Initialize the JSON reporter."""
        super().__init__()
        self.name = "json"

    def generate_report(self, results, output_path):
        """
        Generate a JSON report from test results.
        
        Args:
            results: Dictionary containing test results
            output_path: Path where the report should be saved
            
        Returns:
            bool: True if report generation was successful, False otherwise
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Format results for JSON output
            formatted_results = self._format_results(results)

            # Write results to JSON file
            with open(output_path, 'w') as file:
                json.dump(formatted_results, file, indent=2)

            logger.info(f"JSON report generated at {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error generating JSON report: {e}", exc_info=True)
            return False
    
    def _format_results(self, results):
        """
        Format results for JSON output.
        
        Args:
            results: Dictionary containing test results
            
        Returns:
            dict: Formatted results for JSON output
        """
        formatted = {
            "metadata": {
                "timestamp": self._get_current_timestamp(),
                "framework_version": "1.0.0",
                "models_tested": list(results.keys())
            },
            "results": {}
        }
        
        for model_id, result in results.items():
            if "error" in result:
                formatted["results"][model_id] = {
                    "error": result["error"],
                    "status": "failed"
                }
            else:
                # Create a copy of the result to avoid modifying the original
                model_result = dict(result)
                
                # Format numerical values for better readability
                self._format_numerical_values(model_result)
                
                # Add status field
                model_result["status"] = "success"
                
                formatted["results"][model_id] = model_result
        
        return formatted
    
    def _format_numerical_values(self, data):
        """
        Format numerical values in data for better readability.
        
        Args:
            data: Dictionary containing data to format
            
        Returns:
            None (modifies data in place)
        """
        if not isinstance(data, dict):
            return
        
        for key, value in data.items():
            if isinstance(value, dict):
                self._format_numerical_values(value)
            elif isinstance(value, float):
                # Round to 6 decimal places for better readability
                data[key] = round(value, 6)