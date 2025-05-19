"""HTML report generator for test results."""

import os
import logging
import json
from datetime import datetime
from .base_reporter import BaseReporter

logger = logging.getLogger(__name__)

class HTMLReporter(BaseReporter):
    """Generates HTML reports from test results."""
    
    def __init__(self):
        """Initialize the HTML reporter."""
        super().__init__()
        self.name = "html"

    def generate_report(self, results, output_path):
        """
        Generate an HTML report from test results.
        
        Args:
            results: Dictionary containing test results
            output_path: Path where the report should be saved
            
        Returns:
            bool: True if report generation was successful, False otherwise
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Create HTML content
            html_content = self._generate_html(results)

            # Write to file
            with open(output_path, 'w') as file:
                file.write(html_content)

            logger.info(f"HTML report generated at {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}", exc_info=True)
            return False

    def _generate_html(self, results):
        """
        Generate HTML content from results.
        
        Args:
            results: Dictionary containing test results
            
        Returns:
            str: HTML content
        """
        timestamp = self._get_current_timestamp()
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>LLM Testing Results</title>
            <style>
                body {{ 
                    font-family: Arial, sans-serif; 
                    margin: 20px; 
                    color: #333;
                    line-height: 1.6;
                }}
                h1, h2, h3 {{ 
                    color: #2c3e50; 
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .header {{
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                .model-card {{
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .model-header {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-bottom: 1px solid #ddd;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }}
                .model-body {{
                    padding: 15px;
                }}
                .error {{
                    color: #dc3545;
                    padding: 10px;
                    background-color: #f8d7da;
                    border-radius: 5px;
                }}
                table {{ 
                    border-collapse: collapse; 
                    width: 100%; 
                    margin-bottom: 20px;
                }}
                th, td {{ 
                    border: 1px solid #ddd; 
                    padding: 8px; 
                    text-align: left; 
                }}
                th {{ 
                    background-color: #f2f2f2; 
                    position: sticky;
                    top: 0;
                }}
                tr:nth-child(even) {{ 
                    background-color: #f9f9f9; 
                }}
                .score {{ 
                    font-weight: bold; 
                }}
                .high {{ color: #28a745; }}
                .medium {{ color: #fd7e14; }}
                .low {{ color: #dc3545; }}
                .sample {{
                    background-color: #f8f9fa;
                    padding: 10px;
                    border-radius: 5px;
                    font-family: monospace;
                    white-space: pre-wrap;
                    max-height: 300px;
                    overflow-y: auto;
                }}
                .metrics-container {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                }}
                .metrics-section {{
                    flex: 1;
                    min-width: 300px;
                }}
                .collapsible {{
                    background-color: #f1f1f1;
                    cursor: pointer;
                    padding: 10px;
                    width: 100%;
                    border: none;
                    text-align: left;
                    outline: none;
                    font-weight: bold;
                }}
                .active, .collapsible:hover {{
                    background-color: #e0e0e0;
                }}
                .content {{
                    padding: 0 18px;
                    max-height: 0;
                    overflow: hidden;
                    transition: max-height 0.2s ease-out;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>LLM Testing Results</h1>
                    <p>Report generated on: {timestamp}</p>
                </div>
        """

        # Add summary table
        html += self._generate_summary_table(results)

        # Add detailed results for each model
        for model_id, result in results.items():
            html += self._generate_model_section(model_id, result)

        html += """
            </div>
            <script>
                // Add collapsible behavior
                var coll = document.getElementsByClassName("collapsible");
                for (var i = 0; i < coll.length; i++) {
                    coll[i].addEventListener("click", function() {
                        this.classList.toggle("active");
                        var content = this.nextElementSibling;
                        if (content.style.maxHeight) {
                            content.style.maxHeight = null;
                        } else {
                            content.style.maxHeight = content.scrollHeight + "px";
                        }
                    });
                }
            </script>
        </body>
        </html>
        """

        return html

    def _generate_summary_table(self, results):
        """Generate a summary table of all models."""
        html = """
        <h2>Summary</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Status</th>
                <th>Overall Score</th>
                <th>Test Category</th>
                <th>Context Length</th>
                <th>Cost</th>
                <th>Tokens</th>
            </tr>
        """

        for model_id, result in results.items():
            if "error" in result:
                html += f"""
                <tr>
                    <td>{model_id}</td>
                    <td class="low">Failed</td>
                    <td colspan="5">{result["error"]}</td>
                </tr>
                """
            else:
                # Determine score class
                score = result.get("overall_score", 0)
                score_class = "high" if score > 0.8 else "medium" if score > 0.5 else "low"
                
                # Get token usage
                total_tokens = "N/A"
                if "usage" in result and "total_tokens" in result["usage"]:
                    total_tokens = result["usage"]["total_tokens"]
                
                # Get cost
                cost = result.get("cost", "N/A")
                if isinstance(cost, (int, float)):
                    cost = f"${cost:.6f}"
                
                html += f"""
                <tr>
                    <td>{model_id}</td>
                    <td class="high">Success</td>
                    <td class="score {score_class}">{score:.3f}</td>
                    <td>{result.get("test_category", "N/A")}</td>
                    <td>{result.get("context_length", "N/A")}</td>
                    <td>{cost}</td>
                    <td>{total_tokens}</td>
                </tr>
                """

        html += "</table>"
        return html

    def _generate_model_section(self, model_id, result):
        """Generate a detailed section for a single model."""
        if "error" in result:
            return f"""
            <div class="model-card">
                <div class="model-header">
                    <h2>{model_id}</h2>
                    <span class="score low">Failed</span>
                </div>
                <div class="model-body">
                    <div class="error">{result["error"]}</div>
                </div>
            </div>
            """
        
        # Determine score class
        score = result.get("overall_score", 0)
        score_class = "high" if score > 0.8 else "medium" if score > 0.5 else "low"
        
        html = f"""
        <div class="model-card">
            <div class="model-header">
                <h2>{model_id}</h2>
                <span class="score {score_class}">{score:.3f}</span>
            </div>
            <div class="model-body">
                <p><strong>Test Category:</strong> {result.get("test_category", "N/A")}</p>
                <p><strong>Context Length:</strong> {result.get("context_length", "N/A")}</p>
                
                <div class="metrics-container">
        """
        
        # Add metrics section
        if "metrics" in result:
            html += """
            <div class="metrics-section">
                <button class="collapsible">Metrics</button>
                <div class="content">
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
            """
            
            metrics = result["metrics"]
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, dict):
                    # Handle nested metrics
                    for sub_metric, sub_value in metric_value.items():
                        if isinstance(sub_value, (int, float)):
                            sub_value = f"{sub_value:.3f}"
                        html += f"""
                        <tr>
                            <td>{metric_name}.{sub_metric}</td>
                            <td>{sub_value}</td>
                        </tr>
                        """
                else:
                    if isinstance(metric_value, (int, float)):
                        metric_value = f"{metric_value:.3f}"
                    html += f"""
                    <tr>
                        <td>{metric_name}</td>
                        <td>{metric_value}</td>
                    </tr>
                    """
            
            html += """
                    </table>
                </div>
            </div>
            """
        
        # Add usage section
        if "usage" in result:
            html += """
            <div class="metrics-section">
                <button class="collapsible">Token Usage</button>
                <div class="content">
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
            """
            
            usage = result["usage"]
            for usage_name, usage_value in usage.items():
                html += f"""
                <tr>
                    <td>{usage_name}</td>
                    <td>{usage_value}</td>
                </tr>
                """
            
            html += """
                    </table>
                </div>
            </div>
            """
        
        # Add timing section
        if "timing" in result:
            html += """
            <div class="metrics-section">
                <button class="collapsible">Timing</button>
                <div class="content">
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
            """
            
            timing = result["timing"]
            for timing_name, timing_value in timing.items():
                if isinstance(timing_value, (int, float)):
                    formatted_time = self._format_duration(timing_value)
                    html += f"""
                    <tr>
                        <td>{timing_name}</td>
                        <td>{formatted_time}</td>
                    </tr>
                    """
                else:
                    html += f"""
                    <tr>
                        <td>{timing_name}</td>
                        <td>{timing_value}</td>
                    </tr>
                    """
            
            html += """
                    </table>
                </div>
            </div>
            """
        
        # Close metrics container
        html += "</div>"  # Close metrics-container
        
        # Add response sample if available
        if "response_sample" in result:
            html += """
            <button class="collapsible">Response Sample</button>
            <div class="content">
                <div class="sample">
            """
            html += result["response_sample"].replace("<", "&lt;").replace(">", "&gt;")
            html += """
                </div>
            </div>
            """
        
        # Close model card
        html += """
            </div>
        </div>
        """
        
        return html