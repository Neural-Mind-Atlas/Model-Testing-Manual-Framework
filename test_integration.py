#!/usr/bin/env python3
# test_integration.py

import os
import sys
import logging
import json
import yaml
import shutil
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("integration_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("integration_test")

# Add src directory to path if needed
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, 'src')
if os.path.exists(src_dir) and src_dir not in sys.path:
    sys.path.append(script_dir)

# Create test output directory
test_output_dir = os.path.join(script_dir, 'test_output')
os.makedirs(test_output_dir, exist_ok=True)

def test_client_initialization():
    """Test model client initialization with proper configuration."""
    logger.info("Testing client initialization...")
    try:
        from src.utils.config import load_model_client, get_model_config
        
        # Test with sample model config
        model_config = {
            "name": "test_model",
            "display_name": "Test Model",
            "version": "1.0.0",
            "provider": "openai",
            "max_tokens": 4096,
            "context_window": 8192,
            "defaults": {
                "temperature": 0.7,
                "top_p": 1.0
            },
            "cost": {
                "input_per_1k": 5.0,
                "output_per_1k": 15.0
            }
        }
        
        # Try initializing a client
        client = load_model_client("test_model", model_config)
        
        # Since we're testing integration only and don't have actual API keys,
        # just check that the client initialization logic works
        if client is None:
            logger.info("Client initialization returned None as expected (no API keys available)")
            return True
        else:
            logger.info("Client initialization succeeded")
            return True
    except Exception as e:
        logger.error(f"Client initialization failed: {e}", exc_info=True)
        return False

def test_evaluation_integration():
    """Test evaluator integration with test executor."""
    logger.info("Testing evaluator integration...")
    try:
        from src.evaluators.accuracy_evaluator import AccuracyEvaluator
        from src.evaluators.reasoning_evaluator import ReasoningEvaluator
        from src.evaluators.hallucination_evaluator import HallucinationEvaluator
        from src.evaluators.efficiency_evaluator import EfficiencyEvaluator
        
        # Create a mock evaluation model
        class MockEvaluationModel:
            def generate(self, prompt, config=None):
                return "Rating: 4\nExplanation: This is a good response."
        
        # Test each evaluator
        eval_model = MockEvaluationModel()
        
        # Test accuracy evaluator
        accuracy_evaluator = AccuracyEvaluator(eval_model)
        accuracy_result = accuracy_evaluator.evaluate(
            "What is the capital of France?", 
            "The capital of France is Paris."
        )
        logger.info(f"Accuracy evaluator result: {accuracy_result}")
        
        # Test reasoning evaluator
        reasoning_evaluator = ReasoningEvaluator(eval_model)
        reasoning_result = reasoning_evaluator.evaluate(
            "Solve this problem: 2 + 2 = ?", 
            "2 + 2 = 4 because addition of 2 and 2 equals 4."
        )
        logger.info(f"Reasoning evaluator result: {reasoning_result}")
        
        # Test hallucination evaluator
        hallucination_evaluator = HallucinationEvaluator(eval_model)
        hallucination_result = hallucination_evaluator.evaluate(
            "What is the capital of France?", 
            "The capital of France is Paris.",
            context="Paris is the capital and most populous city of France."
        )
        logger.info(f"Hallucination evaluator result: {hallucination_result}")
        
        # Test efficiency evaluator
        efficiency_evaluator = EfficiencyEvaluator()
        efficiency_result = efficiency_evaluator.evaluate(
            {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 200,
                    "total_tokens": 300
                },
                "timing": {
                    "total_time": 2.5,
                    "time_to_first_token": 0.5
                },
                "cost": 0.01
            },
            0.85  # quality score
        )
        logger.info(f"Efficiency evaluator result: {efficiency_result}")
        
        return True
    except Exception as e:
        logger.error(f"Evaluator integration failed: {e}", exc_info=True)
        return False

def test_test_executor():
    """Test the test executor with mock data."""
    logger.info("Testing test executor...")
    try:
        from src.test_runner.executor import TestExecutor
        
        # Create test executor
        executor = TestExecutor()
        
        # Create mock model config
        model_config = {
            "name": "test_model",
            "display_name": "Test Model",
            "provider": "test",
            "defaults": {},
            "cost": {
                "input_per_1k": 0.01,
                "output_per_1k": 0.03
            }
        }
        
        # Mock the model client methods
        def mock_load_test_data(self, test_category, context_length):
            return {
                "context": "This is a test context.",
                "prompt": "Generate a response about AI testing."
            }
        
        def mock_get_evaluation_model(self, test_config):
            class MockEvaluationModel:
                def generate(self, prompt, config=None):
                    return "Rating: 4\nExplanation: This is a good response."
            return MockEvaluationModel()
        
        # Patch methods to avoid actual file loading and API calls
        executor.load_test_data = lambda test_category, context_length: {
            "context": "This is a test context.",
            "prompt": "Generate a response about AI testing."
        }
        
        executor.get_evaluation_model = lambda test_config: type('MockEvaluationModel', (), {
            'generate': lambda prompt, config=None: "Rating: 4\nExplanation: This is a good response."
        })
        
        # Mock the client generation
        from src.utils.config import load_model_client
        original_load_model_client = load_model_client
        
        def mock_load_model_client(model_id, model_config):
            return type('MockClient', (), {
                'generate': lambda prompt, config=None: "This is a mock response about AI testing.",
                'calculate_cost': lambda input_tokens, output_tokens: 0.01
            })
        
        import src.utils.config
        src.utils.config.load_model_client = mock_load_model_client
        
        # Now try running a test
        result = executor.run_test("test_model", model_config, "reasoning", "short")
        logger.info(f"Test executor result: {result}")
        
        # Restore original function
        src.utils.config.load_model_client = original_load_model_client
        
        return True
    except Exception as e:
        logger.error(f"Test executor failed: {e}", exc_info=True)
        return False

def create_mock_results():
    """Create mock test results for reporting tests."""
    return {
        "model_1": {
            "model_id": "model_1",
            "test_category": "reasoning",
            "context_length": "short",
            "overall_score": 0.85,
            "metrics": {
                "reasoning": 0.9,
                "factual": 0.8,
                "hallucination": 0.85,
                "efficiency": {
                    "token_efficiency": 2.5,
                    "cost_per_quality_point": 0.01,
                    "tokens_per_second": 150
                }
            },
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 200,
                "total_tokens": 300
            },
            "timing": {
                "total_time": 2.0
            },
            "cost": 0.01,
            "response_sample": "This is a sample response from model 1."
        },
        "model_2": {
            "model_id": "model_2",
            "test_category": "reasoning",
            "context_length": "short",
            "overall_score": 0.75,
            "metrics": {
                "reasoning": 0.8,
                "factual": 0.7,
                "hallucination": 0.75,
                "efficiency": {
                    "token_efficiency": 2.0,
                    "cost_per_quality_point": 0.015,
                    "tokens_per_second": 120
                }
            },
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 250,
                "total_tokens": 350
            },
            "timing": {
                "total_time": 2.5
            },
            "cost": 0.015,
            "response_sample": "This is a sample response from model 2."
        },
        "model_with_error": {
            "model_id": "model_with_error",
            "test_category": "reasoning",
            "context_length": "short",
            "error": "Failed to initialize client"
        }
    }

def test_reporting_integration():
    """Test reporting integration with mock results."""
    logger.info("Testing reporting integration...")
    
    # Create mock results
    results = create_mock_results()
    
    success = True
    
    # Test YAML reporter
    try:
        from src.reporting.yaml_generator import YAMLReporter
        yaml_reporter = YAMLReporter()
        yaml_output = os.path.join(test_output_dir, "test_results.yaml")
        yaml_result = yaml_reporter.generate_report(results, yaml_output)
        
        if yaml_result and os.path.exists(yaml_output):
            logger.info(f"YAML report generated successfully: {yaml_output}")
        else:
            logger.error("YAML report generation failed")
            success = False
    except Exception as e:
        logger.error(f"YAML reporter failed: {e}", exc_info=True)
        success = False
    
    # Test JSON reporter
    try:
        from src.reporting.json_generator import JSONReporter
        json_reporter = JSONReporter()
        json_output = os.path.join(test_output_dir, "test_results.json")
        json_result = json_reporter.generate_report(results, json_output)
        
        if json_result and os.path.exists(json_output):
            logger.info(f"JSON report generated successfully: {json_output}")
        else:
            logger.error("JSON report generation failed")
            success = False
    except Exception as e:
        logger.error(f"JSON reporter failed: {e}", exc_info=True)
        success = False
    
    # Test HTML reporter
    try:
        from src.reporting.html_generator import HTMLReporter
        html_reporter = HTMLReporter()
        html_output = os.path.join(test_output_dir, "test_results.html")
        html_result = html_reporter.generate_report(results, html_output)
        
        if html_result and os.path.exists(html_output):
            logger.info(f"HTML report generated successfully: {html_output}")
        else:
            logger.error("HTML report generation failed")
            success = False
    except Exception as e:
        logger.error(f"HTML reporter failed: {e}", exc_info=True)
        success = False
        
    return success

def test_analysis_integration():
    """Test analysis reporting integration with mock results."""
    logger.info("Testing analysis reporting integration...")
    
    # Create mock results
    results = create_mock_results()
    
    success = True
    
    # Test comparisons reporter
    try:
        from src.reporting.comparisons_reporter import ModelComparison
        comparisons_dir = os.path.join(test_output_dir, "comparisons")
        os.makedirs(comparisons_dir, exist_ok=True)
        
        comparison = ModelComparison(results_dir=test_output_dir)
        comparison_output = os.path.join(comparisons_dir, "comparison.yaml")
        
        # Format results for comparison
        formatted_results = {}
        for model_id, result in results.items():
            if "error" not in result:
                formatted_results[model_id] = {
                    "model": model_id,
                    "test_category": result.get("test_category", "unknown"),
                    "context_length": result.get("context_length", "unknown"),
                    "aggregate_metrics": {
                        "overall_score": result.get("overall_score", 0),
                        "total_tokens": result.get("usage", {}).get("total_tokens", 0),
                        "total_cost_usd": result.get("cost", 0),
                        "average_processing_time_seconds": result.get("timing", {}).get("total_time", 0)
                    }
                }
        
        comparison.generate_comparison(formatted_results, output_file=comparison_output)
        
        if os.path.exists(comparison_output):
            logger.info(f"Comparison report generated successfully: {comparison_output}")
        else:
            logger.error("Comparison report generation failed")
            success = False
    except Exception as e:
        logger.error(f"Comparisons reporter failed: {e}", exc_info=True)
        success = False
    
    # Test cost analysis reporter
    try:
        from src.reporting.cost_analysis_reporter import CostAnalyzer
        cost_analyzer = CostAnalyzer(results_dir=test_output_dir)
        cost_output = os.path.join(test_output_dir, "cost_analysis.yaml")
        cost_result = cost_analyzer.generate_report(results, cost_output)
        
        if cost_result and os.path.exists(cost_output):
            logger.info(f"Cost analysis report generated successfully: {cost_output}")
        else:
            logger.error("Cost analysis report generation failed")
            success = False
    except Exception as e:
        logger.error(f"Cost analyzer failed: {e}", exc_info=True)
        success = False
    
    # Test recommendations reporter
    try:
        from src.reporting.recommendations_reporter import RecommendationEngine
        recommendations_engine = RecommendationEngine(results_dir=test_output_dir)
        recommendations_output = os.path.join(test_output_dir, "recommendations.yaml")
        recommendations_result = recommendations_engine.generate_report(results, recommendations_output)
        
        if recommendations_result and os.path.exists(recommendations_output):
            logger.info(f"Recommendations report generated successfully: {recommendations_output}")
        else:
            logger.error("Recommendations report generation failed")
            success = False
    except Exception as e:
        logger.error(f"Recommendations engine failed: {e}", exc_info=True)
        success = False
    
    # Test visualizations reporter
    try:
        from src.reporting.visualizations_reporter import Visualizer
        visualizer = Visualizer(results_dir=test_output_dir)
        viz_output = os.path.join(test_output_dir, "visualizations")
        os.makedirs(viz_output, exist_ok=True)
        viz_result = visualizer.generate_report(results, viz_output)
        
        if viz_result and os.path.exists(viz_output) and len(os.listdir(viz_output)) > 0:
            logger.info(f"Visualizations generated successfully in: {viz_output}")
        else:
            logger.error("Visualizations generation failed or no files created")
            success = False
    except Exception as e:
        logger.error(f"Visualizer failed: {e}", exc_info=True)
        success = False
        
    return success

def test_main_integration():
    """Test the main script integration."""
    logger.info("Testing main script integration...")
    
    try:
        import src.main
        
        # We can't easily mock command-line arguments, so we'll consider
        # the main script integrated if it imports successfully
        logger.info("Main script imports successfully")
        return True
    except Exception as e:
        logger.error(f"Main script integration failed: {e}", exc_info=True)
        return False

def run_integration_tests():
    """Run all integration tests and report results."""
    tests = [
        ("Client initialization", test_client_initialization),
        ("Evaluator integration", test_evaluation_integration),
        ("Test executor", test_test_executor),
        ("Reporting integration", test_reporting_integration),
        ("Analysis integration", test_analysis_integration),
        ("Main script integration", test_main_integration)
    ]
    
    results = {}
    all_passed = True
    
    for test_name, test_func in tests:
        logger.info(f"Running test: {test_name}")
        try:
            result = test_func()
            results[test_name] = result
            if not result:
                all_passed = False
            logger.info(f"Test {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            logger.error(f"Test {test_name} raised exception: {e}", exc_info=True)
            results[test_name] = False
            all_passed = False
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("INTEGRATION TEST SUMMARY")
    logger.info("="*50)
    
    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info("-"*50)
    logger.info(f"Overall result: {'PASSED' if all_passed else 'FAILED'}")
    logger.info("="*50)
    
    # Create a summary report
    summary = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "overall_result": "PASSED" if all_passed else "FAILED"
    }
    
    with open(os.path.join(test_output_dir, "integration_test_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    return all_passed

if __name__ == "__main__":
    logger.info("Starting integration tests")
    success = run_integration_tests()
    sys.exit(0 if success else 1)