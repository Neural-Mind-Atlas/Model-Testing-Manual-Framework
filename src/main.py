# src/main.py
import os
import sys
import argparse
import yaml
import logging
import json
from datetime import datetime
from dotenv import load_dotenv
from src.utils.config import load_config, get_model_config
from src.test_runner.executor import TestExecutor
from src.reporting.yaml_generator import YAMLReporter
from src.test_runner.logger import TestLogger

# Load environment variables
load_dotenv()

def setup_logging():
    """Configure logging for the application"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_level_str = os.getenv("LOG_LEVEL", "INFO")
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"test_run_{timestamp}.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def get_available_models():
    """Load available models from configuration files"""
    models = {}
    model_dir = os.path.join("config", "models")
    if not os.path.exists(model_dir):
        model_dir = os.path.join("config")  # Fallback to main config directory

    for file_name in os.listdir(model_dir):
        if file_name.endswith('.yaml'):
            file_path = os.path.join(model_dir, file_name)
            try:
                # Determine provider from filename
                provider = file_name.replace('.yaml', '').lower()

                with open(file_path, 'r') as file:
                    model_config = yaml.safe_load(file)
                    if model_config and 'models' in model_config:
                        # Handle list-style model configurations
                        if isinstance(model_config['models'], list):
                            for model in model_config['models']:
                                if 'name' in model:
                                    model_id = model['name']
                                    # Add provider if not present
                                    if 'provider' not in model:
                                        if provider == 'others':
                                            # For others.yaml, try to determine provider from name
                                            if 'cohere' in model_id:
                                                model['provider'] = 'cohere'
                                            elif 'dbrx' in model_id or 'databricks' in model_id:
                                                model['provider'] = 'databricks'
                                            elif 'qwen' in model_id:
                                                model['provider'] = 'qwen'
                                            elif 'falcon' in model_id:
                                                model['provider'] = 'falcon'
                                            else:
                                                model['provider'] = 'unknown'
                                        else:
                                            model['provider'] = provider
                                    models[model_id] = model
                        # Handle dictionary-style model configurations
                        elif isinstance(model_config['models'], dict):
                            for model_id, model in model_config['models'].items():
                                # Add provider if not present
                                if 'provider' not in model:
                                    if provider == 'others':
                                        # For others.yaml, try to determine provider from name
                                        if 'cohere' in model_id:
                                            model['provider'] = 'cohere'
                                        elif 'dbrx' in model_id or 'databricks' in model_id:
                                            model['provider'] = 'databricks'
                                        elif 'qwen' in model_id:
                                            model['provider'] = 'qwen'
                                        elif 'falcon' in model_id:
                                            model['provider'] = 'falcon'
                                        else:
                                            model['provider'] = 'unknown'
                                    else:
                                        model['provider'] = provider
                                models[model_id] = model
            except Exception as e:
                logging.error(f"Error loading model config file {file_path}: {e}")

    return models

def get_test_suite_config(test_category):
    """Get configuration for a specific test suite"""
    test_suite_path = os.path.join("config", "test_suites", f"{test_category}.yaml")
    if os.path.exists(test_suite_path):
        with open(test_suite_path, 'r') as file:
            return yaml.safe_load(file)
    return None

def validate_api_keys(logger):
    """Validate required API keys are set as environment variables"""
    required_api_keys = {
        "OPENAI_API_KEY": "OpenAI",
        "ANTHROPIC_API_KEY": "Anthropic",
        "GOOGLE_API_KEY": "Google",
        "MISTRAL_API_KEY": "Mistral",
        "COHERE_API_KEY": "Cohere"
    }
    
    missing_keys = []
    for env_var, provider in required_api_keys.items():
        if not os.environ.get(env_var) and os.environ.get(f"TEST_{provider.upper()}", "true").lower() == "true":
            missing_keys.append(f"{provider} ({env_var})")
    
    if missing_keys:
        logger.warning(f"Missing API keys for: {', '.join(missing_keys)}")
        logger.warning("Tests for these providers may fail. Set API keys in .env file or disable testing with TEST_{PROVIDER}=false")

def main():
    logger = setup_logging()
    
    # Validate API keys
    validate_api_keys(logger)
    
    parser = argparse.ArgumentParser(description="LLM Testing Framework")
    parser.add_argument('--model', type=str, help='Specific model to test')
    parser.add_argument('--models', type=str, help='Comma-separated list of models to test')
    parser.add_argument('--all-models', action='store_true', help='Test all available models')
    parser.add_argument('--test', type=str, default='ppt_writing',
                        help='Test category to run (default: ppt_writing)')
    parser.add_argument('--context', type=str, default='short',
                        choices=['short', 'medium', 'long'],
                        help='Context length for testing')
    parser.add_argument('--output', type=str, default='yaml',
                        choices=['console', 'json', 'yaml', 'html'],
                        help='Output format for results')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--eval-model', type=str, default='gpt_4o',
                        help='Model to use for evaluation (default: gpt_4o)')
    parser.add_argument('--parallel', action='store_true', 
                        help='Run tests in parallel (uses MAX_PARALLEL_REQUESTS from .env)')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load available models
    available_models = get_available_models()
    if not available_models:
        logger.error("No models found in configuration files.")
        return

    logger.info(f"Found {len(available_models)} models in configuration")

    # Determine which models to test
    models_to_test = []
    if args.all_models:
        models_to_test = list(available_models.keys())
    elif args.models:
        models_to_test = [model.strip() for model in args.models.split(',')]
    elif args.model:
        models_to_test = [args.model]
    else:
        logger.error("No model specified. Use --model, --models, or --all-models")
        return

    # Validate models
    valid_models = []
    for model_id in models_to_test:
        if model_id in available_models:
            valid_models.append(model_id)
        else:
            logger.warning(f"Model {model_id} not found in configuration")

    if not valid_models:
        logger.error("No valid models to test")
        return

    logger.info(f"Testing {len(valid_models)} models: {', '.join(valid_models)}")

    # Load test suite configuration
    test_suite = get_test_suite_config(args.test)
    if not test_suite:
        logger.warning(f"No specific configuration found for test category {args.test}. Using default settings.")
        test_suite = {
            "name": args.test,
            "description": f"Tests for {args.test}",
            "weight": 1.0,
            "test_cases": []
        }

    # Set up the evaluation model
    eval_model_id = args.eval_model
    logger.info(f"Using {eval_model_id} for evaluation")

    # Create test executor
    executor = TestExecutor()
    
    # If the eval model is specified, initialize it
    if eval_model_id and eval_model_id in available_models:
        eval_model_config = available_models[eval_model_id]
        executor.default_eval_model_id = eval_model_id

    # Create dictionary of models to test
    models_dict = {model_id: available_models[model_id] for model_id in valid_models}
    
    # Run tests
    if args.parallel:
        results = executor.run_tests_parallel(models_dict, {
            "test_categories": [args.test],
            "context_lengths": [args.context]
        })
    else:
        # Run tests for each model
        results = {}
        for model_id in valid_models:
            logger.info(f"Testing model: {model_id}")
            try:
                model_config = available_models[model_id]
                test_result = executor.run_test(
                    model_id=model_id,
                    model_config=model_config,
                    test_category=args.test,
                    context_length=args.context
                )
                results[model_id] = test_result
                logger.info(f"Testing completed for {model_id}")
            except Exception as e:
                logger.error(f"Error testing model {model_id}: {e}", exc_info=True)
                results[model_id] = {
                    "model_id": model_id,
                    "test_category": args.test,
                    "context_length": args.context,
                    "error": str(e)
                }

    # Generate reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.getenv("RESULTS_DIR", "./results")
    output_dir = os.path.join(results_dir, args.test)
    os.makedirs(output_dir, exist_ok=True)

    # Choose reporter based on output format
    if args.output == 'yaml':
        reporter = YAMLReporter()
        output_file = os.path.join(output_dir, f"test_results_{args.context}_{timestamp}.yaml")
        reporter.generate_report(results, output_file)
        logger.info(f"Results saved to {output_file}")
    elif args.output == 'json':
        from src.reporting.json_generator import JSONReporter
        reporter = JSONReporter()
        output_file = os.path.join(output_dir, f"test_results_{args.context}_{timestamp}.json")
        reporter.generate_report(results, output_file)
        logger.info(f"Results saved to {output_file}")
    elif args.output == 'html':
        from src.reporting.html_generator import HTMLReporter
        reporter = HTMLReporter()
        output_file = os.path.join(output_dir, f"test_results_{args.context}_{timestamp}.html")
        reporter.generate_report(results, output_file)
        logger.info(f"Results saved to {output_file}")
    else:
        # Console output
        for model_id, result in results.items():
            print(f"\n=== Results for {model_id} ===")
            if "error" in result:
                print(f"Error: {result['error']}")
                continue
                
            print(f"Overall score: {result.get('overall_score', 'N/A')}")
            print(f"Test category: {result.get('test_category', 'N/A')}")
            print(f"Context length: {result.get('context_length', 'N/A')}")
            
            if "metrics" in result:
                print("\nMetrics:")
                for metric_name, metric_value in result["metrics"].items():
                    if isinstance(metric_value, dict):
                        print(f"  {metric_name}:")
                        for sub_metric, sub_value in metric_value.items():
                            print(f"    {sub_metric}: {sub_value}")
                    else:
                        print(f"  {metric_name}: {metric_value}")
            
            if "usage" in result:
                print("\nToken usage:")
                for usage_name, usage_value in result["usage"].items():
                    print(f"  {usage_name}: {usage_value}")
            
            if "timing" in result:
                print("\nTiming:")
                for timing_name, timing_value in result["timing"].items():
                    print(f"  {timing_name}: {timing_value:.2f} seconds")
            
            if "cost" in result:
                print(f"\nEstimated cost: ${result['cost']:.6f}")

    # Generate comparison report if multiple models were tested
    if len(valid_models) > 1:
        from src.reporting.comparisons_reporter import ModelComparison
        comparisons_dir = os.path.join(results_dir, "comparisons")
        os.makedirs(comparisons_dir, exist_ok=True)
        
        comparison = ModelComparison(results_dir=results_dir)
        comparison_file = os.path.join(comparisons_dir, f"comparison_{args.test}_{args.context}_{timestamp}.yaml")
        
        try:
            # Format results for comparison
            formatted_results = {}
            for model_id, result in results.items():
                if "error" not in result:
                    formatted_results[model_id] = {
                        "model": model_id,
                        "test_category": args.test,
                        "context_length": args.context,
                        "aggregate_metrics": {
                            "overall_score": result.get("overall_score", 0),
                            "total_tokens": result.get("usage", {}).get("total_tokens", 0),
                            "total_cost_usd": result.get("cost", 0),
                            "average_processing_time_seconds": result.get("timing", {}).get("total_time", 0)
                        }
                    }
            
            # Generate comparison if there are valid results
            if formatted_results:
                comparison.generate_comparison(formatted_results, output_file=comparison_file)
                logger.info(f"Comparison report saved to {comparison_file}")
        except Exception as e:
            logger.error(f"Error generating comparison report: {e}", exc_info=True)
            
    # Generate visualizations
    if len(valid_models) > 1:
        try:
            from src.reporting.visualizations_reporter import Visualizer
            viz_dir = os.path.join(results_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            
            visualizer = Visualizer(results_dir=results_dir)
            visualizer.generate_report(results, viz_dir)
            logger.info(f"Visualizations generated in {viz_dir}")
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}", exc_info=True)

    logger.info("Testing completed successfully")

if __name__ == "__main__":
    main()