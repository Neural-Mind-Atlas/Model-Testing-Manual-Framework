#!/usr/bin/env python3
# run_comprehensive_tests.py

import os
import sys
import subprocess
import logging
import time
import json
import yaml
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"comprehensive_test_{timestamp}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("comprehensive_test")

# Define test categories based on the test_suites directory
def get_test_categories():
    """Get test categories from test_suites directory"""
    test_suites_dir = os.path.join("config", "test_suites")
    if not os.path.exists(test_suites_dir):
        logger.warning(f"Test suites directory not found: {test_suites_dir}")
        return [
            "reasoning",
            "factual",
            "hallucination",
            "instruction",
            "context",
            "creative",
            "ppt_writing",
            "meta_prompting",
            "image_prompts"
        ]
    
    test_categories = []
    for filename in os.listdir(test_suites_dir):
        if filename.endswith(".yaml"):
            test_categories.append(filename.replace(".yaml", ""))
    
    if not test_categories:
        logger.warning("No test categories found in test_suites directory")
        return [
            "reasoning",
            "factual",
            "hallucination",
            "instruction",
            "context",
            "creative",
            "ppt_writing",
            "meta_prompting",
            "image_prompts"
        ]
    
    return test_categories

# Define context lengths
CONTEXT_LENGTHS = ["short", "medium", "long"]

# Output formats
OUTPUT_FORMAT = os.getenv("OUTPUT_FORMAT", "html")  # Options: "console", "yaml", "json", "html"

# Whether to use parallel execution
USE_PARALLEL = os.getenv("USE_PARALLEL", "true").lower() == "true"

# Get test categories
TEST_CATEGORIES = get_test_categories()

# Check if specific models should be tested
def get_models_to_test():
    """Get the list of models to test based on environment variables"""
    # Check if specific models are set in environment
    models_env = os.getenv("TEST_MODELS")
    if models_env:
        return models_env.split(",")
    
    # Otherwise test all available models
    return None  # This will trigger --all-models

# Function to format time duration
def format_duration(seconds):
    """Format seconds to a human-readable duration"""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)} minutes {int(seconds)} seconds"
    
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds"

def run_test(test_category, context_length, output_format, use_parallel=False):
    """Run a specific test configuration"""
    
    # Build base command
    cmd = [
        "python", "-m", "src.main"
    ]
    
    # Add model selection
    models_to_test = get_models_to_test()
    if models_to_test:
        cmd.extend(["--models", ",".join(models_to_test)])
    else:
        cmd.append("--all-models")
    
    # Add test parameters
    cmd.extend([
        "--test", test_category,
        "--context", context_length,
        "--output", output_format
    ])
    
    if use_parallel:
        cmd.append("--parallel")
    
    # Add verbosity if set in environment
    if os.getenv("VERBOSE", "false").lower() == "true":
        cmd.append("--verbose")
    
    # Get evaluation model from environment if set
    eval_model = os.getenv("EVAL_MODEL")
    if eval_model:
        cmd.extend(["--eval-model", eval_model])
    
    logger.info(f"Running test: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"Test completed successfully in {format_duration(duration)}")
            return {
                "success": True,
                "duration": duration,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        else:
            logger.error(f"Test failed with exit code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return {
                "success": False,
                "duration": duration,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Exception running test: {str(e)}")
        return {
            "success": False,
            "duration": duration,
            "exception": str(e)
        }

def generate_model_summary(results_dir):
    """Generate a summary of model performance across all tests"""
    summary = {
        "models": {},
        "test_categories": {},
        "overall": {
            "best_model": None,
            "average_score": 0,
            "total_tests": 0,
            "successful_tests": 0
        }
    }
    
    # Try to find all result files
    result_files = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.startswith("test_results_") and (file.endswith(".yaml") or file.endswith(".json")):
                result_files.append(os.path.join(root, file))
    
    if not result_files:
        logger.warning("No result files found for model summary")
        return summary
    
    # Parse result files
    model_scores = {}
    test_category_scores = {}
    total_tests = 0
    successful_tests = 0
    
    for file_path in result_files:
        try:
            # Load results based on file type
            if file_path.endswith(".yaml"):
                with open(file_path, 'r') as f:
                    results = yaml.safe_load(f)
            elif file_path.endswith(".json"):
                with open(file_path, 'r') as f:
                    results = json.load(f)
            else:
                continue
            
            # Extract test category from file path
            path_parts = file_path.split(os.sep)
            test_category = path_parts[-2] if len(path_parts) >= 2 else "unknown"
            
            # Process results
            if results and "results" in results:
                for model_id, result in results["results"].items():
                    # Count tests
                    total_tests += 1
                    
                    # Skip if error
                    if "error" in result:
                        continue
                        
                    successful_tests += 1
                    
                    # Get overall score
                    score = result.get("overall_score", 0)
                    
                    # Update model scores
                    if model_id not in model_scores:
                        model_scores[model_id] = {
                            "total_score": 0,
                            "count": 0,
                            "average": 0,
                            "categories": {}
                        }
                    
                    model_scores[model_id]["total_score"] += score
                    model_scores[model_id]["count"] += 1
                    
                    # Update category for this model
                    if test_category not in model_scores[model_id]["categories"]:
                        model_scores[model_id]["categories"][test_category] = {
                            "total_score": 0,
                            "count": 0,
                            "average": 0
                        }
                    
                    model_scores[model_id]["categories"][test_category]["total_score"] += score
                    model_scores[model_id]["categories"][test_category]["count"] += 1
                    
                    # Update test category scores
                    if test_category not in test_category_scores:
                        test_category_scores[test_category] = {
                            "total_score": 0,
                            "count": 0,
                            "average": 0,
                            "best_model": None,
                            "best_score": 0
                        }
                    
                    test_category_scores[test_category]["total_score"] += score
                    test_category_scores[test_category]["count"] += 1
                    
                    # Check if this is the best score for this category
                    if score > test_category_scores[test_category]["best_score"]:
                        test_category_scores[test_category]["best_score"] = score
                        test_category_scores[test_category]["best_model"] = model_id
                    
        except Exception as e:
            logger.error(f"Error processing result file {file_path}: {str(e)}")
    
    # Calculate averages and find best model
    best_model = None
    best_score = 0
    
    for model_id, data in model_scores.items():
        if data["count"] > 0:
            data["average"] = data["total_score"] / data["count"]
            
            # Update category averages for this model
            for category, cat_data in data["categories"].items():
                if cat_data["count"] > 0:
                    cat_data["average"] = cat_data["total_score"] / cat_data["count"]
            
            # Check if this is the best model
            if data["average"] > best_score:
                best_score = data["average"]
                best_model = model_id
    
    for category, data in test_category_scores.items():
        if data["count"] > 0:
            data["average"] = data["total_score"] / data["count"]
    
    # Update summary
    summary["models"] = model_scores
    summary["test_categories"] = test_category_scores
    summary["overall"]["best_model"] = best_model
    summary["overall"]["average_score"] = best_score
    summary["overall"]["total_tests"] = total_tests
    summary["overall"]["successful_tests"] = successful_tests
    
    return summary

def main():
    """Run comprehensive tests across all categories and context lengths"""
    logger.info("Starting comprehensive testing suite")
    logger.info(f"Test categories: {', '.join(TEST_CATEGORIES)}")
    logger.info(f"Context lengths: {', '.join(CONTEXT_LENGTHS)}")
    logger.info(f"Output format: {OUTPUT_FORMAT}")
    logger.info(f"Parallel execution: {'Enabled' if USE_PARALLEL else 'Disabled'}")
    
    models_to_test = get_models_to_test()
    if models_to_test:
        logger.info(f"Testing specific models: {', '.join(models_to_test)}")
    else:
        logger.info("Testing all available models")
    
    # Create a summary file
    results_dir = os.getenv("RESULTS_DIR", "./results")
    os.makedirs(results_dir, exist_ok=True)
    summary_file = os.path.join(results_dir, f"comprehensive_test_summary_{timestamp}.txt")
    
    total_tests = len(TEST_CATEGORIES) * len(CONTEXT_LENGTHS)
    successful_tests = 0
    failed_tests = []
    test_results = {}
    
    logger.info(f"Starting comprehensive testing of {total_tests} test configurations")
    
    with open(summary_file, "w") as f:
        f.write(f"Comprehensive Test Summary - {datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n\n")
        
        # Write environment information
        f.write("Environment Information:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Python version: {sys.version}\n")
        f.write(f"Output format: {OUTPUT_FORMAT}\n")
        f.write(f"Parallel execution: {'Enabled' if USE_PARALLEL else 'Disabled'}\n")
        f.write(f"Models: {'All' if not models_to_test else ', '.join(models_to_test)}\n\n")
        
        # Track overall start time
        overall_start = time.time()
        
        # Run each test configuration
        for test_category in TEST_CATEGORIES:
            for context_length in CONTEXT_LENGTHS:
                test_config = f"{test_category} - {context_length}"
                
                logger.info(f"Starting test configuration: {test_config}")
                f.write(f"Test: {test_config}\n")
                
                test_start = time.time()
                result = run_test(test_category, context_length, OUTPUT_FORMAT, USE_PARALLEL)
                test_duration = time.time() - test_start
                
                # Store test result
                if test_category not in test_results:
                    test_results[test_category] = {}
                test_results[test_category][context_length] = result
                
                if result["success"]:
                    status = "SUCCESS"
                    successful_tests += 1
                else:
                    status = "FAILED"
                    failed_tests.append(test_config)
                
                f.write(f"  Status: {status}\n")
                f.write(f"  Duration: {format_duration(test_duration)}\n")
                
                # Write error message if failed
                if not result["success"] and "stderr" in result:
                    f.write(f"  Error: {result['stderr'][:500]}...\n" if len(result['stderr']) > 500 
                           else f"  Error: {result['stderr']}\n")
                
                f.write("\n")
                
                # Log progress
                completed = successful_tests + len(failed_tests)
                logger.info(f"Progress: {completed}/{total_tests} tests completed ({(completed/total_tests)*100:.1f}%)")
                
                # Small delay between tests to prevent resource issues
                time.sleep(2)
        
        # Generate model performance summary
        logger.info("Generating model performance summary...")
        model_summary = generate_model_summary(results_dir)
        
        # Write overall statistics
        overall_duration = time.time() - overall_start
        est_completion_time = datetime.now() + timedelta(seconds=overall_duration)
        
        f.write("\nSummary Statistics\n")
        f.write("-" * 40 + "\n")
        f.write(f"Test started: {datetime.fromtimestamp(overall_start).isoformat()}\n")
        f.write(f"Test completed: {datetime.now().isoformat()}\n")
        f.write(f"Total tests: {total_tests}\n")
        f.write(f"Successful tests: {successful_tests}\n")
        f.write(f"Failed tests: {len(failed_tests)}\n")
        f.write(f"Success rate: {(successful_tests/total_tests)*100:.1f}%\n")
        f.write(f"Total duration: {format_duration(overall_duration)}\n\n")
        
        # Write model performance summary if available
        if model_summary["models"]:
            f.write("Model Performance Summary\n")
            f.write("-" * 40 + "\n")
            
            # Write best model overall
            best_model = model_summary["overall"]["best_model"]
            if best_model:
                best_score = model_summary["models"][best_model]["average"]
                f.write(f"Best performing model overall: {best_model} (Score: {best_score:.2f})\n\n")
            
            # Write model rankings
            f.write("Model Rankings:\n")
            ranked_models = sorted(
                [(model_id, data["average"]) for model_id, data in model_summary["models"].items() if data["count"] > 0],
                key=lambda x: x[1],
                reverse=True
            )
            
            for i, (model_id, avg_score) in enumerate(ranked_models, 1):
                f.write(f"  {i}. {model_id}: {avg_score:.2f}\n")
            
            f.write("\n")
            
            # Write best model per category
            f.write("Best Model Per Category:\n")
            for category, data in model_summary["test_categories"].items():
                if data["best_model"]:
                    f.write(f"  {category}: {data['best_model']} (Score: {data['best_score']:.2f})\n")
            
            f.write("\n")
        
        if failed_tests:
            f.write("Failed Tests:\n")
            for test in failed_tests:
                f.write(f"  - {test}\n")
    
    # Save detailed results as JSON for potential further analysis
    detailed_results_file = os.path.join(results_dir, f"comprehensive_test_detailed_{timestamp}.json")
    try:
        with open(detailed_results_file, "w") as f:
            # Clean up results to make them serializable
            clean_results = {}
            for category, contexts in test_results.items():
                clean_results[category] = {}
                for context, result in contexts.items():
                    # Remove stdout/stderr which might be too large
                    cleaned_result = {k: v for k, v in result.items() if k not in ['stdout', 'stderr']}
                    clean_results[category][context] = cleaned_result
            
            json.dump({
                "timestamp": timestamp,
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": len(failed_tests),
                "duration": overall_duration,
                "test_results": clean_results,
                "model_summary": model_summary
            }, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving detailed results: {str(e)}")
    
    logger.info(f"Comprehensive testing completed. Summary saved to {summary_file}")
    logger.info(f"Total duration: {format_duration(overall_duration)}")
    logger.info(f"Success rate: {successful_tests}/{total_tests} ({(successful_tests/total_tests)*100:.1f}%)")
    
    # Return success if all tests passed
    return successful_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)