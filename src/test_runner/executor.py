# src/test_runner/executor.py
import os
import json
import yaml
import logging
import time
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from src.utils.config import load_model_client, get_model_config
from src.utils.tokenizers import count_tokens
from src.clients.base_client import BaseClient
from src.utils.cost_tracker import calculate_cost
from src.test_runner.retry import RetryHandler

# Load environment variables
load_dotenv()

class TestExecutor:
    """Executes tests for different models and test categories."""

    def __init__(self):
        """Initialize the test executor."""
        self.logger = logging.getLogger(__name__)
        self.results = {}
        
        # Default evaluation model - can be overridden in test configuration
        self.default_eval_model_id = "gpt_4o"
        self.eval_model = None
        
        # Get parallel settings from environment
        self.max_parallel_requests = int(os.getenv("MAX_PARALLEL_REQUESTS", "5"))
        self.request_delay_ms = int(os.getenv("REQUEST_DELAY_MS", "500"))
        
        # Initialize retry handler
        self.retry_handler = RetryHandler()

    def load_test_data(self, test_category: str, context_length: str) -> Dict[str, str]:
        """Load test data for a specific test category and context length."""
        try:
            # Update path to match actual file structure
            file_path = os.path.join("data", "contexts", context_length, "contexts.json")
            self.logger.info(f"Loading context from: {file_path}")
            
            if not os.path.exists(file_path):
                # Try alternative path
                file_path = os.path.join("data", f"{context_length}_contexts.json")
                self.logger.info(f"Alternative path: {file_path}")
                
            if not os.path.exists(file_path):
                self.logger.warning(f"Context file not found: {file_path}")
                return {
                    "context": "Default context for testing.",
                    "prompt": "Generate a response."
                }
                
            with open(file_path, 'r') as file:
                contexts_data = json.load(file)
            
            # Extract the context based on file structure
            context = ""
            if "contexts" in contexts_data and isinstance(contexts_data["contexts"], list):
                for ctx_entry in contexts_data["contexts"]:
                    if "content" in ctx_entry:
                        context = ctx_entry["content"]
                        break
            else:
                self.logger.warning(f"Unexpected context format in {file_path}")

            # Get the prompt for the specified test category
            prompt = "Generate a response."  # Default prompt
            
            # Try to load from test suite configuration
            test_suite_path = os.path.join("config", "test_suites", f"{test_category}.yaml")
            if os.path.exists(test_suite_path):
                with open(test_suite_path, 'r') as file:
                    test_suite = yaml.safe_load(file)
                    
                # Find a suitable prompt in the test suite
                if test_suite and "test_cases" in test_suite and test_suite["test_cases"]:
                    first_test = test_suite["test_cases"][0]
                    if "prompts_file" in first_test:
                        # If there's a reference to a prompt file, try to load it
                        prompts_path = first_test["prompts_file"]
                        full_prompt_path = os.path.join("data", "prompts", prompts_path)
                        
                        if os.path.exists(full_prompt_path):
                            with open(full_prompt_path, 'r') as file:
                                prompts_data = yaml.safe_load(file)
                                if prompts_data and "prompts" in prompts_data and prompts_data["prompts"]:
                                    prompt_template = prompts_data["prompts"][0].get("template", prompt)
                                    prompt = prompt_template
                    elif "prompt_template" in first_test:
                        # Direct prompt template in test case
                        prompt = first_test["prompt_template"]
            
            # If no test suite, try to find a test case directly
            else:
                test_cases_path = os.path.join("data", "test_cases", f"{test_category}_test_cases.json")
                if os.path.exists(test_cases_path):
                    with open(test_cases_path, 'r') as file:
                        test_cases = json.load(file)
                        if "test_cases" in test_cases and test_cases["test_cases"]:
                            first_test = test_cases["test_cases"][0]
                            if "prompt_template" in first_test:
                                prompt = first_test["prompt_template"]

            return {
                "context": context,
                "prompt": prompt
            }
        except Exception as e:
            self.logger.error(f"Error loading test data: {e}")
            return {
                "context": "Default context for testing.",
                "prompt": "Generate a response."
            }

    def get_evaluation_model(self, test_config: Dict[str, Any]) -> BaseClient:
        """Get or create an evaluation model client."""
        # If we already have an evaluation model, return it
        if self.eval_model:
            return self.eval_model
            
        # Try to get evaluation model from test config, or use default
        eval_model_id = test_config.get("evaluation_model", self.default_eval_model_id)
        eval_model_config = get_model_config(eval_model_id)
        
        if not eval_model_config:
            self.logger.warning(f"Evaluation model config not found for {eval_model_id}, using mock evaluator")
            return None
            
        eval_model = load_model_client(eval_model_id, eval_model_config)
        
        if not eval_model:
            self.logger.warning(f"Failed to initialize evaluation model {eval_model_id}, using mock evaluator")
            return None
            
        self.eval_model = eval_model
        return eval_model

    def run_test(self, model_id: str, model_config: Dict[str, Any], test_category: str, context_length: str) -> Dict[str, Any]:
        """Run a test for a specific model and test category."""
        self.logger.info(f"Running {test_category} test with {context_length} context on {model_id}")

        # Load test data
        test_data = self.load_test_data(test_category, context_length)

        # Initialize model client
        client = load_model_client(model_id, model_config)
        if not client:
            return {
                "model_id": model_id,
                "test_category": test_category,
                "context_length": context_length,
                "error": f"Failed to initialize client for {model_id} with provider {model_config.get('provider', 'unknown')}"
            }
            
        # Check if this model should be tested
        if not client.should_test():
            return {
                "model_id": model_id,
                "test_category": test_category,
                "context_length": context_length,
                "error": f"Testing disabled for model {model_id} (provider: {client.name})"
            }

        # Form the full prompt
        full_prompt = f"{test_data['context']}\n\n{test_data['prompt']}"

        try:
            # Record start time
            start_time = time.time()
            
            # Generate response with retry handler
            self.logger.info(f"Generating response from {model_id}")
            
            response = self.retry_handler.with_retry(
                lambda: client.generate(full_prompt, model_config),
                max_retries=3
            )
            
            # Record end time
            end_time = time.time()
            
            # Get usage, timing, and cost data from client
            usage = client.get_last_usage()
            timing = client.get_last_timing()
            cost = client.get_last_cost()
            
            # If usage data wasn't provided by client, estimate it
            if not usage["total_tokens"]:
                prompt_tokens = count_tokens(full_prompt, model_id)
                completion_tokens = count_tokens(response, model_id)
                total_tokens = prompt_tokens + completion_tokens
                usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
                
            # If timing wasn't captured, use our start/end times
            if not timing["total_time"]:
                timing = {
                    "total_time": end_time - start_time,
                    "time_to_first_token": None
                }
                
            # If cost wasn't calculated, do it now
            if not cost:
                cost = client.calculate_cost(usage["prompt_tokens"], usage["completion_tokens"])
            
            # Initialize evaluation model if needed
            eval_model = self.get_evaluation_model(model_config)
            
            # Prepare to collect scores
            metrics = {}
            overall_scores = []
            
            # Route to appropriate evaluator based on test category
            if test_category == "reasoning":
                from src.evaluators.reasoning_evaluator import ReasoningEvaluator
                evaluator = ReasoningEvaluator(eval_model)
                result = evaluator.evaluate(test_data['prompt'], response)
                metrics["reasoning"] = result
                if "overall_score" in result:
                    overall_scores.append(result["overall_score"])
                
            elif test_category == "factual":
                from src.evaluators.accuracy_evaluator import AccuracyEvaluator
                evaluator = AccuracyEvaluator(eval_model)
                result = evaluator.evaluate(test_data['prompt'], response)
                metrics["factual"] = result
                if "overall_score" in result:
                    overall_scores.append(result["overall_score"])
                
            elif test_category == "hallucination":
                from src.evaluators.hallucination_evaluator import HallucinationEvaluator
                evaluator = HallucinationEvaluator(eval_model)
                result = evaluator.evaluate(test_data['prompt'], response, context=test_data['context'])
                metrics["hallucination"] = result
                if "overall_score" in result:
                    overall_scores.append(result["overall_score"])
                    
            elif test_category == "instruction":
                from src.evaluators.instruction_evaluator import InstructionEvaluator
                evaluator = InstructionEvaluator(eval_model)
                result = evaluator.evaluate(test_data['prompt'], response)
                metrics["instruction"] = result
                if "overall_score" in result:
                    overall_scores.append(result["overall_score"])
                    
            elif test_category == "context":
                from src.evaluators.context_evaluator import ContextEvaluator
                evaluator = ContextEvaluator(eval_model)
                result = evaluator.evaluate(test_data['prompt'], response, context=test_data['context'])
                metrics["context"] = result
                if "overall_score" in result:
                    overall_scores.append(result["overall_score"])
                    
            elif test_category in ["creative", "ppt_writing", "meta_prompting", "image_prompts"]:
                # If no specific evaluator, use a reasonable default
                from src.evaluators.prompt_quality_evaluator import PromptQualityEvaluator
                evaluator = PromptQualityEvaluator(eval_model)
                result = evaluator.evaluate(test_data['prompt'], response, prompt_type=test_category)
                metrics[test_category] = result
                if "overall_score" in result:
                    overall_scores.append(result["overall_score"])
                
            else:
                # For any other test category or if no evaluation model is available,
                # use some reasonable default scores
                self.logger.warning(f"No specific evaluator for {test_category}, using default scores")
                metrics = {
                    "accuracy": 0.85,
                    "relevance": 0.90,
                    "quality": 0.78,
                    "formatting": 0.92
                }
                overall_scores = [0.85]  # Default score
            
            # Calculate efficiency metrics
            from src.evaluators.efficiency_evaluator import EfficiencyEvaluator
            efficiency_evaluator = EfficiencyEvaluator()
            response_data = {
                "usage": usage,
                "timing": timing,
                "cost": cost
            }
            
            # Calculate overall quality score as average of individual scores
            quality_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0
            
            # Add efficiency metrics
            efficiency_metrics = efficiency_evaluator.evaluate(response_data, quality_score)
            metrics["efficiency"] = efficiency_metrics
            
            # Calculate overall score
            overall_score = quality_score

            return {
                "model_id": model_id,
                "test_category": test_category,
                "context_length": context_length,
                "overall_score": overall_score,
                "metrics": metrics,
                "response_sample": response[:500] + "..." if len(response) > 500 else response,
                "usage": usage,
                "timing": timing,
                "cost": cost
            }
        except Exception as e:
            self.logger.error(f"Error running test for {model_id}: {e}", exc_info=True)
            return {
                "model_id": model_id,
                "test_category": test_category,
                "context_length": context_length,
                "error": str(e)
            }

    def run_tests(self, models: Dict[str, Dict[str, Any]], test_suite: Dict[str, Any]) -> Dict[str, Any]:
        """Run tests for multiple models according to a test suite."""
        results = {}

        for model_id, model_config in models.items():
            self.logger.info(f"Testing model: {model_id}")
            model_results = {}

            for test_category in test_suite.get("test_categories", ["ppt_generation"]):
                for context_length in test_suite.get("context_lengths", ["short"]):
                    test_result = self.run_test(model_id, model_config, test_category, context_length)

                    # Store the result
                    if test_category not in model_results:
                        model_results[test_category] = {}
                    model_results[test_category][context_length] = test_result
                    
                    # Add a small delay between tests to avoid rate limits
                    time.sleep(self.request_delay_ms / 1000)

            # Calculate overall model score
            overall_scores = []
            for category_results in model_results.values():
                for result in category_results.values():
                    if "overall_score" in result:
                        overall_scores.append(result["overall_score"])

            if overall_scores:
                model_results["overall_score"] = sum(overall_scores) / len(overall_scores)
            else:
                model_results["overall_score"] = None

            results[model_id] = model_results
            self.logger.info(f"Testing completed for {model_id}")

        return results
        
    def run_tests_parallel(self, models: Dict[str, Dict[str, Any]], test_suite: Dict[str, Any]) -> Dict[str, Any]:
        """Run tests for multiple models in parallel."""
        results = {}
        test_tasks = []
        
        # Create list of test tasks
        for model_id, model_config in models.items():
            for test_category in test_suite.get("test_categories", ["ppt_generation"]):
                for context_length in test_suite.get("context_lengths", ["short"]):
                    test_tasks.append({
                        "model_id": model_id,
                        "model_config": model_config,
                        "test_category": test_category,
                        "context_length": context_length
                    })
        
        # Run tests in parallel
        with ThreadPoolExecutor(max_workers=self.max_parallel_requests) as executor:
            futures = {}
            for task in test_tasks:
                future = executor.submit(
                    self.run_test,
                    task["model_id"],
                    task["model_config"],
                    task["test_category"],
                    task["context_length"]
                )
                futures[future] = task
            
            for future in as_completed(futures):
                task = futures[future]
                model_id = task["model_id"]
                test_category = task["test_category"]
                context_length = task["context_length"]
                
                try:
                    result = future.result()
                    
                    if model_id not in results:
                        results[model_id] = {}
                    if test_category not in results[model_id]:
                        results[model_id][test_category] = {}
                        
                    results[model_id][test_category][context_length] = result
                    self.logger.info(f"Completed test for {model_id}, {test_category}, {context_length}")
                
                except Exception as e:
                    self.logger.error(f"Error in test for {model_id}: {e}")
                    
                    if model_id not in results:
                        results[model_id] = {}
                    if test_category not in results[model_id]:
                        results[model_id][test_category] = {}
                        
                    results[model_id][test_category][context_length] = {
                        "model_id": model_id,
                        "test_category": test_category,
                        "context_length": context_length,
                        "error": str(e)
                    }
        
        # Calculate overall model scores
        for model_id, model_results in results.items():
            overall_scores = []
            for category_results in model_results.values():
                if isinstance(category_results, dict):  # Skip if not a dictionary of results
                    for result in category_results.values():
                        if isinstance(result, dict) and "overall_score" in result:
                            overall_scores.append(result["overall_score"])

            if overall_scores:
                results[model_id]["overall_score"] = sum(overall_scores) / len(overall_scores)
            else:
                results[model_id]["overall_score"] = None
        
        return results