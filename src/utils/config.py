# src/utils/config.py
import os
import yaml
import logging

logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load a YAML configuration file"""
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Error loading config file {config_path}: {e}")

    return {}

def load_evaluation_metrics(metrics_path=None):
    """Load evaluation metrics configuration"""
    if metrics_path is None:
        metrics_path = os.path.join("config", "evaluation", "metrics.yaml")
    
    if not os.path.exists(metrics_path):
        # Try alternative path
        metrics_path = os.path.join("config", "evaluation.yaml")

    config = load_config(metrics_path)
    if config and 'metrics' in config:
        return config['metrics']

    return {}

def load_model_config(model_id):
    """Load configuration for a specific model"""
    # First try to find in model-specific files
    model_dir = os.path.join("config", "models")
    
    if not os.path.exists(model_dir):
        logger.warning(f"Model directory {model_dir} does not exist")
        return None
        
    for file_name in os.listdir(model_dir):
        if file_name.endswith('.yaml'):
            file_path = os.path.join(model_dir, file_name)
            config = load_config(file_path)
            
            # Check for list-style model configurations
            if config and 'models' in config and isinstance(config['models'], list):
                for model in config['models']:
                    if model.get('name') == model_id:
                        return model
            
            # Check for dictionary-style model configurations
            elif config and 'models' in config and isinstance(config['models'], dict) and model_id in config['models']:
                return config['models'][model_id]

    # If not found, try the main models.yaml file
    models_config = load_config(os.path.join("config", "models.yaml"))
    if models_config and 'models' in models_config:
        # Check both list and dict formats
        if isinstance(models_config['models'], list):
            for model in models_config['models']:
                if model.get('name') == model_id:
                    return model
        elif isinstance(models_config['models'], dict) and model_id in models_config['models']:
            return models_config['models'][model_id]

    logger.warning(f"No configuration found for model {model_id}")
    return None

def get_model_config(model_id):
    """Get configuration for a specific model"""
    return load_model_config(model_id)

def load_test_suite(suite_name):
    """Load a test suite configuration"""
    suite_path = os.path.join("config", "test_suites", f"{suite_name}.yaml")
    if not os.path.exists(suite_path):
        # Try alternative path
        suite_path = os.path.join("config", "test_suites.yaml")

    config = load_config(suite_path)
    if config and 'test_suites' in config and suite_name in config['test_suites']:
        return config['test_suites'][suite_name]
    elif config:  # The file itself might be the test suite config
        return config

    return None

def load_model_client(model_id, model_config):
    """Load the appropriate client for the model"""
    try:
        provider = model_config.get('provider', '').lower()

        if provider == 'openai':
            from src.clients.openai_client import OpenAIClient
            return OpenAIClient(api_key=os.environ.get('OPENAI_API_KEY'), model_config=model_config)

        elif provider == 'anthropic':
            from src.clients.anthropic_client import AnthropicClient
            return AnthropicClient(api_key=os.environ.get('ANTHROPIC_API_KEY'), model_config=model_config)

        elif provider == 'google':
            from src.clients.google_client import GoogleClient
            return GoogleClient(api_key=os.environ.get('GOOGLE_API_KEY'), model_config=model_config)

        elif provider == 'mistral':
            from src.clients.mistral_client import MistralClient
            return MistralClient(api_key=os.environ.get('MISTRAL_API_KEY'), model_config=model_config)

        elif provider == 'meta':
            from src.clients.meta_client import MetaClient
            return MetaClient(api_key=os.environ.get('META_API_KEY'), model_config=model_config)

        elif provider == 'databricks':
            from src.clients.others import DatabricksClient
            return DatabricksClient(api_key=os.environ.get('DATABRICKS_API_KEY'), model_config=model_config)

        elif provider == 'cohere':
            from src.clients.others import CohereClient
            return CohereClient(api_key=os.environ.get('COHERE_API_KEY'), model_config=model_config)

        else:
            logger.error(f"Unsupported provider: {provider}")
            return None

    except Exception as e:
        logger.error(f"Error creating client for model {model_id}: {e}")
        return None