import os
import dspy
from typing import Dict, Optional

def read_env_file() -> Dict[str, str]:
    """
    Read environment variables directly from .env file
    instead of using dotenv which seems to be loading incorrect values
    """
    # Get the absolute path to the .env file
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    env_path = os.path.join(base_dir, '.env')
    
    env_vars = {}
    try:
        with open(env_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip()
    except Exception as e:
        print(f"Error reading .env file: {e}")
    
    return env_vars

def load_environment() -> Dict[str, str]:
    """Load environment variables from .env file"""
    # Read variables directly from file
    env_vars = read_env_file()
    
    # Check if the required environment variables are in the parsed env_vars
    required_vars = ["ANTHROPIC_API_KEY", "ANTHROPIC_MODEL"]
    missing_vars = [var for var in required_vars if var not in env_vars]
    
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return env_vars

def init_dspy(verbose: bool = None):
    """
    Initialize DSPy with Anthropic Claude LLM
    
    Args:
        verbose: Override the DSPY_VERBOSE setting from .env
        
    Returns:
        The configured DSPy LM
    """
    # Load environment variables directly from file
    env_vars = load_environment()
    
    # Get configuration from environment vars
    api_key = env_vars.get("ANTHROPIC_API_KEY")
    model_name = env_vars.get("ANTHROPIC_MODEL")
    dspy_verbose = verbose if verbose is not None else env_vars.get("DSPY_VERBOSE", "False").lower() == "true"
    
    # Print information about the configuration (masked for security)
    if api_key:
        masked_key = api_key[:4] + '*' * (len(api_key) - 8) + api_key[-4:] if len(api_key) > 8 else "***"
        print(f"Using API key: {masked_key}")
    print(f"Using model: {model_name}")
    
    # Configure DSPy with model name
    llm = dspy.LM(model_name, api_key=api_key)
    
    # Set the LLM as the default for DSPy
    dspy.configure(lm=llm, verbose=dspy_verbose)
    
    return llm

def get_llm():
    """
    Get the configured DSPy LLM instance
    
    Returns:
        The configured LLM
    """
    if not hasattr(dspy.settings, "lm") or dspy.settings.lm is None:
        return init_dspy()
    return dspy.settings.lm

if __name__ == "__main__":
    print(os.getenv("ANTHROPIC_API_KEY"))