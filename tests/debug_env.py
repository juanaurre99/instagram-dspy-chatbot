import os
import sys

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv

def main():
    # Print current directory
    print(f"Current directory: {os.getcwd()}")
    
    # Try to find .env file
    env_path = os.path.join(os.getcwd(), '.env')
    print(f"Looking for .env at: {env_path}")
    print(f"File exists: {os.path.exists(env_path)}")
    
    # Try to load .env and print the API key (masked for security)
    print("Loading .env file...")
    load_dotenv()
    
    api_key = os.getenv("ANTHROPIC_API_KEY", "Not found")
    model = os.getenv("ANTHROPIC_MODEL", "Not found")
    
    # Mask the API key for security
    if api_key != "Not found":
        # Show first 4 and last 4 characters only
        masked_key = api_key[:4] + '*' * (len(api_key) - 8) + api_key[-4:] if len(api_key) > 8 else "***"
        print(f"ANTHROPIC_API_KEY: {masked_key}")
    else:
        print("ANTHROPIC_API_KEY: Not found")
    
    print(f"ANTHROPIC_MODEL: {model}")
    
    # Try to read the .env file directly
    print("\nAttempting to read .env file directly...")
    try:
        with open(env_path, 'r') as f:
            env_content = f.read()
            # Mask API keys in the content
            lines = env_content.split('\n')
            for i, line in enumerate(lines):
                if 'API_KEY' in line and '=' in line:
                    key, value = line.split('=', 1)
                    if value and len(value) > 8:
                        # Show first 4 and last 4 characters only
                        masked_value = value[:4] + '*' * (len(value) - 8) + value[-4:]
                        lines[i] = f"{key}={masked_value}"
            
            masked_content = '\n'.join(lines)
            print(masked_content)
    except Exception as e:
        print(f"Error reading .env file: {e}")

if __name__ == "__main__":
    main() 