import os
import sys
import dspy

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.setup import init_dspy
from src.dspy_modules import MessageIntentClassifier

def test_message_classifier():
    """Test the message classifier with various types of messages."""
    print("\n----- Testing Message Classifier Module -----")
    
    classifier = MessageIntentClassifier()
    
    # Example messages to classify
    messages = [
        "Hi there, how are you doing today?",
        "What's the weather like in Barcelona?",
        "Can you tell me about the best places to visit in Paris?",
        "I'm having trouble with my account, can you help me fix it?",
        "Book a flight to New York for next Friday",
        "I'd like to know more about your travel packages to Italy",
        "What did you mean in your last post about sustainable travel?",
        "Thank you for your help yesterday!",
        "Can you share some travel tips for budget travelers?",
        "When is the best time to visit Japan?"
    ]
    
    for i, message in enumerate(messages):
        print(f"\n[{i+1}] Message: \"{message}\"")
        
        result = classifier(message=message)
        
        print(f"Intent: {result['intent']}")
        print(f"Category: {result['category']}")
        print(f"Requires Context: {'Yes' if result['requires_context'] else 'No'}")

def main():
    """Run example of the message classifier module."""
    # Initialize DSPy with Claude
    init_dspy(verbose=True)
    
    # Test the message classifier
    test_message_classifier()

if __name__ == "__main__":
    main() 