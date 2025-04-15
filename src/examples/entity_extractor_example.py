import os
import sys
import dspy
import json

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.setup import init_dspy
from src.dspy_modules import MessageEntityExtractor

def format_entities(entities, entity_type):
    """Format entities for display."""
    if not entities:
        return f"No {entity_type} found."
    return ", ".join(entities)

def test_entity_extractor():
    """Test the entity extractor with various messages."""
    print("\n----- Testing Entity Extractor Module -----")
    
    extractor = MessageEntityExtractor()
    
    # Example messages
    messages = [
        "I'm planning to visit Barcelona and Madrid next summer with my family.",
        "Can you recommend some good restaurants in New York for my trip next Friday?",
        "I saw John and Mary at the conference in San Francisco last week.",
        "Could you tell me more about your travel packages to Italy and Greece for August 2024?",
        "I'm interested in learning about sustainable travel and eco-friendly accommodations.",
        "My flight to Tokyo departs on December 15th at 9:30 PM.",
        "What did you think about the latest climate change report published by Dr. Smith?",
        "I need to book a hotel in Paris for three nights starting on June 10th.",
        "The best time to visit Japan is during cherry blossom season in April.",
        "I'm organizing a hiking trip to Mount Everest with Sarah and David next spring."
    ]
    
    for i, message in enumerate(messages):
        print(f"\n[{i+1}] Message: \"{message}\"")
        
        result = extractor(message=message)
        
        print("Locations:", format_entities(result["locations"], "locations"))
        print("People:", format_entities(result["people"], "people"))
        print("Dates:", format_entities(result["dates"], "dates"))
        print("Topics:", format_entities(result["topics"], "topics"))
        print("Keywords:", format_entities(result["keywords"], "keywords"))

def main():
    """Run example of the entity extractor module."""
    # Initialize DSPy with Claude
    init_dspy(verbose=True)
    
    # Test the entity extractor
    test_entity_extractor()

if __name__ == "__main__":
    main() 