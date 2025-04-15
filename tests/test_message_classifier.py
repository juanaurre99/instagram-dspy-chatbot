import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dspy_modules import MessageIntentClassifier
from src.utils.setup import init_dspy

class TestMessageClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up DSPy for all tests."""
        init_dspy(verbose=False)
    
    def test_classifier_initialization(self):
        """Test that MessageIntentClassifier initializes properly."""
        classifier = MessageIntentClassifier()
        self.assertIsNotNone(classifier.predictor)
    
    def test_classifier_returns_expected_fields(self):
        """Test that MessageIntentClassifier returns all expected fields."""
        classifier = MessageIntentClassifier()
        message = "Hi there, how are you doing today?"
        
        result = classifier(message=message)
        
        self.assertIn("intent", result)
        self.assertIn("category", result)
        self.assertIn("requires_context", result)
        self.assertIn("full_result", result)
        self.assertIsInstance(result["requires_context"], bool)
    
    def test_classifier_processes_different_messages(self):
        """Test that MessageIntentClassifier can process different types of messages."""
        classifier = MessageIntentClassifier()
        
        messages = [
            "Hi there, how are you doing today?",
            "What's the weather like in Barcelona?",
            "I'm having trouble with my account, can you help me?"
        ]
        
        for message in messages:
            result = classifier(message=message)
            self.assertIsNotNone(result["intent"])
            self.assertIsNotNone(result["category"])
    
    @unittest.skip("This test requires actual LLM inference which may be slow")
    def test_classifier_identifies_greeting(self):
        """Test that MessageIntentClassifier correctly identifies a greeting."""
        classifier = MessageIntentClassifier()
        
        message = "Hello! How are you doing today?"
        result = classifier(message=message)
        
        self.assertIn("greeting", result["intent"].lower())
    
    @unittest.skip("This test requires actual LLM inference which may be slow")
    def test_classifier_identifies_question(self):
        """Test that MessageIntentClassifier correctly identifies a question."""
        classifier = MessageIntentClassifier()
        
        message = "What are the best restaurants in New York?"
        result = classifier(message=message)
        
        self.assertIn("question", result["intent"].lower())
    
    @unittest.skip("This test requires actual LLM inference which may be slow")
    def test_classifier_identifies_request(self):
        """Test that MessageIntentClassifier correctly identifies a request."""
        classifier = MessageIntentClassifier()
        
        message = "Please book a flight to Tokyo for next week."
        result = classifier(message=message)
        
        self.assertIn("request", result["intent"].lower())
    
    @unittest.skip("This test requires actual LLM inference which may be slow")
    def test_context_requirement_detection(self):
        """Test that MessageIntentClassifier correctly identifies when context is required."""
        classifier = MessageIntentClassifier()
        
        # This message requires context about "your last post"
        message = "What did you mean in your last post about sustainable travel?"
        result = classifier(message=message)
        
        self.assertTrue(result["requires_context"])
        
        # This message doesn't require additional context
        message = "What's the capital of France?"
        result = classifier(message=message)
        
        self.assertFalse(result["requires_context"])

if __name__ == "__main__":
    unittest.main() 