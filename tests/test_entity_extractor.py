import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dspy_modules import MessageEntityExtractor
from src.utils.setup import init_dspy

class TestEntityExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up DSPy for all tests."""
        init_dspy(verbose=False)
    
    def test_extractor_initialization(self):
        """Test that MessageEntityExtractor initializes properly."""
        extractor = MessageEntityExtractor()
        self.assertIsNotNone(extractor.predictor)
    
    def test_extractor_returns_expected_fields(self):
        """Test that MessageEntityExtractor returns all expected fields."""
        extractor = MessageEntityExtractor()
        message = "I'm going to Paris next week with John."
        
        result = extractor(message=message)
        
        self.assertIn("locations", result)
        self.assertIn("people", result)
        self.assertIn("dates", result)
        self.assertIn("topics", result)
        self.assertIn("keywords", result)
        self.assertIn("full_result", result)
    
    def test_extractor_returns_lists(self):
        """Test that MessageEntityExtractor returns lists for all entity types."""
        extractor = MessageEntityExtractor()
        message = "I'm going to Paris next week with John."
        
        result = extractor(message=message)
        
        self.assertIsInstance(result["locations"], list)
        self.assertIsInstance(result["people"], list)
        self.assertIsInstance(result["dates"], list)
        self.assertIsInstance(result["topics"], list)
        self.assertIsInstance(result["keywords"], list)
    
    def test_extractor_handles_empty_message(self):
        """Test that MessageEntityExtractor handles empty messages."""
        extractor = MessageEntityExtractor()
        message = ""
        
        result = extractor(message=message)
        
        self.assertEqual(result["locations"], [])
        self.assertEqual(result["people"], [])
        self.assertEqual(result["dates"], [])
        self.assertEqual(result["topics"], [])
        self.assertEqual(result["keywords"], [])
    
    def test_extractor_handles_json_parsing_error(self):
        """Test that MessageEntityExtractor handles JSON parsing errors."""
        extractor = MessageEntityExtractor()
        
        # Create a mock prediction object with invalid JSON
        mock_prediction = MagicMock()
        mock_prediction.locations = "invalid json"
        mock_prediction.people = "invalid json"
        mock_prediction.dates = "invalid json"
        mock_prediction.topics = "invalid json"
        mock_prediction.keywords = "invalid json"
        
        # Patch the predictor to return our mock prediction
        with patch.object(extractor, "predictor", return_value=mock_prediction):
            result = extractor(message="test message")
            
            # Verify that the method handles the error and returns empty lists
            self.assertEqual(result["locations"], [])
            self.assertEqual(result["people"], [])
            self.assertEqual(result["dates"], [])
            self.assertEqual(result["topics"], [])
            self.assertEqual(result["keywords"], [])
    
    @unittest.skip("This test requires actual LLM inference which may be slow")
    def test_extractor_finds_locations(self):
        """Test that MessageEntityExtractor correctly identifies locations."""
        extractor = MessageEntityExtractor()
        
        message = "I'm planning to visit Paris and Rome next month."
        result = extractor(message=message)
        
        self.assertIn("Paris", result["locations"])
        self.assertIn("Rome", result["locations"])
    
    @unittest.skip("This test requires actual LLM inference which may be slow")
    def test_extractor_finds_people(self):
        """Test that MessageEntityExtractor correctly identifies people."""
        extractor = MessageEntityExtractor()
        
        message = "I'm meeting with John and Sarah tomorrow."
        result = extractor(message=message)
        
        self.assertIn("John", result["people"])
        self.assertIn("Sarah", result["people"])
    
    @unittest.skip("This test requires actual LLM inference which may be slow")
    def test_extractor_finds_dates(self):
        """Test that MessageEntityExtractor correctly identifies dates."""
        extractor = MessageEntityExtractor()
        
        message = "My flight is on January 15th at 3 PM."
        result = extractor(message=message)
        
        # Either the date or the full date with time should be present
        date_found = any("January 15" in date for date in result["dates"])
        self.assertTrue(date_found)

if __name__ == "__main__":
    unittest.main() 