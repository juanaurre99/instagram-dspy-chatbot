import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.setup import load_environment, init_dspy, get_llm

class TestSetup(unittest.TestCase):
    @patch('src.utils.setup.load_dotenv')
    def test_load_environment_success(self, mock_load_dotenv):
        """Test that load_environment calls load_dotenv"""
        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': 'test_key',
            'ANTHROPIC_MODEL': 'test_model'
        }):
            load_environment()
            mock_load_dotenv.assert_called_once()
    
    @patch('src.utils.setup.load_dotenv')
    def test_load_environment_missing_vars(self, mock_load_dotenv):
        """Test that load_environment raises an error when environment variables are missing"""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(EnvironmentError):
                load_environment()
    
    @patch('src.utils.setup.dspy.LM')
    @patch('src.utils.setup.load_environment')
    def test_init_dspy(self, mock_load_environment, mock_lm):
        """Test that init_dspy initializes DSPy with the correct parameters"""
        mock_llm = MagicMock()
        mock_lm.return_value = mock_llm
        
        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': 'test_key',
            'ANTHROPIC_MODEL': 'test_model',
            'DSPY_VERBOSE': 'True'
        }):
            result = init_dspy()
            
            mock_load_environment.assert_called_once()
            mock_lm.assert_called_once_with(
                "test_model",
                api_key='test_key'
            )
            self.assertEqual(result, mock_llm)
    
    @patch('src.utils.setup.init_dspy')
    @patch('src.utils.setup.dspy')
    def test_get_llm_initialization(self, mock_dspy, mock_init_dspy):
        """Test that get_llm initializes the LLM if not already initialized"""
        # Simulate uninitialized settings
        mock_dspy.settings = MagicMock()
        delattr(mock_dspy.settings, 'lm')
        
        mock_llm = MagicMock()
        mock_init_dspy.return_value = mock_llm
        
        result = get_llm()
        
        mock_init_dspy.assert_called_once()
        self.assertEqual(result, mock_llm)
    
    @patch('src.utils.setup.init_dspy')
    @patch('src.utils.setup.dspy')
    def test_get_llm_already_initialized(self, mock_dspy, mock_init_dspy):
        """Test that get_llm returns the existing LLM if already initialized"""
        # Simulate initialized settings
        mock_llm = MagicMock()
        mock_dspy.settings = MagicMock()
        mock_dspy.settings.lm = mock_llm
        
        result = get_llm()
        
        mock_init_dspy.assert_not_called()
        self.assertEqual(result, mock_llm)

if __name__ == '__main__':
    unittest.main() 