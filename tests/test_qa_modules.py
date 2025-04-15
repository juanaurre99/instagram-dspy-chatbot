import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dspy_modules import ContextQA, GeneralKnowledgeQA
from src.utils.setup import init_dspy

class TestQAModules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up DSPy for all tests."""
        init_dspy(verbose=False)
    
    def test_context_qa_initialization(self):
        """Test that ContextQA initializes properly."""
        qa_module = ContextQA()
        self.assertIsNotNone(qa_module.predictor)
    
    def test_general_qa_initialization(self):
        """Test that GeneralKnowledgeQA initializes properly."""
        qa_module = GeneralKnowledgeQA()
        self.assertIsNotNone(qa_module.predictor)
    
    def test_context_qa_returns_answer(self):
        """Test that ContextQA returns an answer when given context and question."""
        qa_module = ContextQA()
        context = "The capital of France is Paris."
        question = "What is the capital of France?"
        
        result = qa_module(context=context, question=question)
        
        self.assertIn("answer", result)
        self.assertIsNotNone(result["answer"])
        self.assertIn("full_result", result)
    
    def test_general_qa_returns_answer(self):
        """Test that GeneralKnowledgeQA returns an answer when given a question."""
        qa_module = GeneralKnowledgeQA()
        question = "What is the capital of France?"
        
        result = qa_module(question=question)
        
        self.assertIn("answer", result)
        self.assertIsNotNone(result["answer"])
        self.assertIn("full_result", result)
    
    @unittest.skip("This test requires actual LLM inference which may be slow")
    def test_context_qa_answer_accuracy(self):
        """Test that ContextQA provides accurate answers based on context."""
        qa_module = ContextQA()
        
        test_cases = [
            {
                "context": "The Golden Gate Bridge was opened in 1937.",
                "question": "When was the Golden Gate Bridge opened?",
                "expected_answer_contains": "1937"
            },
            {
                "context": "Alice is 5 years older than Bob. Bob is 3 years younger than Charlie. Charlie is 10 years old.",
                "question": "How old is Alice?",
                "expected_answer_contains": "12"
            }
        ]
        
        for tc in test_cases:
            result = qa_module(context=tc["context"], question=tc["question"])
            self.assertIn(tc["expected_answer_contains"], result["answer"], 
                          f"Expected answer to contain '{tc['expected_answer_contains']}' but got '{result['answer']}'")
    
    @unittest.skip("This test requires actual LLM inference which may be slow")
    def test_general_qa_reasonable_answers(self):
        """Test that GeneralKnowledgeQA provides reasonable answers."""
        qa_module = GeneralKnowledgeQA()
        
        test_cases = [
            {
                "question": "What is the capital of France?",
                "expected_answer_contains": "Paris"
            },
            {
                "question": "Who wrote Romeo and Juliet?",
                "expected_answer_contains": "Shakespeare"
            }
        ]
        
        for tc in test_cases:
            result = qa_module(question=tc["question"])
            self.assertIn(tc["expected_answer_contains"].lower(), result["answer"].lower(), 
                          f"Expected answer to contain '{tc['expected_answer_contains']}' but got '{result['answer']}'")

if __name__ == "__main__":
    unittest.main() 