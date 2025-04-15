import os
import sys
import dspy

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.setup import init_dspy
from src.dspy_modules import ContextQA, GeneralKnowledgeQA

def test_context_qa():
    """Test the ContextQA module with a sample context and question."""
    print("\n----- Testing ContextQA Module -----")
    
    context_qa = ContextQA()
    
    # Example 1: Simple factual QA
    context = """
    The Golden Gate Bridge is a suspension bridge spanning the Golden Gate, 
    the one-mile-wide strait connecting San Francisco Bay and the Pacific Ocean. 
    The structure links the U.S. city of San Francisco, California—the northern tip of 
    the San Francisco Peninsula—to Marin County. It was opened in 1937 and had the world's 
    longest suspension bridge main span at that time.
    """
    question = "When was the Golden Gate Bridge opened?"
    
    print(f"Context: {context.strip()}")
    print(f"Question: {question}")
    
    result = context_qa(context=context, question=question)
    
    print(f"Answer: {result['answer']}")
    
    # Example 2: Reasoning QA
    context = """
    Alice is 5 years older than Bob. Bob is 3 years younger than Charlie.
    Charlie is 10 years old.
    """
    question = "How old is Alice?"
    
    print("\n")
    print(f"Context: {context.strip()}")
    print(f"Question: {question}")
    
    result = context_qa(context=context, question=question)
    
    print(f"Answer: {result['answer']}")
    
def test_general_qa():
    """Test the GeneralKnowledgeQA module with sample questions."""
    print("\n----- Testing GeneralKnowledgeQA Module -----")
    
    general_qa = GeneralKnowledgeQA()
    
    # Example 1: General knowledge question
    question = "What is the capital of France?"
    
    print(f"Question: {question}")
    
    result = general_qa(question=question)
    
    print(f"Answer: {result['answer']}")
    
    # Example 2: More complex question
    question = "How does photosynthesis work?"
    
    print("\n")
    print(f"Question: {question}")
    
    result = general_qa(question=question)
    
    print(f"Answer: {result['answer']}")

def main():
    """Run examples of the QA modules."""
    # Initialize DSPy with Claude
    init_dspy(verbose=True)
    
    # Test the QA modules
    test_context_qa()
    test_general_qa()

if __name__ == "__main__":
    main() 