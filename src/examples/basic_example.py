import os
import sys
import dspy

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.setup import init_dspy

def main():
    """Simple example to demonstrate DSPy with Anthropic Claude"""
    # Initialize DSPy with Claude
    llm = init_dspy(verbose=True)
    
    # Define a simple signature for question answering
    class SimpleQA(dspy.Signature):
        """Answer questions based on the provided context."""
        context = dspy.InputField(desc="Context information")
        question = dspy.InputField(desc="Question to answer")
        answer = dspy.OutputField(desc="Answer to the question")
    
    # Create a predictor using the signature
    predictor = dspy.Predict(SimpleQA)
    
    # Example usage
    context = "DSPy is a framework for algorithmically optimizing LM prompts and weights."
    question = "What is DSPy used for?"
    
    print(f"Context: {context}")
    print(f"Question: {question}")
    
    # Get the prediction
    result = predictor(context=context, question=question)
    
    print(f"Answer: {result.answer}")

if __name__ == "__main__":
    main() 