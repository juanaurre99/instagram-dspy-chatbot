import dspy
from typing import Optional, Dict, Any

from .signatures import BasicQA, GeneralQA

class ContextQA(dspy.Module):
    """
    A question answering module that answers questions based on a provided context.
    """
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(BasicQA)
    
    def forward(self, context: str, question: str) -> Dict[str, Any]:
        """
        Answer a question based on the provided context.
        
        Args:
            context: The context information to use for answering the question
            question: The question to answer
            
        Returns:
            Dict containing the answer
        """
        prediction = self.predictor(context=context, question=question)
        return {"answer": prediction.answer, "full_result": prediction}


class GeneralKnowledgeQA(dspy.Module):
    """
    A question answering module that answers general knowledge questions without specific context.
    """
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(GeneralQA)
    
    def forward(self, question: str) -> Dict[str, Any]:
        """
        Answer a general knowledge question.
        
        Args:
            question: The question to answer
            
        Returns:
            Dict containing the answer
        """
        prediction = self.predictor(question=question)
        return {"answer": prediction.answer, "full_result": prediction}
