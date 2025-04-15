import dspy
import json
from typing import Optional, Dict, Any, List

from .signatures import BasicQA, GeneralQA, MessageClassifier, EntityExtractor

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


class MessageIntentClassifier(dspy.Module):
    """
    A module that classifies messages by intent and extracts key information.
    """
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(MessageClassifier)
    
    def forward(self, message: str) -> Dict[str, Any]:
        """
        Classify a message by intent and extract key information.
        
        Args:
            message: The message to classify
            
        Returns:
            Dict containing the classification results
        """
        prediction = self.predictor(message=message)
        
        return {
            "intent": prediction.intent,
            "category": prediction.category,
            "requires_context": prediction.requires_context.lower() == "yes",
            "full_result": prediction
        }


class MessageEntityExtractor(dspy.Module):
    """
    A module that extracts entities from a message such as locations, people, dates, topics and keywords.
    """
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(EntityExtractor)
    
    def forward(self, message: str) -> Dict[str, Any]:
        """
        Extract entities from a message.
        
        Args:
            message: The message to extract entities from
            
        Returns:
            Dict containing the extracted entities
        """
        prediction = self.predictor(message=message)
        
        # Parse JSON strings into Python lists
        try:
            locations = json.loads(prediction.locations)
            people = json.loads(prediction.people)
            dates = json.loads(prediction.dates)
            topics = json.loads(prediction.topics)
            keywords = json.loads(prediction.keywords)
        except json.JSONDecodeError as e:
            # Handle parsing errors gracefully
            print(f"Error parsing entity JSON: {e}")
            locations = []
            people = []
            dates = []
            topics = []
            keywords = []
            
        return {
            "locations": locations,
            "people": people,
            "dates": dates,
            "topics": topics,
            "keywords": keywords,
            "full_result": prediction
        }
