import dspy

class BasicQA(dspy.Signature):
    """Answer questions based on the provided context."""
    context = dspy.InputField(desc="Context information to answer the question")
    question = dspy.InputField(desc="Question that needs to be answered")
    answer = dspy.OutputField(desc="Answer to the question based on the context provided")
    
class GeneralQA(dspy.Signature):
    """Answer general questions without specific context."""
    question = dspy.InputField(desc="Question that needs to be answered")
    answer = dspy.OutputField(desc="Answer to the question based on the model's knowledge")

class MessageClassifier(dspy.Signature):
    """Classify messages by intent and extract key information."""
    message = dspy.InputField(desc="The message text to classify")
    intent = dspy.OutputField(desc="The intent of the message (question, greeting, request, etc.)")
    category = dspy.OutputField(desc="The category or topic of the message (travel, personal, technical, etc.)")
    requires_context = dspy.OutputField(desc="Whether additional context is needed to properly respond (yes/no)")
