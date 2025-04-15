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
