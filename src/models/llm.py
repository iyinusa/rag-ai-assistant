"""
GROQ LLM model implementation.
"""
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from src.config import settings

def get_llm():
    """
    Initialize the GROQ LLM.
    
    Returns:
        ChatGroq: The GROQ LLM
    """
    # Initialize LLM
    llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL
    )
    
    return llm
