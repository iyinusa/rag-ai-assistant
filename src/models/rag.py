"""
RAG (Retrieval Augmented Generation) implementation.
"""
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser

from src.config import settings
from src.models.llm import get_llm

def create_rag_chain(vector_store):
    """
    Create a RAG chain with the vector store and LLM.
    
    Args:
        vector_store: The vector database
        
    Returns:
        Chain: The RAG chain
    """
    # Initialize LLM
    llm = get_llm()
    
    # Create a retriever from the vector store
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_template(settings.QUERY_PROMPT_TEMPLATE)
    
    # Create the document chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create the RAG chain
    from langchain_core.runnables import RunnablePassthrough
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} 
        | document_chain
    )
    
    return rag_chain
