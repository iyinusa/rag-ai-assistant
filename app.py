"""
RAG-based AI Publication Agent for answering questions based on documents.
"""
import os
import sys

from src.utils.document_loader import load_documents, split_documents
from src.database.vector_store import create_vector_store, load_vector_store
from src.models.rag import create_rag_chain
from src.config import settings

def init_vector_db():
    """
    Initialize the vector database with documents.
    
    Returns:
        The vector store object
    """
    print("Loading documents...")
    documents = load_documents()
    
    if not documents:
        print(f"No documents found in {settings.DOCS_DIRECTORY}. Please add some .txt files.")
        return None
    
    print(f"Found {len(documents)} documents. Processing...")
    split_docs = split_documents(documents)
    print(f"Documents split into {len(split_docs)} chunks.")
    
    print("Creating vector store...")
    db = create_vector_store(split_docs)
    print("Vector store created successfully.")
    
    return db

def get_vector_db():
    """
    Get the vector database, creating it if it doesn't exist.
    
    Returns:
        The vector store object
    """
    db = load_vector_store()
    
    if db is None:
        db = init_vector_db()
    
    return db

def query_agent(question):
    """
    Query the RAG agent with a question.
    
    Args:
        question: The question to ask
        
    Returns:
        str: The agent's response
    """
    # Get the vector database
    vector_db = get_vector_db()
    
    if vector_db is None:
        return "No documents available to answer questions. Please add some documents to the docs directory."
    
    # Create the RAG chain
    rag_chain = create_rag_chain(vector_db)
    
    # Query the chain with direct question input
    response = rag_chain.invoke(question)
    
    # Handle different output formats based on the response structure
    if isinstance(response, dict) and "answer" in response:
        return response["answer"]
    elif isinstance(response, str):
        return response
    else:
        # For other cases, return the whole response as string
        return str(response)

def run_cli():
    """Run the agent in CLI mode."""
    print("\nRAG-based AI Publication Agent")
    print("------------------------------")
    print("Type 'exit' or 'quit' to exit the program.\n")
    
    while True:
        question = input("\nEnter your question: ")
        
        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        print("\nThinking...")
        answer = query_agent(question)
        print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    run_cli()