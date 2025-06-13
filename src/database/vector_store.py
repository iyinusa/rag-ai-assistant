"""
Vector database module using ChromaDB.
"""
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings

def get_embedding_model():
    """
    Get the HuggingFace embedding model.
    
    Returns:
        HuggingFaceEmbeddings: The embedding model
    """
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    return embeddings

def create_vector_store(documents):
    """
    Create a vector store from documents.
    
    Args:
        documents: The documents to add to the vector store
        
    Returns:
        Chroma: The vector store
    """
    # Initialize embedding model
    embeddings = get_embedding_model()
    
    # Create directory if it doesn't exist
    if not os.path.exists(settings.CHROMA_PERSIST_DIRECTORY):
        os.makedirs(settings.CHROMA_PERSIST_DIRECTORY)
    
    # Create vector store
    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=settings.CHROMA_PERSIST_DIRECTORY
    )
    
    # Persist the database
    db.persist()
    
    return db

def load_vector_store():
    """
    Load an existing vector store.
    
    Returns:
        Chroma: The vector store or None if it doesn't exist
    """
    # Initialize embedding model
    embeddings = get_embedding_model()
    
    # Check if the directory exists
    if not os.path.exists(settings.CHROMA_PERSIST_DIRECTORY):
        return None
    
    # Load vector store
    db = Chroma(
        persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
        embedding_function=embeddings
    )
    
    return db
