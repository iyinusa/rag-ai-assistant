"""
Document loading utilities for processing text files in the docs directory.
"""
import os
from typing import List
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.config import settings

def load_documents():
    """
    Load all text documents from the docs directory.
    
    Returns:
        List[Document]: The loaded documents
    """
    if not os.path.exists(settings.DOCS_DIRECTORY):
        os.makedirs(settings.DOCS_DIRECTORY)
        print(f"Created documents directory at {settings.DOCS_DIRECTORY}")
        return []
    
    # Check if there are any text files
    txt_files = [f for f in os.listdir(settings.DOCS_DIRECTORY) 
               if f.endswith('.txt') and os.path.isfile(os.path.join(settings.DOCS_DIRECTORY, f))]
    
    if not txt_files:
        print(f"No .txt files found in {settings.DOCS_DIRECTORY}")
        return []
        
    # Load the documents
    loader = DirectoryLoader(
        settings.DOCS_DIRECTORY,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True}
    )
    
    documents = loader.load()
    
    if documents:
        print(f"Loaded {len(documents)} documents from {settings.DOCS_DIRECTORY}")
        
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into chunks for better processing.
    
    Args:
        documents: The documents to split
        chunk_size: The size of each chunk
        chunk_overlap: The overlap between chunks
        
    Returns:
        List[Document]: The split documents
    """
    if not documents:
        return []
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    split_docs = text_splitter.split_documents(documents)
    return split_docs
