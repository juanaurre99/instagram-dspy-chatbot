"""
RAG (Retrieval-Augmented Generation) components for the Instagram DSPy chatbot.

This module provides classes for loading, processing, embedding, and retrieving 
documents from a knowledge base to augment LLM responses with external knowledge.
"""

from .document_processor import Document, DocumentChunk, DocumentLoader, DocumentChunker, MetadataExtractor
from .embeddings import EmbeddingGenerator
from .indexer import VectorStoreIndexer
from .retriever import QueryTransformer, Retriever

__all__ = [
    'Document',
    'DocumentChunk',
    'DocumentLoader',
    'DocumentChunker',
    'MetadataExtractor',
    'EmbeddingGenerator',
    'VectorStoreIndexer',
    'QueryTransformer',
    'Retriever',
]
