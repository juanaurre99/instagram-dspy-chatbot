import os
import logging
import numpy as np
from typing import List, Dict, Any, Union, Optional, Tuple
from pathlib import Path

# For type hints
from .document_processor import Document, DocumentChunk
from .embeddings import EmbeddingGenerator
from .indexer import VectorStoreIndexer

logger = logging.getLogger(__name__)

class QueryTransformer:
    """Transforms user queries for optimal retrieval."""
    
    def __init__(self):
        """Initialize the query transformer."""
        pass
    
    def transform_query(self, query: str) -> str:
        """
        Transform a query for better retrieval.
        
        Args:
            query: The original user query
            
        Returns:
            Transformed query
        """
        # In a real implementation, this would add search context, 
        # expand the query, etc. For now, we'll just return the original.
        return query

class Retriever:
    """Retrieves relevant document chunks based on a query."""
    
    def __init__(
        self,
        indexer: Optional[VectorStoreIndexer] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        max_results: int = 5,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize the retriever.
        
        Args:
            indexer: Vector store indexer to use for retrieval
            embedding_generator: Embedding generator to use for query embedding
            max_results: Maximum number of results to return
            similarity_threshold: Minimum similarity score for results
        """
        self.indexer = indexer
        self.embedding_generator = embedding_generator
        self.max_results = max_results
        self.similarity_threshold = similarity_threshold
        self.query_transformer = QueryTransformer()
    
    def retrieve(
        self, 
        query: str,
        filter_criteria: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        include_documents: bool = True,
        rerank: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks based on a query.
        
        Args:
            query: The query to search for
            filter_criteria: Criteria to filter results (e.g., content_type, category)
            include_metadata: Whether to include metadata in results
            include_documents: Whether to include document text in results
            rerank: Whether to rerank results (not implemented yet)
            
        Returns:
            List of relevant document chunks with similarity scores
        """
        if self.indexer is None:
            self.indexer = VectorStoreIndexer()
            self.indexer._get_collection()
        
        # Transform the query for better retrieval
        transformed_query = self.query_transformer.transform_query(query)
        
        # Set up includes for the query
        includes = []
        if include_metadata:
            includes.append("metadatas")
        if include_documents:
            includes.append("documents")
        
        # Generate query embedding if embedding generator is available
        query_embedding = None
        if self.embedding_generator:
            query_embedding = self.embedding_generator.generate_embedding(transformed_query)
            # Convert to list for ChromaDB if it's a numpy array
            if hasattr(query_embedding, 'tolist'):
                query_embedding = query_embedding.tolist()
        
        # Perform the query
        if query_embedding:
            results = self.indexer.collection.query(
                query_embeddings=[query_embedding],
                n_results=self.max_results,
                where=filter_criteria,
                include=includes
            )
        else:
            # If no embedding generator, let ChromaDB generate the embedding
            results = self.indexer.collection.query(
                query_texts=[transformed_query],
                n_results=self.max_results,
                where=filter_criteria,
                include=includes
            )
        
        # Format the results
        formatted_results = []
        
        # Extract results
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0] if include_metadata else [{}] * len(ids)
        documents = results.get("documents", [[]])[0] if include_documents else [""] * len(ids)
        
        # Convert distances to similarity scores (assuming cosine distance)
        similarity_scores = [1 - distance for distance in distances]
        
        # Filter by similarity threshold
        for i, (doc_id, score, metadata, document) in enumerate(zip(ids, similarity_scores, metadatas, documents)):
            if score >= self.similarity_threshold:
                formatted_results.append({
                    "chunk_id": doc_id,
                    "similarity_score": score,
                    "metadata": metadata,
                    "content": document
                })
        
        # If reranking is enabled, rerank the results
        if rerank and formatted_results:
            # This would be implemented with a more sophisticated reranking algorithm
            # For now, we just keep the existing order
            pass
        
        return formatted_results
    
    def retrieve_and_build_context(
        self, 
        query: str,
        filter_criteria: Optional[Dict[str, Any]] = None,
        max_tokens: int = 2000,
        token_estimator: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Retrieve relevant document chunks and build a coherent context for the query.
        
        Args:
            query: The query to search for
            filter_criteria: Criteria to filter results
            max_tokens: Maximum tokens in the context
            token_estimator: Function to estimate tokens in a string
            
        Returns:
            Dictionary with context, sources, and other information
        """
        # Retrieve relevant chunks
        retrieved_chunks = self.retrieve(
            query=query,
            filter_criteria=filter_criteria,
            include_metadata=True,
            include_documents=True
        )
        
        # If no chunks found, return empty context
        if not retrieved_chunks:
            return {
                "context": "",
                "sources": [],
                "query": query,
                "chunks_used": 0
            }
        
        # Simple token estimation if no estimator provided (very rough approximation)
        if token_estimator is None:
            token_estimator = lambda text: len(text.split())
        
        # Build context by combining chunks until max_tokens is reached
        context_parts = []
        sources = []
        total_tokens = 0
        chunks_used = 0
        
        for chunk in retrieved_chunks:
            content = chunk["content"]
            chunk_tokens = token_estimator(content)
            
            # If adding this chunk would exceed max_tokens, stop
            if total_tokens + chunk_tokens > max_tokens:
                break
            
            # Add chunk to context
            context_parts.append(content)
            
            # Add source information
            if chunk["metadata"]:
                source = {
                    "chunk_id": chunk["chunk_id"],
                    "similarity_score": chunk["similarity_score"]
                }
                
                # Copy relevant metadata
                for key in ["title", "source", "source_file", "doc_id"]:
                    if key in chunk["metadata"]:
                        source[key] = chunk["metadata"][key]
                
                sources.append(source)
            
            total_tokens += chunk_tokens
            chunks_used += 1
        
        # Join context parts with newlines
        context = "\n\n".join(context_parts)
        
        return {
            "context": context,
            "sources": sources,
            "query": query,
            "chunks_used": chunks_used,
            "total_tokens": total_tokens
        }
