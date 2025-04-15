import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Union, Optional, Tuple
from pathlib import Path

# Try to import vector store libraries, but allow graceful failure
try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

# For type hints
from .document_processor import Document, DocumentChunk
from .embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)

class VectorStoreIndexer:
    """Manages the vector store and indexing of document embeddings."""
    
    def __init__(
        self,
        persist_directory: str = "data/vector_db",
        collection_name: str = "instagram_chatbot",
        embedding_function_name: str = "sentence_transformer",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        distance_metric: str = "cosine"
    ):
        """
        Initialize the vector store indexer.
        
        Args:
            persist_directory: Directory to persist the vector store
            collection_name: Name of the collection to store embeddings
            embedding_function_name: Name of the embedding function to use
            embedding_model: Name of the embedding model to use
            distance_metric: Distance metric to use for similarity search
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_function_name = embedding_function_name
        self.embedding_model = embedding_model
        self.distance_metric = distance_metric
        self.client = None
        self.collection = None
        
        # Check if ChromaDB is available
        if not CHROMADB_AVAILABLE:
            logger.warning(
                "ChromaDB is not available. "
                "Please install it with `pip install chromadb`."
            )
    
    def _initialize_client(self):
        """Initialize the ChromaDB client."""
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "ChromaDB is not available. "
                "Please install it with `pip install chromadb`."
            )
        
        # Create directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
    
    def _get_embedding_function(self):
        """Get the embedding function for ChromaDB."""
        if self.embedding_function_name == "sentence_transformer":
            return embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model
            )
        elif self.embedding_function_name == "openai":
            # Would need to set OPENAI_API_KEY as env var
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ.get("OPENAI_API_KEY"),
                model_name="text-embedding-ada-002"
            )
        else:
            # Default to None, which means you'll need to provide embeddings
            return None
    
    def _get_collection(self):
        """Get or create the ChromaDB collection."""
        if self.client is None:
            self._initialize_client()
        
        embedding_function = self._get_embedding_function()
        
        # Get existing collection or create new one
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=embedding_function
            )
            logger.info(f"Retrieved existing collection: {self.collection_name}")
        except ValueError:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=embedding_function,
                metadata={"hnsw:space": self.distance_metric}
            )
            logger.info(f"Created new collection: {self.collection_name}")
    
    def index_document(
        self, 
        document: Document,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        batch_size: int = 100
    ):
        """
        Index a document in the vector store.
        
        Args:
            document: Document to index
            embedding_generator: Optional embedding generator to use
            batch_size: Batch size for indexing chunks
        """
        # Ensure the collection is initialized
        if self.collection is None:
            self._get_collection()
        
        # If document has no chunks, chunk it
        if not document.chunks:
            from .document_processor import DocumentChunker
            chunker = DocumentChunker()
            document = chunker.chunk_document(document)
        
        # If embedding generator is provided, use it to generate embeddings
        if embedding_generator:
            embedded_document = embedding_generator.embed_document(document)
            
            # Index chunks in batches
            chunks = embedded_document["chunks"]
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                
                # Extract data for batch
                ids = [chunk["chunk_id"] for chunk in batch]
                documents = [chunk["content"] for chunk in batch]
                embeddings = [chunk["embedding"].tolist() for chunk in batch]
                metadatas = [
                    {
                        **chunk["metadata"],
                        "chunk_index": chunk["chunk_index"],
                        "source_file": chunk["source_file"],
                        "doc_id": document.doc_id
                    }
                    for chunk in batch
                ]
                
                # Add to collection
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
        else:
            # If no embedding generator, let ChromaDB generate embeddings
            # This requires setting up an embedding function in the collection
            
            # Index chunks in batches
            chunks = document.chunks
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                
                # Extract data for batch
                ids = [chunk.chunk_id for chunk in batch]
                documents = [chunk.content for chunk in batch]
                metadatas = [
                    {
                        **chunk.metadata,
                        "chunk_index": chunk.chunk_index,
                        "source_file": chunk.source_file,
                        "doc_id": document.doc_id
                    }
                    for chunk in batch
                ]
                
                # Add to collection
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
    
    def index_documents(
        self, 
        documents: List[Document],
        embedding_generator: Optional[EmbeddingGenerator] = None,
        batch_size: int = 100
    ):
        """
        Index multiple documents in the vector store.
        
        Args:
            documents: List of documents to index
            embedding_generator: Optional embedding generator to use
            batch_size: Batch size for indexing chunks
        """
        for document in documents:
            self.index_document(
                document=document,
                embedding_generator=embedding_generator,
                batch_size=batch_size
            )
    
    def delete_document(self, doc_id: str):
        """
        Delete a document from the vector store.
        
        Args:
            doc_id: ID of the document to delete
        """
        # Ensure the collection is initialized
        if self.collection is None:
            self._get_collection()
        
        # Delete all chunks for the document
        self.collection.delete(
            where={"doc_id": doc_id}
        )
    
    def delete_documents(self, doc_ids: List[str]):
        """
        Delete multiple documents from the vector store.
        
        Args:
            doc_ids: List of document IDs to delete
        """
        for doc_id in doc_ids:
            self.delete_document(doc_id)
    
    def delete_collection(self):
        """Delete the entire collection."""
        if self.client is None:
            self._initialize_client()
        
        self.client.delete_collection(self.collection_name)
        self.collection = None
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary containing collection statistics
        """
        if self.collection is None:
            self._get_collection()
        
        # Get count of items in collection
        count = self.collection.count()
        
        # Get unique document IDs
        results = self.collection.get(
            include=["metadatas"],
            limit=count
        )
        
        metadatas = results["metadatas"]
        doc_ids = set()
        content_types = {}
        categories = {}
        
        for metadata in metadatas:
            if "doc_id" in metadata:
                doc_ids.add(metadata["doc_id"])
            
            if "content_type" in metadata:
                content_type = metadata["content_type"]
                content_types[content_type] = content_types.get(content_type, 0) + 1
            
            if "category" in metadata:
                category = metadata["category"]
                categories[category] = categories.get(category, 0) + 1
        
        return {
            "total_chunks": count,
            "total_documents": len(doc_ids),
            "content_types": content_types,
            "categories": categories
        }
