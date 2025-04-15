import os
import numpy as np
import logging
from typing import List, Dict, Any, Union, Optional, TYPE_CHECKING
from pathlib import Path

# Try to import sentence-transformers, but allow graceful failure
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# For type hints with forward references
if TYPE_CHECKING:
    from .document_processor import DocumentChunk, Document

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generates embeddings for document chunks using various embedding models."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize_embeddings: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the embedding model to use
            device: Device to use for generating embeddings ("cpu" or "cuda")
            normalize_embeddings: Whether to normalize embeddings
            cache_dir: Directory to cache models
        """
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.cache_dir = cache_dir
        self.model = None
        
        # Check if sentence-transformers is available
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning(
                "sentence-transformers is not available. "
                "Please install it with `pip install sentence-transformers`."
            )
    
    def load_model(self):
        """Load the embedding model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is not available. "
                "Please install it with `pip install sentence-transformers`."
            )
        
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            
            # Create model with specified cache directory if provided
            kwargs = {"device": self.device}
            if self.cache_dir:
                kwargs["cache_folder"] = self.cache_dir
                
            # Load the model
            self.model = SentenceTransformer(self.model_name, **kwargs)
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate an embedding for a single text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Embedding as numpy array
        """
        self.load_model()
        
        # Generate embedding
        embedding = self.model.encode(
            text, 
            normalize_embeddings=self.normalize_embeddings
        )
        
        return embedding
    
    def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embeddings as numpy arrays
        """
        self.load_model()
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts, 
            normalize_embeddings=self.normalize_embeddings,
            batch_size=32,  # Adjust batch size as needed
            show_progress_bar=len(texts) > 100  # Show progress bar for large batches
        )
        
        return embeddings
    
    def embed_document_chunks(self, chunks: List["DocumentChunk"]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for document chunks and return them with metadata.
        
        Args:
            chunks: List of document chunks to generate embeddings for
            
        Returns:
            List of dictionaries containing the embedding and metadata for each chunk
        """
        if not chunks:
            return []
        
        # Extract text from chunks
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Combine embeddings with metadata
        embedded_chunks = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            embedded_chunks.append({
                "content": chunk.content,
                "embedding": embedding,
                "metadata": chunk.metadata,
                "chunk_id": chunk.chunk_id,
                "source_file": chunk.source_file,
                "chunk_index": chunk.chunk_index
            })
        
        return embedded_chunks
    
    def embed_document(self, document: "Document") -> Dict[str, Any]:
        """
        Generate embeddings for a document's chunks and return them with metadata.
        
        Args:
            document: Document to generate embeddings for
            
        Returns:
            Dictionary containing the document metadata and embedded chunks
        """
        # If document has no chunks, chunk it (but this should normally be done in advance)
        if not document.chunks:
            from .document_processor import DocumentChunker
            chunker = DocumentChunker()
            document = chunker.chunk_document(document)
        
        # Generate embeddings for chunks
        embedded_chunks = self.embed_document_chunks(document.chunks)
        
        return {
            "doc_id": document.doc_id,
            "metadata": document.metadata,
            "source_file": document.source_file,
            "chunks": embedded_chunks
        }
    
    def embed_documents(self, documents: List["Document"]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for multiple documents.
        
        Args:
            documents: List of documents to generate embeddings for
            
        Returns:
            List of dictionaries containing document metadata and embedded chunks
        """
        embedded_documents = []
        
        for document in documents:
            embedded_document = self.embed_document(document)
            embedded_documents.append(embedded_document)
        
        return embedded_documents
