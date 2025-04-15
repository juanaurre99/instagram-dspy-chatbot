import os
import yaml
import json
import logging
import markdown
from typing import Dict, List, Union, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field, asdict
from pathlib import Path
import re
import hashlib
import tiktoken

# Use TYPE_CHECKING to prevent circular imports
if TYPE_CHECKING:
    from src.rag.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of a document with its metadata."""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str = ""
    source_file: str = ""
    chunk_index: int = 0

@dataclass
class Document:
    """Class representing a document or chunk of text with metadata."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_file: str = ""
    doc_id: str = ""
    id: Optional[str] = None
    embedding: Optional[List[float]] = None
    chunks: List["DocumentChunk"] = field(default_factory=list)
    
    def __post_init__(self):
        """Generate an ID for the document if not provided."""
        if self.id is None:
            # Create a hash of the content as the ID
            self.id = hashlib.md5(self.content.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the document to a dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create a Document from a dictionary."""
        return cls(**data)

class DocumentLoader:
    """Loads documents from various file formats."""
    
    def __init__(self, base_dir: str = "data/knowledge_base"):
        """
        Initialize the document loader.
        
        Args:
            base_dir: Base directory for the knowledge base
        """
        self.base_dir = Path(base_dir)
    
    def load_document(self, file_path: Union[str, Path]) -> Document:
        """
        Load a document from a file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document object
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} not found")
        
        # Generate a document ID based on the file path
        doc_id = str(file_path.relative_to(self.base_dir))
        
        if file_path.suffix.lower() == '.md':
            return self._load_markdown(file_path, doc_id)
        elif file_path.suffix.lower() == '.json':
            return self._load_json(file_path, doc_id)
        elif file_path.suffix.lower() == '.yaml' or file_path.suffix.lower() == '.yml':
            return self._load_yaml(file_path, doc_id)
        elif file_path.suffix.lower() == '.txt':
            return self._load_text(file_path, doc_id)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _load_markdown(self, file_path: Path, doc_id: str) -> Document:
        """Parse a markdown file into a Document."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract metadata from markdown
        metadata = {}
        lines = content.split('\n')
        
        # Look for metadata section
        metadata_section = False
        metadata_lines = []
        
        for i, line in enumerate(lines):
            if line.strip() == '## Metadata':
                metadata_section = True
                continue
            
            if metadata_section and line.strip().startswith('##'):
                metadata_section = False
                break
                
            if metadata_section and line.strip().startswith('-'):
                metadata_lines.append(line)
        
        # Parse metadata lines
        for line in metadata_lines:
            parts = line.strip('- ').split(':', 1)
            if len(parts) == 2:
                key, value = parts
                metadata[key.strip().lower().replace(' ', '_')] = value.strip()
        
        # If title not in metadata, try to extract from first heading
        if 'title' not in metadata:
            for line in lines:
                if line.startswith('# '):
                    metadata['title'] = line[2:].strip()
                    break
        
        return Document(
            content=content,
            metadata=metadata,
            source_file=str(file_path),
            doc_id=doc_id
        )
    
    def _load_json(self, file_path: Path, doc_id: str) -> Document:
        """Parse a JSON file into a Document."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract metadata fields from the JSON
        metadata = {}
        content = ""
        
        # Common metadata fields
        metadata_fields = [
            'title', 'date_created', 'last_updated', 
            'category', 'tags', 'source', 'content_type',
            'relevance_score'
        ]
        
        for field in metadata_fields:
            if field in data:
                metadata[field] = data[field]
        
        # Handle content field (might be nested)
        if 'content' in data:
            if isinstance(data['content'], str):
                content = data['content']
            elif isinstance(data['content'], dict):
                content_parts = []
                if 'summary' in data['content']:
                    content_parts.append(f"Summary: {data['content']['summary']}")
                
                if 'sections' in data['content']:
                    for section in data['content']['sections']:
                        if 'heading' in section:
                            content_parts.append(f"## {section['heading']}")
                        if 'text' in section:
                            content_parts.append(section['text'])
                
                content = "\n\n".join(content_parts)
            else:
                # Try to convert the content to a string representation
                content = str(data['content'])
        else:
            # If no content field, use the whole JSON as content
            content = json.dumps(data, indent=2)
        
        return Document(
            content=content,
            metadata=metadata,
            source_file=str(file_path),
            doc_id=doc_id
        )
    
    def _load_yaml(self, file_path: Path, doc_id: str) -> Document:
        """Parse a YAML file into a Document."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Extract metadata fields from the YAML
        metadata = {}
        content_parts = []
        
        # Common metadata fields at the top level
        metadata_fields = [
            'title', 'date_created', 'last_updated', 
            'category', 'tags', 'source', 'content_type',
            'relevance_score'
        ]
        
        for field in metadata_fields:
            if field in data:
                metadata[field] = data[field]
        
        # Convert the YAML content to formatted text
        for key, value in data.items():
            if key not in metadata_fields:
                if isinstance(value, dict):
                    content_parts.append(f"## {key.capitalize()}")
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, list):
                            content_parts.append(f"### {sub_key.capitalize()}")
                            for item in sub_value:
                                if isinstance(item, dict):
                                    for item_key, item_value in item.items():
                                        content_parts.append(f"- {item_key}: {item_value}")
                                else:
                                    content_parts.append(f"- {item}")
                        else:
                            content_parts.append(f"### {sub_key.capitalize()}")
                            content_parts.append(f"{sub_value}")
                elif isinstance(value, list):
                    content_parts.append(f"## {key.capitalize()}")
                    for item in value:
                        if isinstance(item, dict):
                            for item_key, item_value in item.items():
                                content_parts.append(f"- {item_key}: {item_value}")
                        else:
                            content_parts.append(f"- {item}")
                else:
                    content_parts.append(f"## {key.capitalize()}")
                    content_parts.append(f"{value}")
        
        content = "\n\n".join(content_parts)
        
        return Document(
            content=content,
            metadata=metadata,
            source_file=str(file_path),
            doc_id=doc_id
        )
    
    def _load_text(self, file_path: Path, doc_id: str) -> Document:
        """Parse a text file into a Document."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract basic metadata from filename
        filename = file_path.stem
        category = file_path.parent.name
        
        metadata = {
            'title': filename.replace('-', ' ').title(),
            'category': category,
            'content_type': 'text'
        }
        
        return Document(
            content=content,
            metadata=metadata,
            source_file=str(file_path),
            doc_id=doc_id
        )
    
    def load_directory(self, directory: Union[str, Path], recursive: bool = True) -> List[Document]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory: Directory to load documents from
            recursive: Whether to recursively load documents from subdirectories
            
        Returns:
            List of Document objects
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory {directory} not found")
        
        documents = []
        
        # Get all files in the directory
        if recursive:
            files = list(directory.glob('**/*.*'))
        else:
            files = list(directory.glob('*.*'))
        
        # Filter out unsupported file types and template files
        supported_extensions = ['.md', '.json', '.yaml', '.yml', '.txt']
        files = [f for f in files if f.suffix.lower() in supported_extensions and f.name != 'template.md' and f.name != 'template.json' and f.name != 'template.yaml' and f.name != 'template.yml' and f.name != 'README.md']
        
        # Load each document
        for file_path in files:
            try:
                document = self.load_document(file_path)
                documents.append(document)
            except Exception as e:
                logger.error(f"Error loading document {file_path}: {e}")
        
        return documents

class DocumentChunker:
    """Chunks documents into smaller pieces for processing."""
    
    def __init__(
        self, 
        chunk_size: int = 512, 
        chunk_overlap: int = 128,
        separator: str = "\n"
    ):
        """
        Initialize the document chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            separator: String to use as separator when splitting text
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
    
    def chunk_document(self, document: Document) -> Document:
        """
        Split a document into chunks.
        
        Args:
            document: Document to chunk
            
        Returns:
            Document with chunks added
        """
        if document.content:
            chunks = self._split_text(document.content, document.source_file)
            document.chunks = [
                DocumentChunk(
                    content=chunk,
                    metadata=document.metadata.copy(),
                    chunk_id=f"{document.doc_id}_chunk_{i}",
                    source_file=document.source_file,
                    chunk_index=i
                )
                for i, chunk in enumerate(chunks)
            ]
        return document
    
    def _split_text(self, text: str, source_file: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Text to split
            source_file: Source file path for logging
            
        Returns:
            List of text chunks
        """
        chunks = []
        
        # Split by separator
        sections = text.split(self.separator)
        
        current_chunk = []
        current_size = 0
        
        for section in sections:
            section_size = len(section)
            
            # If a single section is larger than the chunk size, we need to split it
            if section_size > self.chunk_size:
                # First add the current chunk if it's not empty
                if current_chunk:
                    chunks.append(self.separator.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                # Then split the large section into smaller chunks
                words = section.split(' ')
                sub_chunk = []
                sub_size = 0
                
                for word in words:
                    word_size = len(word) + 1  # +1 for the space
                    if sub_size + word_size <= self.chunk_size:
                        sub_chunk.append(word)
                        sub_size += word_size
                    else:
                        chunks.append(' '.join(sub_chunk))
                        sub_chunk = [word]
                        sub_size = word_size
                
                if sub_chunk:
                    chunks.append(' '.join(sub_chunk))
            
            # If adding this section would make the chunk too large, start a new chunk
            elif current_size + section_size + len(self.separator) > self.chunk_size:
                chunks.append(self.separator.join(current_chunk))
                
                # Start new chunk with overlap from the previous chunk
                if self.chunk_overlap > 0 and current_chunk:
                    # Calculate overlap characters
                    overlap_text = self.separator.join(current_chunk)
                    overlap_start = max(0, len(overlap_text) - self.chunk_overlap)
                    overlap_text = overlap_text[overlap_start:]
                    
                    # Find a clean break at a separator if possible
                    overlap_sections = overlap_text.split(self.separator)
                    if len(overlap_sections) > 1:
                        current_chunk = overlap_sections[1:]
                        current_size = sum(len(s) for s in current_chunk) + len(self.separator) * (len(current_chunk) - 1)
                    else:
                        current_chunk = []
                        current_size = 0
                else:
                    current_chunk = []
                    current_size = 0
                
                current_chunk.append(section)
                current_size += section_size
            else:
                current_chunk.append(section)
                current_size += section_size + len(self.separator)
        
        # Add the last chunk if there is one
        if current_chunk:
            chunks.append(self.separator.join(current_chunk))
        
        return chunks

class MetadataExtractor:
    """Extracts and enhances metadata from documents."""
    
    def extract_metadata(self, document: Document) -> Dict[str, Any]:
        """
        Extract and enhance metadata from a document.
        
        Args:
            document: Document to extract metadata from
            
        Returns:
            Enhanced metadata dictionary
        """
        metadata = document.metadata.copy()
        
        # Add source file information if not already present
        if 'source_file' not in metadata:
            metadata['source_file'] = document.source_file
        
        # Extract document type from path if not present
        if 'content_type' not in metadata:
            source_path = Path(document.source_file)
            parent_dir = source_path.parent.name
            metadata['content_type'] = parent_dir
        
        # Generate a title if not present
        if 'title' not in metadata:
            source_path = Path(document.source_file)
            metadata['title'] = source_path.stem.replace('-', ' ').title()
        
        return metadata
