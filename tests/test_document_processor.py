import os
import sys
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag.document_processor import (
    Document, 
    DocumentChunk, 
    DocumentLoader, 
    DocumentChunker,
    MetadataExtractor
)

class TestDocumentProcessor(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = Path("test_data")
        self.test_dir.mkdir(exist_ok=True)
        
        # Create test knowledge base directory structure
        self.kb_dir = self.test_dir / "knowledge_base"
        self.kb_dir.mkdir(exist_ok=True)
        
        self.faqs_dir = self.kb_dir / "faqs"
        self.faqs_dir.mkdir(exist_ok=True)
        
        # Create a test markdown file
        self.test_md_file = self.faqs_dir / "test-faq.md"
        with open(self.test_md_file, "w") as f:
            f.write("""# Test FAQ
            
## Metadata
- Date: 2023-06-01
- Category: Test
- Tags: test, unit test
- Source: Unit Test
- Last Updated: 2023-06-01
- Content Type: FAQ
- Relevance Score: 5

## Question
What is this test for?

## Short Answer
This is a test for the document processor.

## Detailed Answer
This is a detailed answer for the document processor test.
It includes multiple paragraphs to test chunking.

This is the second paragraph.
""")
        
        # Create a test JSON file
        self.test_json_file = self.faqs_dir / "test-json.json"
        with open(self.test_json_file, "w") as f:
            f.write("""
{
    "title": "Test JSON",
    "date_created": "2023-06-01",
    "category": "Test",
    "tags": ["test", "json"],
    "content": {
        "summary": "This is a test JSON file.",
        "sections": [
            {
                "heading": "Test Section",
                "text": "This is a test section."
            }
        ]
    }
}
""")
    
    def tearDown(self):
        # Remove test files
        if self.test_md_file.exists():
            self.test_md_file.unlink()
        
        if self.test_json_file.exists():
            self.test_json_file.unlink()
        
        # Remove test directories
        if self.faqs_dir.exists():
            self.faqs_dir.rmdir()
        
        if self.kb_dir.exists():
            self.kb_dir.rmdir()
        
        if self.test_dir.exists():
            self.test_dir.rmdir()
    
    def test_document_creation(self):
        """Test that Document objects can be created."""
        doc = Document(
            content="Test content",
            metadata={"title": "Test"},
            source_file="test.md"
        )
        
        self.assertEqual(doc.content, "Test content")
        self.assertEqual(doc.metadata, {"title": "Test"})
        self.assertEqual(doc.source_file, "test.md")
        self.assertEqual(doc.chunks, [])
    
    def test_document_chunk_creation(self):
        """Test that DocumentChunk objects can be created."""
        chunk = DocumentChunk(
            content="Test chunk",
            metadata={"title": "Test"},
            chunk_id="test_1",
            source_file="test.md",
            chunk_index=1
        )
        
        self.assertEqual(chunk.content, "Test chunk")
        self.assertEqual(chunk.metadata, {"title": "Test"})
        self.assertEqual(chunk.chunk_id, "test_1")
        self.assertEqual(chunk.source_file, "test.md")
        self.assertEqual(chunk.chunk_index, 1)
    
    def test_document_loader_markdown(self):
        """Test loading markdown documents."""
        loader = DocumentLoader(base_dir=self.test_dir)
        doc = loader.load_document(self.test_md_file)
        
        self.assertIsInstance(doc, Document)
        self.assertIn("# Test FAQ", doc.content)
        self.assertEqual(doc.metadata.get("category"), "Test")
        self.assertEqual(doc.metadata.get("content_type"), "FAQ")
        self.assertEqual(doc.source_file, str(self.test_md_file))
    
    def test_document_loader_json(self):
        """Test loading JSON documents."""
        loader = DocumentLoader(base_dir=self.test_dir)
        doc = loader.load_document(self.test_json_file)
        
        self.assertIsInstance(doc, Document)
        self.assertIn("This is a test JSON file", doc.content)
        self.assertEqual(doc.metadata.get("title"), "Test JSON")
        self.assertEqual(doc.metadata.get("category"), "Test")
        self.assertEqual(doc.source_file, str(self.test_json_file))
    
    def test_document_loader_directory(self):
        """Test loading a directory of documents."""
        loader = DocumentLoader(base_dir=self.test_dir)
        docs = loader.load_directory(self.kb_dir)
        
        self.assertEqual(len(docs), 2)
        self.assertIsInstance(docs[0], Document)
        self.assertIsInstance(docs[1], Document)
    
    def test_document_chunker(self):
        """Test chunking a document."""
        # Create a document with long content
        long_content = "This is line one.\nThis is line two.\nThis is line three.\n" * 10
        doc = Document(
            content=long_content,
            metadata={"title": "Test"},
            source_file="test.md"
        )
        
        # Chunk the document
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        chunked_doc = chunker.chunk_document(doc)
        
        # Check that the document has chunks
        self.assertTrue(len(chunked_doc.chunks) > 0)
        
        # Check that each chunk has the correct metadata
        for chunk in chunked_doc.chunks:
            self.assertEqual(chunk.metadata, {"title": "Test"})
            self.assertEqual(chunk.source_file, "test.md")
    
    def test_metadata_extractor(self):
        """Test extracting metadata from a document."""
        doc = Document(
            content="Test content",
            metadata={"title": "Test"},
            source_file=str(self.test_md_file),
            doc_id="test_doc"
        )
        
        extractor = MetadataExtractor()
        metadata = extractor.extract_metadata(doc)
        
        self.assertEqual(metadata["title"], "Test")
        self.assertEqual(metadata["source_file"], str(self.test_md_file))
        self.assertEqual(metadata["content_type"], "faqs")

class TestDocument(unittest.TestCase):
    """Tests for the Document class."""
    
    def test_init_with_default_values(self):
        """Test initializing a Document with default values."""
        content = "This is a test document."
        doc = Document(content=content)
        
        self.assertEqual(doc.content, content)
        self.assertEqual(doc.metadata, {})
        self.assertIsNotNone(doc.id)
        self.assertIsNone(doc.embedding)
    
    def test_init_with_custom_values(self):
        """Test initializing a Document with custom values."""
        content = "This is a test document."
        metadata = {"title": "Test Doc", "author": "Test Author"}
        doc_id = "test_doc_id"
        embedding = [0.1, 0.2, 0.3]
        
        doc = Document(
            content=content,
            metadata=metadata,
            id=doc_id,
            embedding=embedding
        )
        
        self.assertEqual(doc.content, content)
        self.assertEqual(doc.metadata, metadata)
        self.assertEqual(doc.id, doc_id)
        self.assertEqual(doc.embedding, embedding)
    
    def test_auto_generate_id(self):
        """Test that an ID is automatically generated if not provided."""
        content = "This is a test document."
        doc = Document(content=content)
        
        self.assertIsNotNone(doc.id)
        # Same content should generate the same ID
        doc2 = Document(content=content)
        self.assertEqual(doc.id, doc2.id)
        
        # Different content should generate different IDs
        doc3 = Document(content="Different content")
        self.assertNotEqual(doc.id, doc3.id)
    
    def test_to_dict(self):
        """Test converting a Document to a dictionary."""
        content = "This is a test document."
        metadata = {"title": "Test Doc", "author": "Test Author"}
        doc_id = "test_doc_id"
        embedding = [0.1, 0.2, 0.3]
        
        doc = Document(
            content=content,
            metadata=metadata,
            id=doc_id,
            embedding=embedding
        )
        
        doc_dict = doc.to_dict()
        
        self.assertEqual(doc_dict["content"], content)
        self.assertEqual(doc_dict["metadata"], metadata)
        self.assertEqual(doc_dict["id"], doc_id)
        self.assertEqual(doc_dict["embedding"], embedding)
    
    def test_from_dict(self):
        """Test creating a Document from a dictionary."""
        doc_dict = {
            "content": "This is a test document.",
            "metadata": {"title": "Test Doc", "author": "Test Author"},
            "id": "test_doc_id",
            "embedding": [0.1, 0.2, 0.3]
        }
        
        doc = Document.from_dict(doc_dict)
        
        self.assertEqual(doc.content, doc_dict["content"])
        self.assertEqual(doc.metadata, doc_dict["metadata"])
        self.assertEqual(doc.id, doc_dict["id"])
        self.assertEqual(doc.embedding, doc_dict["embedding"])

class TestDocumentLoader(unittest.TestCase):
    """Tests for the DocumentLoader class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)
        self.loader = DocumentLoader(base_dir=str(self.base_dir))
        
        # Create test files
        self.md_content = """# Test Document
        
## Metadata
- title: Test Markdown
- category: test
- date_created: 2023-01-01

## Section 1
This is section 1.

## Section 2
This is section 2.
"""
        self.md_file = self.base_dir / "test.md"
        with open(self.md_file, 'w', encoding='utf-8') as f:
            f.write(self.md_content)
        
        self.json_content = {
            "title": "Test JSON",
            "date_created": "2023-01-01",
            "category": "test",
            "content": "This is JSON content."
        }
        self.json_file = self.base_dir / "test.json"
        with open(self.json_file, 'w', encoding='utf-8') as f:
            json.dump(self.json_content, f)
        
        self.yaml_content = """
title: Test YAML
date_created: 2023-01-01
category: test
sections:
  - heading: Section 1
    content: This is section 1.
  - heading: Section 2
    content: This is section 2.
"""
        self.yaml_file = self.base_dir / "test.yaml"
        with open(self.yaml_file, 'w', encoding='utf-8') as f:
            f.write(self.yaml_content)
        
        self.txt_content = "This is a text file content."
        self.txt_file = self.base_dir / "test.txt"
        with open(self.txt_file, 'w', encoding='utf-8') as f:
            f.write(self.txt_content)
    
    def tearDown(self):
        """Clean up after the test."""
        self.temp_dir.cleanup()
    
    def test_load_document_markdown(self):
        """Test loading a Markdown document."""
        doc = self.loader.load_document(self.md_file)
        
        self.assertEqual(doc.content, self.md_content)
        self.assertIn("title", doc.metadata)
        self.assertEqual(doc.metadata["title"], "Test Markdown")
        self.assertEqual(doc.source_file, str(self.md_file))
    
    def test_load_document_json(self):
        """Test loading a JSON document."""
        doc = self.loader.load_document(self.json_file)
        
        self.assertIn("This is JSON content.", doc.content)
        self.assertIn("title", doc.metadata)
        self.assertEqual(doc.metadata["title"], "Test JSON")
        self.assertEqual(doc.source_file, str(self.json_file))
    
    def test_load_document_txt(self):
        """Test loading a text document."""
        doc = self.loader.load_document(self.txt_file)
        
        self.assertEqual(doc.content, self.txt_content)
        self.assertIn("title", doc.metadata)
        self.assertEqual(doc.source_file, str(self.txt_file))
    
    def test_load_nonexistent_file(self):
        """Test loading a nonexistent file."""
        with self.assertRaises(FileNotFoundError):
            self.loader.load_document(self.base_dir / "nonexistent.md")
    
    def test_load_unsupported_format(self):
        """Test loading an unsupported file format."""
        unsupported_file = self.base_dir / "test.xyz"
        with open(unsupported_file, 'w', encoding='utf-8') as f:
            f.write("Test content")
        
        with self.assertRaises(ValueError):
            self.loader.load_document(unsupported_file)
    
    def test_load_directory(self):
        """Test loading documents from a directory."""
        docs = self.loader.load_directory(self.base_dir)
        
        # Should load 4 documents (md, json, yaml, txt)
        self.assertEqual(len(docs), 4)
        
        # Create a subdirectory with a document
        subdir = self.base_dir / "subdir"
        os.makedirs(subdir, exist_ok=True)
        subdir_file = subdir / "subtest.md"
        with open(subdir_file, 'w', encoding='utf-8') as f:
            f.write("# Subdir Test")
        
        # Test with recursive=True
        docs = self.loader.load_directory(self.base_dir, recursive=True)
        self.assertEqual(len(docs), 5)
        
        # Test with recursive=False
        docs = self.loader.load_directory(self.base_dir, recursive=False)
        self.assertEqual(len(docs), 4)

class TestDocumentChunker(unittest.TestCase):
    """Tests for the DocumentChunker class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    
    def test_chunk_document(self):
        """Test chunking a document."""
        # Create a document with content larger than chunk size
        content = "\n".join([f"Line {i}" for i in range(20)])
        doc = Document(
            content=content,
            metadata={"title": "Test Doc"},
            id="test_doc",
            source_file="test.txt",
            doc_id="test_doc"
        )
        
        chunked_doc = self.chunker.chunk_document(doc)
        
        self.assertIsNotNone(chunked_doc.chunks)
        self.assertGreater(len(chunked_doc.chunks), 0)
        
        # Test that each chunk has the correct metadata
        for i, chunk in enumerate(chunked_doc.chunks):
            self.assertEqual(chunk.metadata["title"], "Test Doc")
            self.assertEqual(chunk.source_file, "test.txt")
            self.assertEqual(chunk.chunk_index, i)
            self.assertEqual(chunk.chunk_id, f"test_doc_chunk_{i}")
    
    def test_empty_document(self):
        """Test chunking an empty document."""
        doc = Document(
            content="",
            metadata={"title": "Empty Doc"},
            id="empty_doc",
            source_file="empty.txt",
            doc_id="empty_doc"
        )
        
        chunked_doc = self.chunker.chunk_document(doc)
        self.assertEqual(len(chunked_doc.chunks), 0)
    
    def test_split_text(self):
        """Test the _split_text method directly."""
        text = "\n".join([f"Line {i}" for i in range(20)])
        chunks = self.chunker._split_text(text, "test.txt")
        
        self.assertGreater(len(chunks), 0)
        
        # Test that chunks have a reasonable size
        for chunk in chunks:
            self.assertLessEqual(len(chunk), self.chunker.chunk_size)

class TestMetadataExtractor(unittest.TestCase):
    """Tests for the MetadataExtractor class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.extractor = MetadataExtractor()
    
    def test_extract_metadata(self):
        """Test extracting metadata from a document."""
        doc = Document(
            content="Test content",
            metadata={"author": "Test Author"},
            source_file="/path/to/test/file.md",
            doc_id="test_doc"
        )
        
        metadata = self.extractor.extract_metadata(doc)
        
        self.assertIn("author", metadata)
        self.assertEqual(metadata["author"], "Test Author")
        self.assertIn("source_file", metadata)
        self.assertEqual(metadata["source_file"], "/path/to/test/file.md")
        self.assertIn("content_type", metadata)
        self.assertIn("title", metadata)
    
    def test_existing_metadata_preserved(self):
        """Test that existing metadata is preserved."""
        doc = Document(
            content="Test content",
            metadata={
                "author": "Test Author",
                "title": "Custom Title",
                "content_type": "custom_type",
                "source_file": "custom_path"
            },
            source_file="/path/to/test/file.md",
            doc_id="test_doc"
        )
        
        metadata = self.extractor.extract_metadata(doc)
        
        self.assertEqual(metadata["title"], "Custom Title")
        self.assertEqual(metadata["content_type"], "custom_type")
        self.assertEqual(metadata["source_file"], "custom_path")

if __name__ == "__main__":
    unittest.main() 