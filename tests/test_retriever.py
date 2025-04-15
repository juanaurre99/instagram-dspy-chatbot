import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag.retriever import QueryTransformer, Retriever
from src.rag.indexer import VectorStoreIndexer
from src.rag.embeddings import EmbeddingGenerator

class TestQueryTransformer(unittest.TestCase):
    def test_transform_query(self):
        """Test that the query transformer returns the original query."""
        transformer = QueryTransformer()
        query = "test query"
        transformed_query = transformer.transform_query(query)
        self.assertEqual(transformed_query, query)

class TestRetriever(unittest.TestCase):
    def setUp(self):
        # Mock the indexer and its collection
        self.mock_indexer = MagicMock(spec=VectorStoreIndexer)
        self.mock_collection = MagicMock()
        self.mock_indexer.collection = self.mock_collection
        
        # Mock the embedding generator
        self.mock_embedding_generator = MagicMock(spec=EmbeddingGenerator)
        
        # Create a retriever with the mocks
        self.retriever = Retriever(
            indexer=self.mock_indexer,
            embedding_generator=self.mock_embedding_generator,
            max_results=3,
            similarity_threshold=0.7
        )
    
    def test_retrieve_with_embedding_generator(self):
        """Test retrieving with an embedding generator."""
        # Set up the mock embedding generator
        mock_embedding = [0.1, 0.2, 0.3]
        self.mock_embedding_generator.generate_embedding.return_value = mock_embedding
        
        # Set up the mock collection
        self.mock_collection.query.return_value = {
            "ids": [["chunk1", "chunk2"]],
            "distances": [[0.1, 0.2]],
            "metadatas": [[{"title": "Test 1"}, {"title": "Test 2"}]],
            "documents": [["Content 1", "Content 2"]]
        }
        
        # Call retrieve
        results = self.retriever.retrieve(
            query="test query",
            filter_criteria={"category": "test"}
        )
        
        # Check that the embedding generator was called
        self.mock_embedding_generator.generate_embedding.assert_called_once_with("test query")
        
        # Check that the collection was queried
        self.mock_collection.query.assert_called_once_with(
            query_embeddings=[mock_embedding],
            n_results=3,
            where={"category": "test"},
            include=["metadatas", "documents"]
        )
        
        # Check the results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["chunk_id"], "chunk1")
        self.assertEqual(results[0]["similarity_score"], 0.9)  # 1 - 0.1
        self.assertEqual(results[0]["metadata"], {"title": "Test 1"})
        self.assertEqual(results[0]["content"], "Content 1")
    
    def test_retrieve_without_embedding_generator(self):
        """Test retrieving without an embedding generator."""
        # Create a retriever without an embedding generator
        retriever = Retriever(
            indexer=self.mock_indexer,
            embedding_generator=None,
            max_results=3,
            similarity_threshold=0.7
        )
        
        # Set up the mock collection
        self.mock_collection.query.return_value = {
            "ids": [["chunk1", "chunk2"]],
            "distances": [[0.1, 0.2]],
            "metadatas": [[{"title": "Test 1"}, {"title": "Test 2"}]],
            "documents": [["Content 1", "Content 2"]]
        }
        
        # Call retrieve
        results = retriever.retrieve(
            query="test query",
            filter_criteria={"category": "test"}
        )
        
        # Check that the collection was queried
        self.mock_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=3,
            where={"category": "test"},
            include=["metadatas", "documents"]
        )
        
        # Check the results
        self.assertEqual(len(results), 2)
    
    def test_filter_by_similarity_threshold(self):
        """Test filtering results by similarity threshold."""
        # Set up the mock embedding generator
        mock_embedding = [0.1, 0.2, 0.3]
        self.mock_embedding_generator.generate_embedding.return_value = mock_embedding
        
        # Set up the mock collection with a result below threshold
        self.mock_collection.query.return_value = {
            "ids": [["chunk1", "chunk2", "chunk3"]],
            "distances": [[0.1, 0.2, 0.4]],  # 0.4 distance = 0.6 similarity, below threshold
            "metadatas": [[{"title": "Test 1"}, {"title": "Test 2"}, {"title": "Test 3"}]],
            "documents": [["Content 1", "Content 2", "Content 3"]]
        }
        
        # Call retrieve
        results = self.retriever.retrieve(
            query="test query"
        )
        
        # Check that only results above threshold are returned
        self.assertEqual(len(results), 2)  # Only chunk1 and chunk2 should be returned
        chunk_ids = [result["chunk_id"] for result in results]
        self.assertIn("chunk1", chunk_ids)
        self.assertIn("chunk2", chunk_ids)
        self.assertNotIn("chunk3", chunk_ids)
    
    def test_retrieve_and_build_context(self):
        """Test retrieving and building context."""
        # Set up the mock embedding generator
        mock_embedding = [0.1, 0.2, 0.3]
        self.mock_embedding_generator.generate_embedding.return_value = mock_embedding
        
        # Set up the mock collection
        self.mock_collection.query.return_value = {
            "ids": [["chunk1", "chunk2"]],
            "distances": [[0.1, 0.2]],
            "metadatas": [[
                {"title": "Test 1", "doc_id": "doc1", "source": "Source 1"},
                {"title": "Test 2", "doc_id": "doc2", "source": "Source 2"}
            ]],
            "documents": [["Content 1", "Content 2"]]
        }
        
        # Call retrieve_and_build_context
        context = self.retriever.retrieve_and_build_context(
            query="test query",
            filter_criteria={"category": "test"},
            max_tokens=1000
        )
        
        # Check the context
        self.assertEqual(context["context"], "Content 1\n\nContent 2")
        self.assertEqual(len(context["sources"]), 2)
        self.assertEqual(context["sources"][0]["doc_id"], "doc1")
        self.assertEqual(context["sources"][0]["title"], "Test 1")
        self.assertEqual(context["sources"][1]["doc_id"], "doc2")
        self.assertEqual(context["chunks_used"], 2)
    
    def test_empty_results(self):
        """Test handling empty results."""
        # Set up the mock embedding generator
        mock_embedding = [0.1, 0.2, 0.3]
        self.mock_embedding_generator.generate_embedding.return_value = mock_embedding
        
        # Set up the mock collection with empty results
        self.mock_collection.query.return_value = {
            "ids": [[]],
            "distances": [[]],
            "metadatas": [[]],
            "documents": [[]]
        }
        
        # Call retrieve
        results = self.retriever.retrieve(
            query="test query"
        )
        
        # Check that empty results are handled
        self.assertEqual(len(results), 0)
        
        # Call retrieve_and_build_context
        context = self.retriever.retrieve_and_build_context(
            query="test query"
        )
        
        # Check that empty context is returned
        self.assertEqual(context["context"], "")
        self.assertEqual(context["sources"], [])
        self.assertEqual(context["chunks_used"], 0)

if __name__ == "__main__":
    unittest.main() 