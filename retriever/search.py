import os
from indexer.fashion_indexer import FashionIndexer
from retriever.query_parser import parse_query
from retriever.candidate_retriever import CandidateRetriever
from retriever.compositional_matcher import CompositionalMatcher
from retriever.reranker import CompositionalReranker


class FashionRetriever:
    """
    Unified search and retrieval interface combining Part A (Indexing) and Part B (Retrieval).
    """

    def __init__(self, index_dir=None, device=None, conf=0.25):
        self.indexer = FashionIndexer(index_dir=index_dir, device=device, conf=conf)
        self.store = self.indexer.store
        self.clip = self.indexer.clip
        self.metadata = self.indexer.metadata

        self.candidate_retriever = CandidateRetriever(self.store, self.clip, self.metadata)
        self.matcher = CompositionalMatcher(self.metadata)
        self.reranker = CompositionalReranker(self.matcher, self.metadata)

    def index_image(self, image_path):
        """Extract and index features for one image."""
        meta = self.indexer.index_image(image_path)
        self.metadata = self.indexer.metadata
        self.candidate_retriever.metadata = self.metadata
        self.matcher.metadata = self.metadata
        self.reranker.metadata = self.metadata
        return meta

    def index_batch(self, image_paths, callback=None):
        """Index multiple images."""
        self.indexer.index_batch(image_paths, callback=callback)
        self.metadata = self.indexer.metadata
        self.candidate_retriever.metadata = self.metadata
        self.matcher.metadata = self.metadata
        self.reranker.metadata = self.metadata

    def save(self, directory=None):
        """Save indices and metadata to disk."""
        self.indexer.save(directory)

    def load(self, directory=None):
        """Load indices and metadata from disk."""
        self.indexer.load(directory)
        self.metadata = self.indexer.metadata
        self.candidate_retriever.metadata = self.metadata
        self.matcher.metadata = self.metadata
        self.reranker.metadata = self.metadata

    def search(self, query, k=10, parsed_query=None):
        """
        Search for top-k images matching a natural language query.

        Args:
            query: natural language query string
            k: number of results to return
            parsed_query: optional pre-parsed query dict (avoids re-parsing)

        Steps:
          1. Parse query (or reuse provided parsed_query).
          2. Retrieve global and regional candidates via FAISS with large pool.
          3. Rerank all candidates — no hard filtering, completeness penalty only.
          4. If fewer than k results, expand search and rerank with larger pool.
        """
        if parsed_query is None:
            parsed_query = parse_query(query)

        k_search = max(100, k * 15)
        candidates = self.candidate_retriever.get_candidates(parsed_query, query, k=k_search)
        results = self.reranker.rerank(candidates, parsed_query, top_k=k)

        return results
