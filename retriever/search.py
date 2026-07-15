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

        self.candidate_retriever = CandidateRetriever(self.store, self.clip)
        self.matcher = CompositionalMatcher(self.metadata)
        self.reranker = CompositionalReranker(self.matcher, self.metadata)

    def index_image(self, image_path):
        """Extract and index features for one image."""
        meta = self.indexer.index_image(image_path)
        self.metadata = self.indexer.metadata
        self.matcher.metadata = self.metadata
        self.reranker.metadata = self.metadata
        return meta

    def index_batch(self, image_paths, callback=None):
        """Index multiple images."""
        self.indexer.index_batch(image_paths, callback=callback)
        self.metadata = self.indexer.metadata
        self.matcher.metadata = self.metadata
        self.reranker.metadata = self.metadata

    def save(self, directory=None):
        """Save indices and metadata to disk."""
        self.indexer.save(directory)

    def load(self, directory=None):
        """Load indices and metadata from disk."""
        self.indexer.load(directory)
        self.metadata = self.indexer.metadata
        self.matcher.metadata = self.metadata
        self.reranker.metadata = self.metadata

    def search(self, query, k=10):
        """
        Search for top-k images matching a natural language query.

        Steps:
          1. Parse natural language query into semantic components (query_parser).
          2. Retrieve global and regional candidate images via FAISS (candidate_retriever).
          3. Evaluate compositional color/garment & scene attributes (compositional_matcher).
          4. Fuse scores with weighting: 0.30 Global + 0.40 Regional + 0.20 Comp + 0.10 Scene (reranker).
        """
        parsed = parse_query(query)
        k_search = max(50, k * 5)
        candidates = self.candidate_retriever.get_candidates(parsed, query, k=k_search)
        results = self.reranker.rerank(candidates, parsed, top_k=k)
        return results
