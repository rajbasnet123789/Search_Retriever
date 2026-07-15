"""
Compatibility wrapper exposing FashionRetriever from search.py.
"""
from retriever.search import FashionRetriever
from retriever.query_parser import parse_query
from retriever.candidate_retriever import CandidateRetriever
from retriever.compositional_matcher import CompositionalMatcher
from retriever.reranker import CompositionalReranker

__all__ = [
    "FashionRetriever",
    "parse_query",
    "CandidateRetriever",
    "CompositionalMatcher",
    "CompositionalReranker",
]
