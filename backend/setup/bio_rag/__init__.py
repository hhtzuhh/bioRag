"""
ADVANCED BIO RAG SYSTEM WITH RECURSIVE RETRIEVAL FOR MEGACONTEXT
==============================================================

This package implements a canonical LlamaIndex pattern for handling "megacontext" scenarios 
where potentially relevant information far exceeds any LLM's context window. The solution uses 
RECURSIVE RETRIEVAL to intelligently navigate hierarchical data structures.

🔄 RECURSIVE RETRIEVAL ARCHITECTURE:

The system implements a two-stage retrieval strategy:
  Stage 1: Find the most semantically relevant CLUSTER SUMMARIES
  Stage 2: Automatically fetch DETAILED PROTEINS only from those clusters
  
This prevents context overflow while maintaining comprehensive coverage.
"""

from .data_parsers import ClusterNodeRec, build_cluster_records
from .graph_builder import build_graph_indexes
from .rag_system import create_bio_rag_system
from .utils import sample_clusters_intelligently
from .config import BioRAGConfig, get_default_config, get_mock_data_config

__version__ = "0.1.0"
__author__ = "BioRAG Team"

__all__ = [
    "ClusterNodeRec",
    "build_cluster_records", 
    "build_graph_indexes",
    "create_bio_rag_system",
    "sample_clusters_intelligently",
    "BioRAGConfig",
    "get_default_config",
    "get_mock_data_config"
] 