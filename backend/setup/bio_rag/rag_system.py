"""
Main RAG system module for the BioRAG system.
Coordinates query processing, retrieval, and response generation.
"""

import traceback
from typing import Dict, Callable
from llama_index.core.schema import NodeRelationship

from .data_parsers import ClusterNodeRec
from .internet_search import (
    search_internet_for_proteins, 
    synthesize_with_internet_data
)
from .utils import extract_protein_ids_from_text
from .config import BioRAGConfig


def create_bio_rag_system(
    recursive_query_engine, 
    cluster_records: Dict[str, ClusterNodeRec],
    config: BioRAGConfig
) -> Callable:
    """
    Create the main bio-RAG system with hooks for future internet search enhancement.
    
    Current: Uses recursive retriever (fast and reliable)
    Future: Add intelligent internet search filtering
    
    Args:
        recursive_query_engine: The configured recursive query engine
        cluster_records: Full dataset of cluster records
        config: Configuration object
        
    Returns:
        Callable query function
    """
    
    def query_bio_system(
        query: str, 
        use_internet_search: bool = None, 
        debug: bool = None
    ) -> str:
        """
        Main query interface for the bio-RAG system.
        
        Args:
            query: User's biological query
            use_internet_search: Whether to enhance with internet data (overrides config)
            debug: Whether to show detailed debug information (overrides config)
        
        Returns:
            str: Comprehensive response combining cluster/protein information
        """
        # Use config defaults if not specified
        if use_internet_search is None:
            use_internet_search = config.enable_internet_search
        if debug is None:
            debug = config.debug_mode
            
        if not query.strip():
            return "Please provide a query about proteins, clusters, or biological pathways."
        
        try:
            if debug:
                print("\n🔧 DEBUG MODE ENABLED")
                print("=" * 50)
                print(f"📝 Query: '{query}'")
                print("🔍 Starting recursive retrieval process...")
                
                # Debug: Check what the retriever components look like
                print("\n🏗️ RECURSIVE RETRIEVER STRUCTURE:")
                retriever = recursive_query_engine.retriever
                print(f"   - Retriever type: {type(retriever).__name__}")
                print(f"   - Has retriever_dict: {hasattr(retriever, 'retriever_dict')}")
                if hasattr(retriever, 'retriever_dict'):
                    print(f"   - Retriever dict keys: {list(retriever.retriever_dict.keys())}")
                print(f"   - Has node_dict: {hasattr(retriever, 'node_dict')}")
                if hasattr(retriever, 'node_dict'):
                    print(f"   - Node dict size: {len(retriever.node_dict)}")
                    
                    # Sample a few nodes to check their types and relationships
                    node_types = {}
                    parent_child_counts = {"has_parent": 0, "has_children": 0, "orphans": 0}
                    
                    for i, (node_id, node) in enumerate(retriever.node_dict.items()):
                        if i >= 5:  # Only check first 5 nodes
                            break
                        node_type = node.metadata.get("node_type", "unknown")
                        node_types[node_type] = node_types.get(node_type, 0) + 1
                        
                        has_parent = NodeRelationship.PARENT in node.relationships
                        has_children = NodeRelationship.CHILD in node.relationships
                        
                        if has_parent:
                            parent_child_counts["has_parent"] += 1
                        if has_children:
                            parent_child_counts["has_children"] += 1
                        if not has_parent and not has_children:
                            parent_child_counts["orphans"] += 1
                            
                        print(f"   - Sample node {i+1}: {node_type}, parent={has_parent}, children={has_children}")
                    
                    print(f"   - Node types in dict: {node_types}")
                    print(f"   - Relationship summary: {parent_child_counts}")
            
            if config.verbose:
                print("🔍 Searching clusters and proteins using recursive retrieval...")
            
            # Step 1: Always get cluster/protein context from recursive retriever
            cluster_results = recursive_query_engine.query(query)
            
            # Step 2: Future internet search enhancement
            if use_internet_search:
                if config.verbose:
                    print("🌐 Enhancing with internet search...")
                
                # Extract protein IDs for targeted internet search
                protein_ids = extract_protein_ids_from_text(str(cluster_results))
                
                # Search internet databases (future implementation)
                internet_data = search_internet_for_proteins(query, str(cluster_results), protein_ids)
                
                # Synthesize cluster context + internet data
                return synthesize_with_internet_data(cluster_results, internet_data, query)
            
            # For now, just return the recursive retriever results
            # (which already include cluster summaries + relevant proteins)
            return str(cluster_results)
            
        except Exception as e:
            if debug:
                print(f"\n❌ DETAILED ERROR DEBUG:")
                print(f"   - Error type: {type(e).__name__}")
                print(f"   - Error message: {str(e)}")
                print(f"   - Full traceback:")
                traceback.print_exc()
            return f"Error in bio-RAG system: {str(e)}"
    
    return query_bio_system


def init_pipeline(config: BioRAGConfig):
    """
    Initialize the complete BioRAG pipeline.
    
    Args:
        config: Configuration object with all settings
        
    Returns:
        Tuple of (bio_rag_system, all_cluster_records, recursive_query_engine)
    """
    from .data_parsers import build_cluster_records
    from .graph_builder import build_graph_indexes
    from .utils import sample_clusters_intelligently, validate_cluster_records
    
    # Validate configuration
    if not config.validate_paths():
        raise FileNotFoundError("Required data files not found. Check your configuration.")
    
    # Setup LlamaIndex
    config.setup_llama_index()
    
    # Step 1: Load ALL 600MB data into memory (complete structure)
    if config.verbose:
        print("Loading full dataset into memory...")
    all_cluster_records, roots = build_cluster_records(config.data_paths)
    
    # Validate the loaded data
    if not validate_cluster_records(all_cluster_records):
        raise ValueError("Cluster records validation failed")
    
    if config.verbose:
        print(f"Loaded {len(all_cluster_records)} total clusters")
    
    # Step 2: Pick subset for embedding (save money/time) AND truncate proteins
    if config.verbose:
        print(f"Selecting {config.max_clusters} clusters for embedding...")
    sampled_for_embedding = sample_clusters_intelligently(
        all_cluster_records, 
        config.max_clusters, 
        config.max_proteins_per_cluster
    )
    
    if config.verbose:
        print(f"Selected {len(sampled_for_embedding)} clusters for embedding")
    
    # Step 3: Build index ONLY for the sampled subset (cheap embedding)
    recursive_query_engine, summary_index = build_graph_indexes(sampled_for_embedding, config)
    
    # Step 4: Create system with full dataset available for lookups
    bio_rag_system = create_bio_rag_system(recursive_query_engine, all_cluster_records, config)
    
    return bio_rag_system, all_cluster_records, recursive_query_engine


def get_system_info(
    bio_rag_system, 
    cluster_records: Dict[str, ClusterNodeRec], 
    recursive_query_engine,
    config: BioRAGConfig
) -> Dict[str, any]:
    """
    Get comprehensive information about the initialized system.
    
    Args:
        bio_rag_system: The query function
        cluster_records: Full cluster records
        recursive_query_engine: The recursive query engine
        config: Configuration object
        
    Returns:
        Dict with system statistics and configuration
    """
    from .utils import get_cluster_stats
    from .graph_builder import get_retriever_stats
    
    system_info = {
        "config": {
            "llm_model": config.llm_model,
            "embedding_model": config.embedding_model,
            "max_clusters": config.max_clusters,
            "max_proteins_per_cluster": config.max_proteins_per_cluster,
            "similarity_top_k": config.similarity_top_k,
            "enable_internet_search": config.enable_internet_search
        },
        "dataset": get_cluster_stats(cluster_records),
        "retriever": get_retriever_stats(recursive_query_engine),
        "status": "initialized"
    }
    
    return system_info 