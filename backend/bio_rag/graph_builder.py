"""
Graph building module for the BioRAG system.
Handles LlamaIndex vector store construction and recursive retriever setup.
"""

import os
import hashlib
from typing import Dict, Tuple

from llama_index.core import VectorStoreIndex, StorageContext, get_response_synthesizer
from llama_index.core.schema import TextNode, IndexNode
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import load_index_from_storage

from .data_parsers import ClusterNodeRec
from .config import BioRAGConfig


def build_graph_indexes(
    cluster_records: Dict[str, ClusterNodeRec], 
    config: BioRAGConfig
) -> Tuple[RetrieverQueryEngine, VectorStoreIndex]:
    """
    Build hierarchical cluster structure with a CORRECT RecursiveRetriever setup.
    This version uses sub-retrievers to fetch raw nodes for unified synthesis.
    
    Args:
        cluster_records: Dictionary of cluster records to index
        config: Configuration object with settings
        
    Returns:
        Tuple of (recursive_query_engine, main_index)
    """
    os.makedirs(config.persist_dir, exist_ok=True)
    
    # Create a hash of the cluster data to detect changes
    data_hash = hashlib.md5(str(sorted(cluster_records.items())).encode()).hexdigest()
    hash_file = os.path.join(config.persist_dir, "data_hash.txt")
    index_dir = os.path.join(config.persist_dir, "recursive_index")

    # --- Build in-memory components (nodes and sub-retrievers) ---
    # This logic is needed whether we load or build the main index from disk.
    if config.verbose:
        print("🧠 Building in-memory components (gateway nodes and sub-retrievers)...")
        
    gateway_nodes = []
    sub_retriever_dict = {}

    for cid, rec in cluster_records.items():
        # This ID links the gateway node to its corresponding sub-retriever
        retriever_id = f"cluster_{cid}_proteins"
        
        # Create a single gateway node per cluster
        cluster_protein_names = [p["preferred_name"] for p in rec.proteins]
        summary_text = f"""Cluster ID: {cid}
Summary: {rec.summary}
Children clusters: {len(rec.children)}
Child cluster IDs: {', '.join(rec.children) if rec.children else 'None'}
Protein count: {len(rec.proteins)}
Key proteins: {', '.join(cluster_protein_names)}
"""
        gateway_node = IndexNode(
            text=summary_text,
            index_id=retriever_id, # This ID points to the sub-retriever
            metadata={
                "node_type": "cluster_gateway",
                "cluster_id": cid,
            }
        )
        gateway_nodes.append(gateway_node)
        
        # Build a sub-RETRIEVER, not a query engine
        if len(rec.proteins) > 0:
            protein_nodes = [
                TextNode(
                    text=f"""Protein ID: {p['protein_id']}
Preferred Name: {p['preferred_name']}
Annotation: {p['annotation']}
Size: {p['size']} amino acids
Cluster: {cid}
Cluster Summary: {rec.summary}""",
                    metadata={
                        "node_type": "protein",
                        "cluster_id": cid,
                        "protein_id": p['protein_id'],
                    }
                ) for p in rec.proteins
            ]
            protein_index = VectorStoreIndex(protein_nodes)
            protein_retriever = protein_index.as_retriever(
                similarity_top_k=config.similarity_top_k
            )
            sub_retriever_dict[retriever_id] = protein_retriever

    # --- Loading or Building the Main Vector Index ---
    # This block correctly handles caching for the main index embeddings.
    if os.path.exists(hash_file) and os.path.exists(index_dir):
        with open(hash_file, 'r') as f:
            stored_hash = f.read().strip()
        if stored_hash == data_hash:
            if config.verbose:
                print(f"📁 Loading existing main_index from {index_dir}...")
            storage_context = StorageContext.from_defaults(persist_dir=index_dir)
            main_index = load_index_from_storage(storage_context)
            if config.verbose:
                print("✅ Successfully loaded main_index from disk.")
        else:
            if config.verbose:
                print("⚠️ Data has changed. Building new main_index...")
            main_index = VectorStoreIndex(gateway_nodes)
            main_index.storage_context.persist(persist_dir=index_dir)
            with open(hash_file, 'w') as f: 
                f.write(data_hash)
            if config.verbose:
                print("✅ New main_index built and saved to disk.")
    else:
        if config.verbose:
            print("🏗️ No existing index found. Building new main_index...")
        main_index = VectorStoreIndex(gateway_nodes)
        main_index.storage_context.persist(persist_dir=index_dir)
        with open(hash_file, 'w') as f: 
            f.write(data_hash)
        if config.verbose:
            print("✅ New main_index built and saved to disk.")

    # --- Assemble the final query engine with the CORRECTED RecursiveRetriever ---
    if config.verbose:
        print("🔧 Creating RecursiveRetriever...")
    
    # This is the retriever for the top-level index of cluster gateways
    root_retriever = main_index.as_retriever(
        similarity_top_k=config.cluster_similarity_top_k, 
        verbose=config.verbose
    )
    
    # Combine all retrievers into a single dictionary
    # The 'vector' key is a default ID for the root retriever.
    full_retriever_dict = {"vector": root_retriever, **sub_retriever_dict}

    recursive_retriever = RecursiveRetriever(
        "vector", # This tells it to start with the root retriever
        retriever_dict=full_retriever_dict,
        verbose=config.verbose,
    )
    
    # The final query engine wraps the corrected retriever for synthesis
    recursive_query_engine = RetrieverQueryEngine.from_args(
        retriever=recursive_retriever,
        response_mode="tree_summarize",
        verbose=config.verbose
    )
    
    if config.verbose:
        print("✅ Summary-first recursive retrieval system built successfully!")
        
    return recursive_query_engine, main_index


def debug_recursive_retriever(query_engine, cluster_records: Dict[str, ClusterNodeRec]):
    """
    A helper function to test the raw output of the RecursiveRetriever.
    This bypasses the final synthesis step to show exactly which nodes are being fetched.
    
    Args:
        query_engine: The recursive query engine to debug
        cluster_records: Full cluster records for context
    """
    print("\n" + "="*50)
    print("🛠️ RUNNING RECURSIVE RETRIEVER DEBUG 🛠️")
    print("="*50)
    
    retriever = query_engine.retriever
    query = input("Enter a query to test retrieval: ").strip()
    if not query:
        print("No query provided. Aborting debug.")
        return

    print(f"\n🔍 Retrieving raw nodes for query: '{query}'...")
    try:
        retrieved_nodes = retriever.retrieve(query)
        print(f"\n✅ Retrieval successful! Found {len(retrieved_nodes)} raw nodes.")
        
        if not retrieved_nodes:
            print("No nodes were retrieved. The query may not have matched any cluster summaries.")
            return

        print("\n--- RETRIEVED NODES ANALYSIS ---")
        for i, node_with_score in enumerate(retrieved_nodes):
            node = node_with_score.node
            node_type = node.metadata.get("node_type", "unknown")
            cluster_id = node.metadata.get("cluster_id", "N/A")
            protein_id = node.metadata.get("protein_id", "N/A")
            
            print(f"\n📄 Node {i+1} (Score: {node_with_score.score:.4f})")
            print(f"   - Type: {node_type}")
            if node_type == "protein":
                print(f"   - From Cluster: {cluster_id}")
                print(f"   - Protein ID: {protein_id}")
            else:
                print(f"   - Cluster ID: {cluster_id}")
            print(f"   - Text: {node.get_content()[:150].strip()}...")
        print("\n" + "="*50)

    except Exception as e:
        print(f"\n❌ An error occurred during retrieval: {e}")
    print("Debug session finished.")


def get_retriever_stats(query_engine) -> Dict[str, any]:
    """
    Get statistics about the recursive retriever setup.
    
    Args:
        query_engine: The recursive query engine
        
    Returns:
        Dict with retriever statistics
    """
    retriever = query_engine.retriever
    stats = {
        "retriever_type": type(retriever).__name__,
        "has_retriever_dict": hasattr(retriever, 'retriever_dict'),
        "has_node_dict": hasattr(retriever, 'node_dict'),
    }
    
    if hasattr(retriever, 'retriever_dict'):
        stats["retriever_dict_keys"] = list(retriever.retriever_dict.keys())
        stats["num_sub_retrievers"] = len(retriever.retriever_dict) - 1  # Exclude 'vector'
        
    if hasattr(retriever, 'node_dict'):
        stats["node_dict_size"] = len(retriever.node_dict)
        
        # Sample node types
        node_types = {}
        for i, (node_id, node) in enumerate(retriever.node_dict.items()):
            if i >= 10:  # Only check first 10 nodes
                break
            node_type = node.metadata.get("node_type", "unknown")
            node_types[node_type] = node_types.get(node_type, 0) + 1
            
        stats["sample_node_types"] = node_types
        
    return stats 