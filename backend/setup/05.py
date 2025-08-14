# requirements:
# pip install llama-index==0.10.52 llama-index-embeddings-openai llama-index-llms-openai faiss-cpu python-dotenv

"""
ADVANCED BIO RAG SYSTEM WITH RECURSIVE RETRIEVAL FOR MEGACONTEXT
==============================================================

This system demonstrates the CANONICAL LlamaIndex pattern for handling "megacontext" scenarios 
where potentially relevant information far exceeds any LLM's context window. The solution uses 
RECURSIVE RETRIEVAL to intelligently navigate hierarchical data structures.

🔄 RECURSIVE RETRIEVAL ARCHITECTURE:

The system implements a two-stage retrieval strategy:
  Stage 1: Find the most semantically relevant CLUSTER SUMMARIES
  Stage 2: Automatically fetch DETAILED PROTEINS only from those clusters
  
This prevents context overflow while maintaining comprehensive coverage.

KEY ARCHITECTURAL INNOVATIONS:

1. CANONICAL RECURSIVE RETRIEVAL PATTERN:
   - Uses LlamaIndex's RecursiveRetriever for cluster-first, protein-second strategy
   - Builds separate indices for summaries (parents) and details (children)
   - Links nodes through explicit NodeRelationship.PARENT/CHILD connections
   - Enables automatic hierarchical traversal without manual navigation

2. INTELLIGENT CONTEXT FILTERING:
   - Only retrieves proteins from semantically relevant clusters
   - Scales to millions of proteins without context window overflow
   - Maintains semantic coherence across abstraction levels
   - Provides both broad understanding and specific details

3. SCALABLE MEGACONTEXT HANDLING:
   - Context size = k_clusters + (proteins_per_cluster × k_clusters)
   - Linear scaling regardless of total dataset size
   - Handles 1M+ proteins through intelligent cluster selection
   - No brute-force search - hierarchy guides retrieval

4. BIDIRECTIONAL NODE RELATIONSHIPS:
   - Parent cluster summaries link to child protein details
   - Child proteins reference their parent cluster context
   - Enables rich contextual understanding at both levels
   - Maintains semantic relationships throughout retrieval

🎯 MEGACONTEXT NAVIGATION PATTERN:
   Query → RecursiveRetriever → Top-K Clusters → Their Proteins → Synthesis

This approach represents the state-of-the-art for handling massive datasets in RAG systems,
scaling horizontally as data grows while maintaining fast, relevant responses.
"""

import os
from dataclasses import dataclass
from collections import defaultdict
from dotenv import load_dotenv
from typing import Optional, Any

from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings, get_response_synthesizer
from llama_index.core.indices.composability import ComposableGraph
from llama_index.core.schema import TextNode, MetadataMode, NodeRelationship, RelatedNodeInfo, IndexNode
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from typing import List, Optional, Dict
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import load_index_from_storage
import pickle
import hashlib

# ---------- setup ----------
load_dotenv()
Settings.llm = OpenAI(model="gpt-4o-mini")  # change if needed
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")

# ---------- data adapters (STRING) ----------
@dataclass
class ClusterNodeRec:
    cluster_id: str
    summary: str
    proteins: list[dict]  # Changed from list[str] to list[dict] to include metadata
    children: list[str]

def parse_clusters_info(path):  # 9606.clusters.info.v12.0
    desc = {}
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.rstrip("\n").split("\t")
            # skip header rows
            if parts and (parts[0] == "string_taxon_id" or parts[1] == "cluster_id"):
                continue
            _, cluster_id, _, best_desc = parts
            desc[cluster_id] = best_desc
    return desc

def parse_clusters_tree(path):  # 9606.clusters.tree.v12.0
    parent_to_children = defaultdict(list)
    child_to_parent = {}
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.rstrip("\n").split("\t")
            # skip header
            if parts and (parts[0] == "string_taxon_id" or parts[1] == "child_cluster_id"):
                continue
            _, child_id, parent_id = parts
            parent_to_children[parent_id].append(child_id)
            child_to_parent[child_id] = parent_id
    return parent_to_children, child_to_parent

def parse_clusters_proteins(path):  # 9606.clusters.proteins.v12.0
    clust_to_prots = defaultdict(list)
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.rstrip("\n").split("\t")
            # skip header
            if parts and (parts[0] == "string_taxon_id" or parts[1] == "cluster_id"):
                continue
            _, clust_id, prot_id = parts
            clust_to_prots[clust_id].append(prot_id)
    return clust_to_prots

def parse_protein_info(path):  # 9606.protein.info.v12.0
    prot_meta = {}
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.rstrip("\n").split("\t")
            # skip header
            if parts and (parts[0] == "string_protein_id" or parts[0] == "string_taxon_id"):
                continue
            pid, pref, size, annot = parts[0], parts[1], parts[2], parts[3] if len(parts) > 3 else ""
            prot_meta[pid] = {"preferred_name": pref, "size": size, "annotation": annot}
    return prot_meta

# ---------- gene → protein resolver (your API) ----------
def gene_to_string_proteins(ensembl_gene_id: str) -> list[str]:
    """
    TODO: call your gene DB API here to return STRING protein IDs for the gene.
    For now, return a small demo list or empty list if unknown.
    """
    # e.g., use Ensembl → UniProt → STRING mapping
    return []  # placeholder

# ---------- build graph: cluster → subclusters → proteins ----------
def build_cluster_records(paths: dict) -> dict[str, ClusterNodeRec]:
    # 9606	CL:39184	7	mixed, incl. Protein FAM81, and Bacterial Ig-like domain 2
    desc = parse_clusters_info(paths["clusters_info"])
    #9606	CL:39184	CL:39182
    p2c, c2p = parse_clusters_tree(paths["clusters_tree"])
    #9606	CL:39184	9606.ENSP00000473493
    c2prots = parse_clusters_proteins(paths["clusters_proteins"])
    
    protein_meta = parse_protein_info(paths["protein_info"])

    # roots = clusters that never appear as child
    all_ids = set(desc.keys())
    children = set(c2p.keys())
    roots = list(all_ids - children)
    
    # assemble records with enriched protein information
    records = {}
    for cid in all_ids:
        # Enrich protein IDs with metadata
        enriched_proteins = []
        for pid in c2prots.get(cid, []):
            meta = protein_meta.get(pid, {})
            enriched_proteins.append({
                "protein_id": pid,
                "preferred_name": meta.get("preferred_name", "NA"),
                "size": meta.get("size", "NA"),
                "annotation": meta.get("annotation", "NA")
            })
        
        records[cid] = ClusterNodeRec(
            cluster_id=cid,
            summary=desc.get(cid, "No description"),
            proteins=enriched_proteins,
            children=p2c.get(cid, [])
        )
    
    return records, roots

def build_graph_indexes(cluster_records: dict[str, ClusterNodeRec], persist_dir: str = "./storage"):
    """
    Build hierarchical cluster structure with recursive retrieval capabilities.
    Uses summary-first retrieval with per-cluster protein query engines via query_engine_dict.
    """
    
    # Create a hash of the cluster data to detect changes
    data_hash = hashlib.md5(str(sorted(cluster_records.items())).encode()).hexdigest()
    hash_file = os.path.join(persist_dir, "data_hash.txt")
    index_dir = os.path.join(persist_dir, "recursive_index")
    
    # Check if we can load existing index
    if os.path.exists(hash_file) and os.path.exists(index_dir):
        with open(hash_file, 'r') as f:
            stored_hash = f.read().strip()
        
        if stored_hash == data_hash:
            print("📁 Loading existing index from storage...")
            try:
                storage_context = StorageContext.from_defaults(persist_dir=index_dir)
                main_index = load_index_from_storage(storage_context)
                
                # Rebuild summary nodes and per-cluster protein query engines using unified approach
                print("🔄 Rebuilding summary nodes and per-cluster protein engines...")
                summary_nodes = []
                query_engine_dict = {}
                
                for cid, rec in cluster_records.items():
                    cluster_protein_names = [p["preferred_name"] for p in rec.proteins]
                    summary_text = f"""Cluster ID: {cid}
Summary: {rec.summary}
Children clusters: {len(rec.children)}
Child cluster IDs: {', '.join(rec.children) if rec.children else 'None'}
Protein count: {len(rec.proteins)}
Key proteins: {', '.join(cluster_protein_names)}
"""
                    
                    engine_id = f"cluster_{cid}_proteins" # ID for the query engine
                    
                    # Use the same unified IndexNode approach as the new building section
                    summary_as_index_node = IndexNode(
                        text=summary_text,          # The text is the full summary.
                        index_id=engine_id,          # It points to the protein engine.
                        metadata={
                            "node_type": "cluster_gateway", # New combined node type
                            "cluster_id": cid,
                            "level": "summary_and_index"
                        }
                    )

                    summary_nodes.append(summary_as_index_node)
                    
                    # Build per-cluster protein nodes and query engine (same as new building section)
                    if len(rec.proteins) > 0:
                        protein_nodes = []
                        for p in rec.proteins:
                            protein_text = f"""Protein ID: {p['protein_id']}
Preferred Name: {p['preferred_name']}
Annotation: {p['annotation']}
Size: {p['size']} amino acids
Cluster: {cid}
Cluster Summary: {rec.summary}
"""
                            protein_node = TextNode(
                                text=protein_text,
                                metadata={
                                    "node_type": "protein",
                                    "cluster_id": cid,
                                    "protein_id": p['protein_id'],
                                    "preferred_name": p['preferred_name'],
                                    "level": "detail"
                                }
                            )
                            protein_nodes.append(protein_node)
                        
                        protein_index = VectorStoreIndex(protein_nodes)
                        protein_query_engine = protein_index.as_query_engine(similarity_top_k=3)
                        query_engine_dict[engine_id] = protein_query_engine
                
                # Build a new index over summary nodes to enable entering engines
                main_index = VectorStoreIndex(summary_nodes)
                summary_retriever = main_index.as_retriever(similarity_top_k=2, verbose=True)
                
                recursive_retriever = RecursiveRetriever(
                    "vector",
                    retriever_dict={"vector": summary_retriever},
                    query_engine_dict=query_engine_dict,
                    verbose=True,
                )
                
                recursive_query_engine = RetrieverQueryEngine.from_args(
                    retriever=recursive_retriever,
                    response_mode="tree_summarize",
                    verbose=True
                )
                
                print("✅ Successfully loaded existing index!")
                print(f"  - Summary nodes (rebuilt): {len(summary_nodes)}")
                print(f"  - Protein engines: {len(query_engine_dict)}")
                
                return recursive_query_engine, main_index
            except Exception as e:
                print(f"⚠️  Failed to load existing index: {e}")
                print("Building new index...")
    
    print("🏗️  Building new recursive retrieval index...")
    print("📚 Using summary-first + per-cluster protein engines pattern...")
    
    # 1) Build summary nodes
    summary_nodes = []
    query_engine_dict = {}
    
    for cid, rec in cluster_records.items():
        cluster_protein_names = [p["preferred_name"] for p in rec.proteins]
        summary_text = f"""Cluster ID: {cid}
Summary: {rec.summary}
Children clusters: {len(rec.children)}
Child cluster IDs: {', '.join(rec.children) if rec.children else 'None'}
Protein count: {len(rec.proteins)}
Key proteins: {', '.join(cluster_protein_names)}
"""
        
        engine_id = f"cluster_{cid}_proteins" # ID for the query engine
        
        # 💡 YOUR PROPOSED CHANGE IS HERE 💡
        # Combine the summary and index node into a single IndexNode.
        summary_as_index_node = IndexNode(
            text=summary_text,          # The text is the full summary.
            index_id=engine_id,          # It points to the protein engine.
            metadata={
                "node_type": "cluster_gateway", # New combined node type
                "cluster_id": cid,
                "level": "summary_and_index"
            }
        )

        summary_nodes.append(summary_as_index_node)
        
        # 2) Per-cluster IndexNode + protein engine
        if len(rec.proteins) > 0:
            # Build per-cluster protein nodes and a proper query engine
            protein_nodes = []
            for p in rec.proteins:
                protein_text = f"""Protein ID: {p['protein_id']}
Preferred Name: {p['preferred_name']}
Annotation: {p['annotation']}
Size: {p['size']} amino acids
Cluster: {cid}
Cluster Summary: {rec.summary}
"""
                protein_node = TextNode(
                    text=protein_text,
                    metadata={
                        "node_type": "protein",
                        "cluster_id": cid,
                        "protein_id": p['protein_id'],
                        "preferred_name": p['preferred_name'],
                        "level": "detail"
                    }
                )
                protein_nodes.append(protein_node)
            
            protein_index = VectorStoreIndex(protein_nodes)
            protein_query_engine = protein_index.as_query_engine(similarity_top_k=3)
            query_engine_dict[engine_id] = protein_query_engine
    
    main_index = VectorStoreIndex(summary_nodes)
    summary_retriever = main_index.as_retriever(similarity_top_k=2, verbose=True)
    
    print(f"🔧 Creating RecursiveRetriever (summary-first → protein engines)...")
    recursive_retriever = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": summary_retriever},
        query_engine_dict=query_engine_dict,
        verbose=True,
    )
    
    print("✅ RecursiveRetriever created successfully!")
    if hasattr(recursive_retriever, '_retriever_dict'):
        print(f"  - Retriever dict keys: {list(recursive_retriever._retriever_dict.keys())}")
    print(f"  - Protein engines: {len(query_engine_dict)}")
    
    
    recursive_query_engine = RetrieverQueryEngine.from_args(
        retriever=recursive_retriever,
        response_mode="tree_summarize",
        verbose=True
    )
    
    print("✅ Summary-first recursive retrieval system built successfully!")
    
    # Save the index
    try:
        os.makedirs(persist_dir, exist_ok=True)
        main_index.storage_context.persist(persist_dir=index_dir)
        with open(hash_file, 'w') as f:
            f.write(data_hash)
        print("💾 Index saved to storage for future use!")
    except Exception as e:
        print(f"⚠️  Failed to save index: {e}")
    
    return recursive_query_engine, main_index

# ---------- internet search functions (fake for now) ----------
def search_protein_online(protein_name: str, protein_id: str, query_context: str = "") -> dict:
    """
    Fake function to simulate searching for protein information online.
    In reality, this would query databases like UniProt, PDB, etc.
    """
    # This is a placeholder - replace with actual web search/API calls
    fake_results = {
        "protein_name": protein_name,
        "protein_id": protein_id,
        "pathways": [f"Pathway related to {protein_name}", f"Secondary pathway for {protein_name}"],
        "interactions": [f"Interacts with protein A", f"Interacts with protein B"],
        "diseases": [f"Associated with disease X", f"Linked to condition Y"],
        "function": f"Detailed function of {protein_name} related to {query_context}",
        "structure": f"3D structure information for {protein_name}",
        "publications": [f"Recent paper about {protein_name}", f"Review of {protein_name} function"],
        "localization": f"Cellular localization of {protein_name}",
        "expression": f"Expression pattern of {protein_name}"
    }
    return fake_results

def get_cluster_proteins_and_search(cluster_id: str, query_str: str, cluster_records: dict) -> dict:
    """
    Get proteins from a specific cluster and search for detailed information about each.
    """
    # normalize inputs like "39184" or "CL_39184" to canonical key "CL:39184"
    lookup_id = cluster_id
    if lookup_id.isdigit():
        lookup_id = f"CL:{lookup_id}"
    elif lookup_id.startswith("CL_"):
        lookup_id = lookup_id.replace("CL_", "CL:", 1)
    
    rec = cluster_records.get(cluster_id)
    if not rec:
        rec = cluster_records.get(lookup_id)
    if not rec:
        return {"error": f"cluster {cluster_id} not found"}
    
    results = {
        "cluster_id": cluster_id,
        "cluster_summary": rec.summary,
        "total_proteins": len(rec.proteins),
        "protein_details": []
    }
    
    # Search for detailed information about each protein (limit to first 5 for performance)
    for protein_dict in rec.proteins[:5]:
        protein_info = search_protein_online(
            protein_dict["preferred_name"], 
            protein_dict["protein_id"], 
            query_str
        )
        protein_info.update({
            "basic_annotation": protein_dict["annotation"],
            "size": protein_dict["size"]
        })
        results["protein_details"].append(protein_info)
    
    return results

def sample_clusters_intelligently(cluster_records: dict[str, ClusterNodeRec], max_clusters: int = 1000, max_proteins_per_cluster: int = 20) -> dict[str, ClusterNodeRec]:
    """
    Intelligently sample clusters to get a diverse, representative subset for embedding.
    
    Strategy:
    - 40% by importance (clusters with most proteins)
    - 40% by functional diversity (different biological categories)
    - 20% random sampling (catch edge cases)
    - Limit proteins per cluster to avoid token overflow
    
    Args:
        cluster_records: Full dataset of all clusters
        max_clusters: Maximum number of clusters to sample
        max_proteins_per_cluster: Maximum proteins to keep per cluster (default: 20)
    
    Returns:
        dict: Sampled subset of clusters with truncated protein lists
    """
    import random
    
    if len(cluster_records) <= max_clusters:
        print(f"Dataset has {len(cluster_records)} clusters, using all (≤ {max_clusters})")
        # Still need to truncate proteins per cluster
        truncated_records = {}
        for cid, rec in cluster_records.items():
            truncated_rec = ClusterNodeRec(
                cluster_id=rec.cluster_id,
                summary=rec.summary,
                proteins=rec.proteins[:max_proteins_per_cluster],  # Truncate proteins
                children=rec.children
            )
            truncated_records[cid] = truncated_rec
        return truncated_records
    
    print(f"Sampling {max_clusters} clusters from {len(cluster_records)} total clusters...")
    print(f"Limiting each cluster to max {max_proteins_per_cluster} proteins...")
    
    sampled = {}
    
    # Stage 1: Importance sampling (40%) - clusters with most proteins
    print("📊 Stage 1: Selecting clusters by importance (protein count)...")
    sorted_by_protein_count = sorted(
        cluster_records.items(), 
        key=lambda x: len(x[1].proteins), 
        reverse=True
    )
    
    importance_count = int(max_clusters * 0.4)
    for cid, rec in sorted_by_protein_count[:importance_count]:
        # Truncate proteins to avoid token overflow
        truncated_rec = ClusterNodeRec(
            cluster_id=rec.cluster_id,
            summary=rec.summary,
            proteins=rec.proteins[:max_proteins_per_cluster],
            children=rec.children
        )
        sampled[cid] = truncated_rec
    
    print(f"   ✓ Selected {len(sampled)} important clusters (top by protein count)")
    
    # Stage 2: Functional diversity sampling (40%) - different biological categories
    print("🧬 Stage 2: Selecting clusters by functional diversity...")
    functional_keywords = [
        'kinase', 'phosphatase', 'transcription', 'metabolism', 'metabolic',
        'signaling', 'signal', 'membrane', 'nuclear', 'mitochondrial',
        'DNA', 'RNA', 'ribosomal', 'transport', 'channel', 'receptor',
        'enzyme', 'binding', 'regulatory', 'oxidase', 'reductase',
        'synthetase', 'hydrolase', 'transferase', 'lyase', 'ligase'
    ]
    
    diversity_count = int(max_clusters * 0.4)
    diversity_added = 0
    
    # Shuffle keywords to avoid bias toward early keywords
    random.shuffle(functional_keywords)
    
    for keyword in functional_keywords:
        if diversity_added >= diversity_count:
            break
            
        # Find clusters matching this keyword that aren't already sampled
        keyword_matches = []
        for cid, rec in cluster_records.items():
            if (cid not in sampled and 
                keyword.lower() in rec.summary.lower()):
                keyword_matches.append((cid, rec))
        
        # Take a few from this category (distribute evenly across keywords)
        max_per_keyword = max(1, diversity_count // len(functional_keywords))
        selected_from_keyword = min(max_per_keyword, len(keyword_matches), diversity_count - diversity_added)
        
        # Sort by protein count within this category to get the best representatives
        keyword_matches.sort(key=lambda x: len(x[1].proteins), reverse=True)
        
        for cid, rec in keyword_matches[:selected_from_keyword]:
            # Truncate proteins to avoid token overflow
            truncated_rec = ClusterNodeRec(
                cluster_id=rec.cluster_id,
                summary=rec.summary,
                proteins=rec.proteins[:max_proteins_per_cluster],
                children=rec.children
            )
            sampled[cid] = truncated_rec
            diversity_added += 1
    
    print(f"   ✓ Selected {diversity_added} functionally diverse clusters")
    
    # Stage 3: Random sampling (20%) - fill remaining slots
    print("🎲 Stage 3: Random sampling to fill remaining slots...")
    remaining_clusters = {
        cid: rec for cid, rec in cluster_records.items() 
        if cid not in sampled
    }
    
    random_count = max_clusters - len(sampled)
    if random_count > 0 and remaining_clusters:
        random_selection = random.sample(
            list(remaining_clusters.items()), 
            min(random_count, len(remaining_clusters))
        )
        
        for cid, rec in random_selection:
            # Truncate proteins to avoid token overflow
            truncated_rec = ClusterNodeRec(
                cluster_id=rec.cluster_id,
                summary=rec.summary,
                proteins=rec.proteins[:max_proteins_per_cluster],
                children=rec.children
            )
            sampled[cid] = truncated_rec
    
    print(f"   ✓ Added {min(random_count, len(remaining_clusters))} random clusters")
    
    # Summary statistics
    total_proteins = sum(len(rec.proteins) for rec in sampled.values())
    non_empty_clusters = sum(1 for rec in sampled.values() if len(rec.proteins) > 0)
    
    print(f"\n📋 Sampling Summary:")
    print(f"   • Total clusters sampled: {len(sampled)}")
    print(f"   • Clusters with proteins: {non_empty_clusters}")
    print(f"   • Total proteins included: {total_proteins}")
    print(f"   • Average proteins per cluster: {total_proteins/len(sampled):.1f}")
    print(f"   • Max proteins per cluster: {max_proteins_per_cluster}")
    
    return sampled

# ---------- ultra-simple linear bio-RAG system ----------
import re

def extract_protein_ids_from_text(text: str) -> list[str]:
    """Extract protein IDs from text for future internet search"""
    protein_patterns = [
        r'9606\.ENSP\d+',  # STRING protein IDs
        r'ENSP\d+',        # Ensembl protein IDs
        r'\b[A-Z][A-Z0-9]{2,8}\b'  # Gene symbols (rough pattern)
    ]
    
    protein_ids = []
    for pattern in protein_patterns:
        matches = re.findall(pattern, text)
        protein_ids.extend(matches)
    
    return list(set(protein_ids))  # Remove duplicates

def search_internet_for_proteins(query: str, cluster_context: str, protein_ids: list[str]) -> dict:
    """
    Future: Search internet databases (UniProt, PDB, etc.) for relevant proteins.
    This will need intelligent filtering to avoid data overload.
    
    Args:
        query: Original user query
        cluster_context: Results from recursive retriever
        protein_ids: Specific protein IDs to search for
    
    Returns:
        dict: Filtered internet search results
    """
    # TODO: Implement when ready for internet search
    # Strategy:
    # 1. Query UniProt API for protein_ids
    # 2. Get recent papers from PubMed
    # 3. Query STRING database for interactions
    # 4. Use LLM to filter results by relevance to original query
    # 5. Limit data volume (e.g., max 5 papers, 10 interactions)
    
    return {
        "status": "not_implemented",
        "note": "Internet search will be implemented here",
        "protein_ids": protein_ids[:5],  # Limit for future implementation
        "query": query
    }

def synthesize_with_internet_data(cluster_results: str, internet_data: dict, query: str) -> str:
    """
    Future: Intelligently combine cluster context with internet search results.
    This will need LLM-based filtering and synthesis.
    """
    if internet_data.get("status") == "not_implemented":
        return str(cluster_results)
    
    # TODO: Implement smart synthesis when internet search is ready
    # Strategy:
    # 1. Use LLM to filter internet data by relevance
    # 2. Combine cluster context with filtered internet findings
    # 3. Ensure response stays within reasonable length
    # 4. Prioritize cluster context (fast, reliable) over internet data
    
    return str(cluster_results)

def create_bio_rag_system(recursive_query_engine, cluster_records: dict[str, ClusterNodeRec]):
    """
    Ultra-simple bio-RAG system with hooks for future internet search enhancement.
    
    Current: Just use recursive retriever (fast and reliable)
    Future: Add intelligent internet search filtering
    """
    
    def query_bio_system(query: str, use_internet_search: bool = False, debug: bool = True) -> str:
        """
        Main query interface for the bio-RAG system.
        
        Args:
            query: User's biological query
            use_internet_search: Whether to enhance with internet data (future feature)
            debug: Whether to show detailed debug information
        
        Returns:
            str: Comprehensive response combining cluster/protein information
        """
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
            
            print("🔍 Searching clusters and proteins using recursive retrieval...")
            
            # Step 1: Always get cluster/protein context from recursive retriever
            cluster_results = recursive_query_engine.query(query)
            
            # Step 2: Future internet search enhancement
            if use_internet_search:
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
                import traceback
                print(f"\n❌ DETAILED ERROR DEBUG:")
                print(f"   - Error type: {type(e).__name__}")
                print(f"   - Error message: {str(e)}")
                print(f"   - Full traceback:")
                traceback.print_exc()
            return f"Error in bio-RAG system: {str(e)}"
    
    return query_bio_system  # Just return the function, no extra methods

# ---------- end-to-end init ----------
def init_pipeline(paths: dict, max_clusters: int = 10, max_proteins_per_cluster: int = 20):
    # Step 1: Load ALL 600MB data into memory (complete structure)
    print("Loading full 600MB dataset into memory...")
    all_cluster_records, roots = build_cluster_records(paths)
    print(f"Loaded {len(all_cluster_records)} total clusters")
    
    # Step 2: Pick subset for embedding (save money/time) AND truncate proteins
    print(f"Selecting {max_clusters} clusters for embedding...")
    sampled_for_embedding = sample_clusters_intelligently(all_cluster_records, max_clusters, max_proteins_per_cluster)
    print(f"Selected {len(sampled_for_embedding)} clusters for embedding")
    
    # Step 3: Build index ONLY for the sampled subset (cheap embedding)
    recursive_query_engine, summary_index = build_graph_indexes(sampled_for_embedding)
    
    # Step 4: Create system with full dataset available for lookups
    bio_rag_system = create_bio_rag_system(recursive_query_engine, all_cluster_records)
    
    return bio_rag_system, all_cluster_records, recursive_query_engine

# ---------- demo ----------
if __name__ == "__main__":
    # TODO: set paths to your real extracted files
    paths = {
        "clusters_info": "data/9606.clusters.info.v12.0.txt",
        "clusters_tree": "data/9606.clusters.tree.v12.0.txt",
        "clusters_proteins": "data/9606.clusters.proteins.v12.0.txt",
        "protein_info": "data/9606.protein.info.v12.0.txt",
    }  
    # paths = {
    #     "clusters_info": "mockdata1/mock_clusters_info.txt",
    #     "clusters_tree": "mockdata1/mock_clusters_tree.txt",
    #     "clusters_proteins": "mockdata1/mock_clusters_proteins.txt",
    #     "protein_info": "mockdata1/mock_protein_info.txt",
    # }
    
    print("Initializing ULTRA-SIMPLE bio RAG system with recursive retrieval...")
    
    # Build the linear bio-RAG system
    bio_rag_system, cluster_records, recursive_query_engine = init_pipeline(paths)
    
    print("✅ Ultra-simple linear architecture established")
    print("✅ Recursive retrieval handles cluster→protein navigation")
    print("✅ No ReAct overhead - direct function calls")
    print("✅ Ready for future internet search enhancement")
    print("✅ System scales to millions of proteins efficiently")
    
    print("\n" + "="*80)
    print("🚀 ULTRA-SIMPLE BIO-RAG SYSTEM READY")
    print("="*80)
    print("Type 'quit' or 'exit' to end the session.")
    print("")
    print("🔥 TRY THESE QUERIES:")
    print("📋 'Find protein kinases involved in cell cycle regulation'")
    print("📋 'What are the main metabolic enzyme clusters?'") 
    print("📋 'Tell me about cluster CL:39184'")
    print("📋 'Proteins involved in DNA repair mechanisms'")
    print("📋 'Show me signaling pathway clusters and their proteins'")
    print("")
    print("🔧 DEBUG COMMANDS:")
    print("📋 'debug' - Run detailed debug analysis of recursive retriever")
    print("")
    print("💡 System workflow:")
    print("   🔍 Query → Recursive Retrieval → Response")
    print("   🌐 Future: Query → Recursive Retrieval → Internet Search → Synthesis")
    print("="*80)
    
    # Interactive chat loop with ultra-simple pipeline
    while True:
        try:
            user_input = input("\n🧬 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye! Thanks for testing the bio-RAG system!")
                break
                
            if not user_input:
                continue
                
            # Special debug command
            if user_input.lower() == 'debug':
                debug_recursive_retriever(recursive_query_engine, cluster_records)
                continue
                
            print("\n⚡ Processing...")
            # For now, internet search is disabled (use_internet_search=False)
            response = bio_rag_system(user_input, use_internet_search=False)
            print(f"\n🤖 Response:\n{response}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.\n")
