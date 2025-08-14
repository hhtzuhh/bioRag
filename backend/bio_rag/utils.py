"""
Utility functions for the BioRAG system.
Includes intelligent sampling, text processing, and helper functions.
"""

import random
import re
from typing import Dict, List, Set
from .data_parsers import ClusterNodeRec


def sample_clusters_intelligently(
    cluster_records: Dict[str, ClusterNodeRec], 
    max_clusters: int = 1000, 
    max_proteins_per_cluster: int = 20
) -> Dict[str, ClusterNodeRec]:
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


def extract_protein_ids_from_text(text: str) -> List[str]:
    """
    Extract protein IDs from text for future internet search.
    
    Args:
        text: Text to search for protein identifiers
        
    Returns:
        List of unique protein IDs found in the text
    """
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


def normalize_cluster_id(cluster_id: str) -> str:
    """
    Normalize cluster ID to canonical format.
    
    Args:
        cluster_id: Input cluster ID in various formats (39184, CL_39184, CL:39184)
        
    Returns:
        Canonical cluster ID in format CL:39184
    """
    if cluster_id.isdigit():
        return f"CL:{cluster_id}"
    elif cluster_id.startswith("CL_"):
        return cluster_id.replace("CL_", "CL:", 1)
    elif cluster_id.startswith("CL:"):
        return cluster_id
    else:
        # Assume it's already in correct format or handle as-is
        return cluster_id


def get_cluster_stats(cluster_records: Dict[str, ClusterNodeRec]) -> Dict[str, any]:
    """
    Get summary statistics about the cluster dataset.
    
    Args:
        cluster_records: Dictionary of cluster records
        
    Returns:
        Dict with dataset statistics
    """
    total_clusters = len(cluster_records)
    total_proteins = sum(len(rec.proteins) for rec in cluster_records.values())
    non_empty_clusters = sum(1 for rec in cluster_records.values() if len(rec.proteins) > 0)
    
    protein_counts = [len(rec.proteins) for rec in cluster_records.values()]
    avg_proteins = sum(protein_counts) / len(protein_counts) if protein_counts else 0
    max_proteins = max(protein_counts) if protein_counts else 0
    
    # Count clusters with children
    clusters_with_children = sum(1 for rec in cluster_records.values() if rec.children)
    
    return {
        "total_clusters": total_clusters,
        "total_proteins": total_proteins,
        "non_empty_clusters": non_empty_clusters,
        "clusters_with_children": clusters_with_children,
        "avg_proteins_per_cluster": avg_proteins,
        "max_proteins_per_cluster": max_proteins,
        "empty_clusters": total_clusters - non_empty_clusters
    }


def validate_cluster_records(cluster_records: Dict[str, ClusterNodeRec]) -> bool:
    """
    Validate the integrity of cluster records.
    
    Args:
        cluster_records: Dictionary of cluster records to validate
        
    Returns:
        True if validation passes, False otherwise
    """
    print("🔍 Validating cluster records...")
    
    errors = []
    
    # Check for empty dataset
    if not cluster_records:
        errors.append("No cluster records found")
        
    # Check record structure
    for cid, rec in cluster_records.items():
        if not isinstance(rec, ClusterNodeRec):
            errors.append(f"Invalid record type for cluster {cid}")
            continue
            
        if not rec.cluster_id:
            errors.append(f"Missing cluster_id for cluster {cid}")
            
        if not rec.summary:
            errors.append(f"Missing summary for cluster {cid}")
            
        # Check protein structure
        for i, protein in enumerate(rec.proteins):
            if not isinstance(protein, dict):
                errors.append(f"Invalid protein format in cluster {cid}, protein {i}")
                continue
                
            required_keys = ["protein_id", "preferred_name", "size", "annotation"]
            missing_keys = [key for key in required_keys if key not in protein]
            if missing_keys:
                errors.append(f"Missing keys {missing_keys} in cluster {cid}, protein {i}")
    
    if errors:
        print("❌ Validation failed:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"   - {error}")
        if len(errors) > 10:
            print(f"   ... and {len(errors) - 10} more errors")
        return False
    
    print("✅ Cluster records validation passed")
    return True 