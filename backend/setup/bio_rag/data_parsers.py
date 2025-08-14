"""
Data parsing module for STRING database files.
Handles parsing of cluster information, tree structure, proteins, and metadata.
"""

from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Tuple


@dataclass
class ClusterNodeRec:
    """Record representing a protein cluster with its metadata and relationships."""
    cluster_id: str
    summary: str
    proteins: List[Dict]  # Changed from list[str] to list[dict] to include metadata
    children: List[str]


def parse_clusters_info(path: str) -> Dict[str, str]:
    """
    Parse the clusters.info file to get cluster descriptions.
    
    Format: 9606	CL:39184	7	mixed, incl. Protein FAM81, and Bacterial Ig-like domain 2
    
    Args:
        path: Path to the 9606.clusters.info.v12.0 file
        
    Returns:
        Dict mapping cluster_id to description
    """
    desc = {}
    try:
        with open(path, "r") as f:
            for line in f:
                if line.startswith("#"): 
                    continue
                    
                parts = line.rstrip("\n").split("\t")
                # Skip header rows
                if parts and (parts[0] == "string_taxon_id" or parts[1] == "cluster_id"):
                    continue
                    
                if len(parts) >= 4:
                    _, cluster_id, _, best_desc = parts
                    desc[cluster_id] = best_desc
                    
    except FileNotFoundError:
        print(f"❌ Clusters info file not found: {path}")
        raise
    except Exception as e:
        print(f"❌ Error parsing clusters info file {path}: {e}")
        raise
        
    print(f"✅ Parsed {len(desc)} cluster descriptions")
    return desc


def parse_clusters_tree(path: str) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Parse the clusters.tree file to get parent-child relationships.
    
    Format: 9606	CL:39184	CL:39182
    
    Args:
        path: Path to the 9606.clusters.tree.v12.0 file
        
    Returns:
        Tuple of (parent_to_children, child_to_parent) mappings
    """
    parent_to_children = defaultdict(list)
    child_to_parent = {}
    
    try:
        with open(path, "r") as f:
            for line in f:
                if line.startswith("#"): 
                    continue
                    
                parts = line.rstrip("\n").split("\t")
                # Skip header
                if parts and (parts[0] == "string_taxon_id" or parts[1] == "child_cluster_id"):
                    continue
                    
                if len(parts) >= 3:
                    _, child_id, parent_id = parts
                    parent_to_children[parent_id].append(child_id)
                    child_to_parent[child_id] = parent_id
                    
    except FileNotFoundError:
        print(f"❌ Clusters tree file not found: {path}")
        raise
    except Exception as e:
        print(f"❌ Error parsing clusters tree file {path}: {e}")
        raise
        
    print(f"✅ Parsed tree structure: {len(parent_to_children)} parents, {len(child_to_parent)} children")
    return parent_to_children, child_to_parent


def parse_clusters_proteins(path: str) -> Dict[str, List[str]]:
    """
    Parse the clusters.proteins file to get protein membership.
    
    Format: 9606	CL:39184	9606.ENSP00000473493
    
    Args:
        path: Path to the 9606.clusters.proteins.v12.0 file
        
    Returns:
        Dict mapping cluster_id to list of protein_ids
    """
    clust_to_prots = defaultdict(list)
    
    try:
        with open(path, "r") as f:
            for line in f:
                if line.startswith("#"): 
                    continue
                    
                parts = line.rstrip("\n").split("\t")
                # Skip header
                if parts and (parts[0] == "string_taxon_id" or parts[1] == "cluster_id"):
                    continue
                    
                if len(parts) >= 3:
                    _, clust_id, prot_id = parts
                    clust_to_prots[clust_id].append(prot_id)
                    
    except FileNotFoundError:
        print(f"❌ Clusters proteins file not found: {path}")
        raise
    except Exception as e:
        print(f"❌ Error parsing clusters proteins file {path}: {e}")
        raise
        
    total_proteins = sum(len(proteins) for proteins in clust_to_prots.values())
    print(f"✅ Parsed protein assignments: {len(clust_to_prots)} clusters, {total_proteins} proteins")
    return clust_to_prots


def parse_protein_info(path: str) -> Dict[str, Dict[str, str]]:
    """
    Parse the protein.info file to get protein metadata.
    
    Format: string_protein_id	preferred_name	protein_size	annotation
    
    Args:
        path: Path to the 9606.protein.info.v12.0 file
        
    Returns:
        Dict mapping protein_id to metadata dict
    """
    prot_meta = {}
    
    try:
        with open(path, "r") as f:
            for line in f:
                if line.startswith("#"): 
                    continue
                    
                parts = line.rstrip("\n").split("\t")
                # Skip header
                if parts and (parts[0] == "string_protein_id" or parts[0] == "string_taxon_id"):
                    continue
                    
                if len(parts) >= 3:
                    pid = parts[0]
                    pref = parts[1] if len(parts) > 1 else ""
                    size = parts[2] if len(parts) > 2 else ""
                    annot = parts[3] if len(parts) > 3 else ""
                    
                    prot_meta[pid] = {
                        "preferred_name": pref, 
                        "size": size, 
                        "annotation": annot
                    }
                    
    except FileNotFoundError:
        print(f"❌ Protein info file not found: {path}")
        raise
    except Exception as e:
        print(f"❌ Error parsing protein info file {path}: {e}")
        raise
        
    print(f"✅ Parsed metadata for {len(prot_meta)} proteins")
    return prot_meta


def build_cluster_records(paths: Dict[str, str]) -> Tuple[Dict[str, ClusterNodeRec], List[str]]:
    """
    Build complete cluster records by combining all parsed data.
    
    Args:
        paths: Dict with keys: clusters_info, clusters_tree, clusters_proteins, protein_info
        
    Returns:
        Tuple of (cluster_records_dict, root_cluster_ids)
    """
    print("📊 Building cluster records from STRING database files...")
    
    # Parse all data files
    desc = parse_clusters_info(paths["clusters_info"])
    p2c, c2p = parse_clusters_tree(paths["clusters_tree"])
    c2prots = parse_clusters_proteins(paths["clusters_proteins"])
    protein_meta = parse_protein_info(paths["protein_info"])

    # Find root clusters (clusters that never appear as children)
    all_ids = set(desc.keys())
    children = set(c2p.keys())
    roots = list(all_ids - children)
    
    print(f"📈 Found {len(roots)} root clusters out of {len(all_ids)} total clusters")
    
    # Assemble records with enriched protein information
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
    
    print(f"✅ Built {len(records)} complete cluster records")
    return records, roots


def gene_to_string_proteins(ensembl_gene_id: str) -> List[str]:
    """
    Convert Ensembl gene ID to STRING protein IDs.
    
    TODO: Implement actual gene DB API integration.
    For now, return empty list as placeholder.
    
    Args:
        ensembl_gene_id: Ensembl gene identifier
        
    Returns:
        List of STRING protein IDs for the gene
    """
    # TODO: Implement gene → protein mapping
    # e.g., use Ensembl → UniProt → STRING mapping
    return []  # placeholder 