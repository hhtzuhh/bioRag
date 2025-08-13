# requirements:
# pip install llama-index==0.10.52 llama-index-embeddings-openai llama-index-llms-openai faiss-cpu python-dotenv

import os
from dataclasses import dataclass
from collections import defaultdict
from dotenv import load_dotenv

from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings, get_response_synthesizer
from llama_index.core.indices.composability import ComposableGraph
from llama_index.core.schema import TextNode, MetadataMode
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import load_index_from_storage

# ---------- setup ----------
load_dotenv()
Settings.llm = OpenAI(model="gpt-4o-mini")  # change if needed
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")

# ---------- data adapters (STRING) ----------
@dataclass
class ClusterNodeRec:
    cluster_id: str
    summary: str
    proteins: list[str]
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
    desc = parse_clusters_info(paths["clusters_info"])
    p2c, c2p = parse_clusters_tree(paths["clusters_tree"])
    c2prots = parse_clusters_proteins(paths["clusters_proteins"])
    print(f"desc: {desc}")
    print(f"p2c: {p2c}")
    print(f"c2p: {c2p}")
    print(f"c2prots: {c2prots}")
    # roots = clusters that never appear as child
    all_ids = set(desc.keys())
    children = set(c2p.keys())
    roots = list(all_ids - children)
    # assemble records
    records = {}
    for cid in all_ids:
        records[cid] = ClusterNodeRec(
            cluster_id=cid,
            summary=desc.get(cid, "No description"),
            proteins=c2prots.get(cid, []),
            children=p2c.get(cid, [])
        )
    return records, roots

# ---------- index builders ----------
def make_cluster_docs(rec: ClusterNodeRec, protein_meta: dict) -> list[Document]:
    """
    Create:
      - one 'cluster summary' doc
      - N 'protein' docs under this cluster
    """
    docs = []
    # cluster summary document
    cluster_text = f"""Cluster: {rec.cluster_id}
                        Summary: {rec.summary}
                        Children: {', '.join(rec.children) if rec.children else 'None'}
                        Protein count: {len(rec.proteins)}
                        """
    d = Document(text=cluster_text, metadata={"node_type": "cluster", "cluster_id": rec.cluster_id})
    docs.append(d)

    # child docs (proteins) — chunkable content
    for pid in rec.proteins:
        meta = protein_meta.get(pid, {})
        ptext = f"""Protein: {pid}
                    Name: {meta.get('preferred_name','NA')}
                    Annotation: {meta.get('annotation','NA')}
                    Cluster: {rec.cluster_id}
                    """
        docs.append(Document(text=ptext, metadata={"node_type": "protein", "cluster_id": rec.cluster_id, "protein_id": pid}))
    return docs

def build_graph_indexes(cluster_records: dict[str, ClusterNodeRec], roots: list[str], protein_meta: dict):
    """
    Build:
      - A VectorStoreIndex per cluster (children content)
      - A ComposableGraph whose root can route queries down to child cluster indexes
    """
    cluster_id_to_index = {}
    cluster_id_to_engine_tool = {}

    # optional on-disk persistence to avoid re-embedding costs
    persist_root = os.getenv("LLAMA_PERSIST_DIR", "")
    if persist_root:
        os.makedirs(persist_root, exist_ok=True)

    # Collect all protein docs while building per-cluster indexes
    all_protein_docs = []

    # build or load vector indexes per cluster
    for cid, rec in cluster_records.items():
        docs = make_cluster_docs(rec, protein_meta)
        # print(f"docs: {docs}")
        persist_dir = os.path.join(persist_root, f"cluster_{cid}") if persist_root else None
        idx = None

        # Load only if a previously persisted index looks complete
        if persist_dir and os.path.isdir(persist_dir):
            # Accept either .json or .jsonl docstore; vector_store may be optional depending on backend
            docstore_candidates = [
                os.path.join(persist_dir, "docstore.json"),
                os.path.join(persist_dir, "docstore.jsonl"),
            ]
            index_store_path = os.path.join(persist_dir, "index_store.json")
            has_docstore = any(os.path.exists(p) for p in docstore_candidates)
            has_index_store = os.path.exists(index_store_path)
            if has_docstore and has_index_store:
                try:
                    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
                    idx = load_index_from_storage(storage_context)
                    print(f"Loaded persisted index for cluster {cid} from {persist_dir}")
                except Exception:
                    idx = None

        if idx is None:
            # Build fresh, then persist (without trying to read non-existent files)
            idx = VectorStoreIndex.from_documents(docs)
            if persist_dir:
                try:
                    os.makedirs(persist_dir, exist_ok=True)
                    idx.storage_context.persist(persist_dir=persist_dir)
                    print(f"Built and persisted index for cluster {cid} at {persist_dir}")
                except Exception:
                    pass

        cluster_id_to_index[cid] = idx
        all_protein_docs.extend([d for d in docs if d.metadata.get("node_type") == "protein"])

        # expose per-cluster query tool
        qe = idx.as_query_engine(similarity_top_k=5, response_mode="tree_summarize")
        cluster_id_to_engine_tool[f"cluster_{cid}"] = QueryEngineTool(
            query_engine=qe,
            metadata=ToolMetadata(
                name=f"cluster_{cid}",
                description=f"Query only cluster {cid}: {rec.summary} (children: {len(rec.children)}, proteins: {len(rec.proteins)})."
            ),
        )
    print(f"all_protein_docs: {all_protein_docs}")
    # Build a global protein index (+ optional persistence), expose as a tool
    protein_persist_dir = os.path.join(persist_root, "protein_index") if persist_root else None
    protein_index = None
    if protein_persist_dir and os.path.isdir(protein_persist_dir):
        try:
            sc = StorageContext.from_defaults(persist_dir=protein_persist_dir)
            protein_index = load_index_from_storage(sc)
            print(f"Loaded persisted global protein index from {protein_persist_dir}")
        except Exception:
            protein_index = None

    if protein_index is None:
        protein_index = VectorStoreIndex.from_documents(all_protein_docs)
        if protein_persist_dir:
            try:
                os.makedirs(protein_persist_dir, exist_ok=True)
                protein_index.storage_context.persist(persist_dir=protein_persist_dir)
                print(f"Built and persisted global protein index at {protein_persist_dir}")
            except Exception:
                pass

    protein_qe = protein_index.as_query_engine(similarity_top_k=5, response_mode="tree_summarize")
    cluster_id_to_engine_tool["protein_router"] = QueryEngineTool(
        query_engine=protein_qe,
        metadata=ToolMetadata(
            name="protein_router",
            description="Find proteins by ID/name/annotation across all clusters; returned nodes include cluster_id in metadata."
        ),
    )

    # Build flat star: one root over all cluster indexes
    children_indices = []
    index_summaries = []
    for cid, idx in cluster_id_to_index.items():
        children_indices.append(idx)
        index_summaries.append(f"{cid}: {cluster_records[cid].summary}")

    graph = ComposableGraph.from_indices(
        root_index_cls=VectorStoreIndex,
        children_indices=children_indices,
        index_summaries=index_summaries,
        response_synthesizer=get_response_synthesizer(response_mode="tree_summarize"),
    )

    root_engine = graph.as_query_engine(similarity_top_k=5, response_mode="tree_summarize")
    return graph, root_engine, cluster_id_to_engine_tool

# ---------- the agent ----------
from llama_index.core.agent import ReActAgent

def build_agent(root_engine, cluster_toolmap: dict[str, QueryEngineTool]):
    """
    Agent with tools:
      - root_engine (broad retrieval)
      - per-cluster engines (fine retrieval)
      - gene→protein resolver tool
    """
    print(f"cluster_toolmap: {cluster_toolmap}")
    # tool: global/root
    tools = [
        QueryEngineTool(
            query_engine=root_engine,
            metadata=ToolMetadata(
                name="root_router",
                description="Routes queries across top-level clusters and returns cluster-level summaries."
            ),
        )
    ] + list(cluster_toolmap.values())

    # simple python function tool for gene→protein mapping
    from llama_index.core.tools import FunctionTool
    gene_tool = FunctionTool.from_defaults(
        fn=gene_to_string_proteins,
        name="gene_to_string_proteins",
        description="Given an Ensembl gene ID, return associated STRING protein IDs."
    )
    tools.append(gene_tool)

    sys_prompt = (
        "You are a bio RAG agent. Given a gene or question, you:\n"
        "1) resolve gene → proteins (using the tool),\n"
        "2) search relevant clusters (root_router, then specific clusters),\n"
        "3) synthesize a concise answer with key proteins, interactions, and hypotheses.\n"
        "Prefer citing cluster IDs and protein preferred_names where available.\n"
        "Prefer: if the query mentions a protein, call protein_router first to identify cluster_id(s), then query specific cluster tool(s)."
    )
    agent = ReActAgent.from_tools(tools, system_prompt=sys_prompt, verbose=True)
    return agent

# ---------- end-to-end init ----------
def init_pipeline(paths: dict):
    cluster_records, roots = build_cluster_records(paths)
    protein_meta = parse_protein_info(paths["protein_info"])
    graph, root_engine, toolmap = build_graph_indexes(cluster_records, roots, protein_meta)
    agent = build_agent(root_engine, toolmap)
    return agent

# ---------- demo ----------
if __name__ == "__main__":
    # TODO: set paths to your real extracted files
    # paths = {
    #     "clusters_info": "9606.clusters.info.v12.0.txt",
    #     "clusters_tree": "9606.clusters.tree.v12.0.txt",
    #     "clusters_proteins": "9606.clusters.proteins.v12.0.txt",
    #     "protein_info": "9606.protein.info.v12.0.txt",
    # }
    paths = {
        "clusters_info": "mockdata/mock_clusters_info.txt",
        "clusters_tree": "mockdata/mock_clusters_tree.txt",
        "clusters_proteins": "mockdata/mock_clusters_proteins.txt",
        "protein_info": "mockdata/mock_protein_info.txt",
    }
    
    print("Initializing bio RAG agent...")
    agent = init_pipeline(paths)
    print("Agent ready! Type 'quit' or 'exit' to end the chat.\n")
    
    # # Example 1: free-text (no gene)
    # q1 = "List key protein families related to phosphorylation and MAPK signaling; highlight notable human proteins."
    # print(agent.chat(q1))

    # # Example 2: with a gene ID
    # q2 = "Given ENSG00000012048, explain likely pathways and key interacting proteins to investigate."
    # print(agent.chat(q2))

    # Interactive chat loop
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            if not user_input:
                continue
                
            print("\nAgent: Thinking...")
            response = agent.chat(user_input)
            print(f"Agent: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.\n")
