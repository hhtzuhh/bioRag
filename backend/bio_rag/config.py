"""
Configuration module for the BioRAG system.
Centralizes all settings, file paths, and model configurations.
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional
from dotenv import load_dotenv


@dataclass
class BioRAGConfig:
    """Configuration class for the BioRAG system."""
    
    # Model settings
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-large"
    
    # Data file paths
    data_paths: Optional[Dict[str, str]] = None
    
    # Sampling parameters
    max_clusters: int = 8
    max_proteins_per_cluster: int = 10
    
    # Retrieval parameters
    similarity_top_k: int = 5
    cluster_similarity_top_k: int = 5
    
    # Storage settings
    persist_dir: str = "backend/storage"
    
    # Debug settings
    verbose: bool = True
    debug_mode: bool = False
    
    # Internet search settings (for future enhancement)
    enable_internet_search: bool = False
    max_internet_results: int = 5
    
    def __post_init__(self):
        """Initialize default data paths if not provided."""
        if self.data_paths is None:
            self.data_paths = {
                "clusters_info": "backend/data/9606.clusters.info.v12.0.txt",
                "clusters_tree": "backend/data/9606.clusters.tree.v12.0.txt", 
                "clusters_proteins": "backend/data/9606.clusters.proteins.v12.0.txt",
                "protein_info": "backend/data/9606.protein.info.v12.0.txt",
            }
    
    def validate_paths(self) -> bool:
        """Validate that all required data files exist."""
        missing_files = []
        for file_type, path in self.data_paths.items():
            if not os.path.exists(path):
                missing_files.append(f"{file_type}: {path}")
        
        if missing_files:
            print(f"❌ Missing required files:")
            for file in missing_files:
                print(f"   - {file}")
            return False
        
        print("✅ All required data files found")
        return True
    
    def setup_llama_index(self):
        """Initialize LlamaIndex with the configured models."""
        load_dotenv()
        
        try:
            from llama_index.core import Settings
            from llama_index.llms.openai import OpenAI
            from llama_index.embeddings.openai import OpenAIEmbedding
            
            Settings.llm = OpenAI(model=self.llm_model)
            Settings.embed_model = OpenAIEmbedding(model=self.embedding_model)
            
            if self.verbose:
                print(f"✅ LlamaIndex configured with:")
                print(f"   - LLM: {self.llm_model}")
                print(f"   - Embeddings: {self.embedding_model}")
                
        except ImportError as e:
            print(f"❌ Failed to import LlamaIndex components: {e}")
            raise
        except Exception as e:
            print(f"❌ Failed to configure LlamaIndex: {e}")
            raise


def get_default_config() -> BioRAGConfig:
    """Get a default configuration instance."""
    return BioRAGConfig()


def get_mock_data_config() -> BioRAGConfig:
    """Get a configuration for mock/test data."""
    config = BioRAGConfig()
    config.data_paths = {
        "clusters_info": "backend/mockdata1/mock_clusters_info.txt",
        "clusters_tree": "backend/mockdata1/mock_clusters_tree.txt",
        "clusters_proteins": "backend/mockdata1/mock_clusters_proteins.txt", 
        "protein_info": "backend/mockdata1/mock_protein_info.txt",
    }
    config.max_clusters = 100  # Smaller for testing
    config.max_proteins_per_cluster = 50
    return config 