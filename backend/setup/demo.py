#!/usr/bin/env python3
"""
Demo script for the modularized BioRAG system.
Shows how to use the system programmatically.
"""

import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from bio_rag import BioRAGConfig, get_default_config, get_mock_data_config
from bio_rag.rag_system import init_pipeline, get_system_info


def demo_basic_usage():
    """Demonstrate basic usage of the BioRAG system."""
    print("🧬 BioRAG System Demo")
    print("=" * 50)
    
    # Option 1: Use default configuration
    print("\n1️⃣ Creating default configuration...")
    config = get_default_config()
    
    # Option 2: Use mock configuration for testing
    print("2️⃣ Using mock configuration for demo...")
    config = get_mock_data_config()
    
    # Option 3: Custom configuration
    print("3️⃣ Creating custom configuration...")
    config = BioRAGConfig(
        llm_model="gpt-4o-mini",
        max_clusters=50,  # Small for demo
        max_proteins_per_cluster=10,
        verbose=True,
        debug_mode=False
    )
    
    # Set mock data paths
    config.data_paths = {
        "clusters_info": "mockdata1/mock_clusters_info.txt",
        "clusters_tree": "mockdata1/mock_clusters_tree.txt",
        "clusters_proteins": "mockdata1/mock_clusters_proteins.txt",
        "protein_info": "mockdata1/mock_protein_info.txt",
    }
    
    print(f"Configuration created:")
    print(f"  - Model: {config.llm_model}")
    print(f"  - Max clusters: {config.max_clusters}")
    print(f"  - Max proteins per cluster: {config.max_proteins_per_cluster}")
    
    try:
        # Initialize the system
        print("\n🚀 Initializing BioRAG system...")
        bio_rag_system, cluster_records, recursive_query_engine = init_pipeline(config)
        
        # Show system information
        print("\n📊 System Information:")
        system_info = get_system_info(bio_rag_system, cluster_records, recursive_query_engine, config)
        
        dataset_stats = system_info["dataset"]
        print(f"  - Total clusters: {dataset_stats['total_clusters']:,}")
        print(f"  - Total proteins: {dataset_stats['total_proteins']:,}")
        print(f"  - Non-empty clusters: {dataset_stats['non_empty_clusters']:,}")
        
        # Run sample queries
        print("\n🔍 Running sample queries...")
        
        queries = [
            "protein kinases",
            "metabolic enzymes", 
            "DNA repair proteins"
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\n🔬 Query {i}: '{query}'")
            print("-" * 40)
            
            try:
                response = bio_rag_system(query)
                # Truncate response for demo
                if len(response) > 200:
                    response = response[:200] + "..."
                print(f"Response: {response}")
                
            except Exception as e:
                print(f"❌ Error processing query: {e}")
        
        print("\n✅ Demo completed successfully!")
        return bio_rag_system, cluster_records, recursive_query_engine
        
    except FileNotFoundError:
        print("\n❌ Mock data files not found. This demo requires mock data files.")
        print("To run with real data, ensure STRING database files are available.")
        return None, None, None
        
    except Exception as e:
        print(f"\n❌ Error during initialization: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def demo_configuration_options():
    """Demonstrate different configuration options."""
    print("\n🔧 Configuration Options Demo")
    print("=" * 50)
    
    # Show different ways to configure the system
    configs = {
        "Production": get_default_config(),
        "Development": get_mock_data_config(),
        "Custom": BioRAGConfig(
            llm_model="gpt-4",
            embedding_model="text-embedding-3-small",
            max_clusters=100,
            max_proteins_per_cluster=20,
            similarity_top_k=3,
            verbose=False,
            debug_mode=True
        )
    }
    
    for name, config in configs.items():
        print(f"\n{name} Configuration:")
        print(f"  - LLM Model: {config.llm_model}")
        print(f"  - Embedding Model: {config.embedding_model}")
        print(f"  - Max Clusters: {config.max_clusters}")
        print(f"  - Max Proteins/Cluster: {config.max_proteins_per_cluster}")
        print(f"  - Similarity Top-K: {config.similarity_top_k}")
        print(f"  - Verbose: {config.verbose}")
        print(f"  - Debug Mode: {config.debug_mode}")


def demo_error_handling():
    """Demonstrate error handling capabilities."""
    print("\n⚠️ Error Handling Demo")
    print("=" * 50)
    
    # Test with invalid configuration
    print("\n1. Testing with missing data files...")
    config = BioRAGConfig()
    config.data_paths = {
        "clusters_info": "nonexistent_file.txt",
        "clusters_tree": "nonexistent_file.txt",
        "clusters_proteins": "nonexistent_file.txt",
        "protein_info": "nonexistent_file.txt",
    }
    
    try:
        if not config.validate_paths():
            print("✅ Configuration validation correctly detected missing files")
        else:
            print("❌ Configuration validation failed to detect missing files")
    except Exception as e:
        print(f"✅ Error handling working: {e}")


def main():
    """Main demo function."""
    print("🧬 BIORAG SYSTEM - MODULAR DEMO")
    print("=" * 80)
    print("This demo shows the key features of the modularized BioRAG system.")
    print()
    
    # Demo 1: Basic usage
    system_components = demo_basic_usage()
    
    # Demo 2: Configuration options
    demo_configuration_options()
    
    # Demo 3: Error handling
    demo_error_handling()
    
    print("\n" + "=" * 80)
    print("🎉 Demo completed! The system is now fully modularized and ready for use.")
    print("\nNext steps:")
    print("  1. Run 'python main.py --interactive' for interactive mode")
    print("  2. Run 'python main.py --query \"your query\"' for single queries")
    print("  3. Import bio_rag modules in your own code")
    print("  4. Check README.md for comprehensive documentation")
    

if __name__ == "__main__":
    main() 