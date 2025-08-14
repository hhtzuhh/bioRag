"""
Command line interface for the BioRAG system.
Provides interactive chat interface and debug commands.
"""

from typing import Dict
from .data_parsers import ClusterNodeRec
from .graph_builder import debug_recursive_retriever
from .rag_system import get_system_info
from .config import BioRAGConfig


def print_welcome_message():
    """Print the welcome message and usage instructions."""
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
    print("🔧 SPECIAL COMMANDS:")
    print("📋 'debug' - Run detailed debug analysis of recursive retriever")
    print("📋 'info' - Show system information and statistics")
    print("📋 'help' - Show this help message")
    print("")
    print("💡 System workflow:")
    print("   🔍 Query → Recursive Retrieval → Response")
    print("   🌐 Future: Query → Recursive Retrieval → Internet Search → Synthesis")
    print("="*80)


def print_help_message():
    """Print help information."""
    print("\n🔧 BIORAG SYSTEM HELP")
    print("="*50)
    print("QUERY COMMANDS:")
    print("  • Ask any question about proteins, clusters, or biological pathways")
    print("  • Examples: 'protein kinases', 'DNA repair', 'metabolic pathways'")
    print("")
    print("SPECIAL COMMANDS:")
    print("  • 'debug' - Test retrieval with a custom query")
    print("  • 'info' - Show system statistics and configuration")
    print("  • 'help' - Show this help message")
    print("  • 'quit' or 'exit' - End the session")
    print("")
    print("TIPS:")
    print("  • Be specific in your queries for better results")
    print("  • Mention cluster IDs (e.g., CL:39184) for direct lookups")
    print("  • Ask about biological processes, pathways, or protein functions")
    print("="*50)


def print_system_info(
    bio_rag_system, 
    cluster_records: Dict[str, ClusterNodeRec], 
    recursive_query_engine,
    config: BioRAGConfig
):
    """Print comprehensive system information."""
    info = get_system_info(bio_rag_system, cluster_records, recursive_query_engine, config)
    
    print("\n📊 SYSTEM INFORMATION")
    print("="*50)
    
    print("🔧 CONFIGURATION:")
    for key, value in info["config"].items():
        print(f"  • {key}: {value}")
    
    print("\n📈 DATASET STATISTICS:")
    for key, value in info["dataset"].items():
        print(f"  • {key}: {value:,}" if isinstance(value, (int, float)) else f"  • {key}: {value}")
    
    print("\n🔍 RETRIEVER STATISTICS:")
    for key, value in info["retriever"].items():
        if key == "sample_node_types":
            print(f"  • {key}:")
            for node_type, count in value.items():
                print(f"    - {node_type}: {count}")
        else:
            print(f"  • {key}: {value}")
    
    print("="*50)


def run_interactive_cli(
    bio_rag_system, 
    cluster_records: Dict[str, ClusterNodeRec], 
    recursive_query_engine,
    config: BioRAGConfig
):
    """
    Run the interactive command line interface.
    
    Args:
        bio_rag_system: The main query function
        cluster_records: Full cluster records for debugging
        recursive_query_engine: The recursive query engine
        config: Configuration object
    """
    print_welcome_message()
    
    # Interactive chat loop with ultra-simple pipeline
    while True:
        try:
            user_input = input("\n🧬 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye! Thanks for testing the bio-RAG system!")
                break
                
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() == 'debug':
                debug_recursive_retriever(recursive_query_engine, cluster_records)
                continue
                
            if user_input.lower() == 'info':
                print_system_info(bio_rag_system, cluster_records, recursive_query_engine, config)
                continue
                
            if user_input.lower() == 'help':
                print_help_message()
                continue
            
            # Process regular queries
            print("\n⚡ Processing...")
            response = bio_rag_system(user_input, use_internet_search=config.enable_internet_search)
            print(f"\n🤖 Response:\n{response}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.\n")


def run_single_query(
    query: str,
    bio_rag_system, 
    config: BioRAGConfig,
    use_internet_search: bool = None,
    debug: bool = None
) -> str:
    """
    Run a single query without the interactive interface.
    
    Args:
        query: The query to process
        bio_rag_system: The main query function
        config: Configuration object
        use_internet_search: Override internet search setting
        debug: Override debug setting
        
    Returns:
        Query response
    """
    if use_internet_search is None:
        use_internet_search = config.enable_internet_search
    if debug is None:
        debug = config.debug_mode
        
    return bio_rag_system(query, use_internet_search=use_internet_search, debug=debug)


def print_initialization_summary(config: BioRAGConfig):
    """Print a summary after system initialization."""
    print("✅ Ultra-simple linear architecture established")
    print("✅ Recursive retrieval handles cluster→protein navigation")
    print("✅ No ReAct overhead - direct function calls")
    print("✅ Ready for future internet search enhancement")
    print("✅ System scales to millions of proteins efficiently")
    
    if config.enable_internet_search:
        print("🌐 Internet search enhancement: ENABLED")
    else:
        print("🔍 Internet search enhancement: DISABLED (placeholder only)")
        
    if config.debug_mode:
        print("🔧 Debug mode: ENABLED")
    else:
        print("🔧 Debug mode: DISABLED") 