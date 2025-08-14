#!/usr/bin/env python3
"""
Main entry point for the BioRAG system.
Demonstrates usage of the modularized bio-RAG components.

Requirements:
pip install llama-index==0.10.52 llama-index-embeddings-openai llama-index-llms-openai faiss-cpu python-dotenv

Usage:
    python main.py --interactive    # Run interactive CLI
    python main.py --query "protein kinases"  # Run single query
    python main.py --config mock   # Use mock data configuration
"""

import argparse
import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from bio_rag import (
    BioRAGConfig, 
    get_default_config, 
    get_mock_data_config
)
from bio_rag.rag_system import init_pipeline
from bio_rag.cli import (
    run_interactive_cli, 
    run_single_query, 
    print_initialization_summary
)


def setup_argument_parser():
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="BioRAG System - Advanced Protein Cluster Retrieval with Recursive Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --interactive                    # Interactive mode
  python main.py --query "protein kinases"        # Single query
  python main.py --config mock --interactive      # Use mock data
  python main.py --query "DNA repair" --debug     # Single query with debug
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--interactive', 
        action='store_true',
        help='Run interactive CLI mode'
    )
    mode_group.add_argument(
        '--query', 
        type=str,
        help='Run a single query and exit'
    )
    
    # Configuration options
    parser.add_argument(
        '--config',
        choices=['default', 'mock'],
        default='default',
        help='Configuration to use (default: %(default)s)'
    )
    
    # Override settings
    parser.add_argument(
        '--max-clusters',
        type=int,
        help='Maximum clusters to embed (overrides config)'
    )
    parser.add_argument(
        '--max-proteins',
        type=int,
        help='Maximum proteins per cluster (overrides config)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--internet-search',
        action='store_true',
        help='Enable internet search (currently placeholder)'
    )
    
    return parser


def create_config(args) -> BioRAGConfig:
    """Create configuration based on command line arguments."""
    # Get base configuration
    if args.config == 'mock':
        config = get_mock_data_config()
    else:
        config = get_default_config()
    
    # Apply command line overrides
    if args.max_clusters:
        config.max_clusters = args.max_clusters
    if args.max_proteins:
        config.max_proteins_per_cluster = args.max_proteins
    if args.debug:
        config.debug_mode = True
    if args.verbose:
        config.verbose = True
    if args.internet_search:
        config.enable_internet_search = True
        
    return config


def main():
    """Main entry point."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Create configuration
    config = create_config(args)
    
    try:
        print("Initializing ULTRA-SIMPLE bio RAG system with recursive retrieval...")
        
        # Initialize the pipeline
        bio_rag_system, cluster_records, recursive_query_engine = init_pipeline(config)
        
        print_initialization_summary(config)
        
        # Run based on mode
        if args.interactive:
            run_interactive_cli(bio_rag_system, cluster_records, recursive_query_engine, config)
        else:
            # Single query mode
            response = run_single_query(
                args.query, 
                bio_rag_system, 
                config, 
                debug=args.debug
            )
            print(f"\n🤖 Response:\n{response}")
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        print("Make sure your data files are in the correct location.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error initializing system: {e}")
        if args.debug or args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()