# BioRAG System - Advanced Protein Cluster Retrieval

A modular, scalable bio-RAG system that uses **recursive retrieval** to handle megacontext scenarios for protein cluster analysis. Built with LlamaIndex and designed to scale to millions of proteins without context overflow.

## 📺 Demo Video

<div align="center">
  <a href="https://youtu.be/Vn04JO0yn1Q">
    <img src="https://img.youtube.com/vi/Vn04JO0yn1Q/maxresdefault.jpg" alt="BioRAG Demo Video" width="600">
  </a>
</div>

## 🔄 Recursive Retrieval Architecture

The system implements a two-stage retrieval strategy:
1. **Stage 1**: Find the most semantically relevant CLUSTER SUMMARIES
2. **Stage 2**: Automatically fetch DETAILED PROTEINS only from those clusters

This prevents context overflow while maintaining comprehensive coverage.

## 📁 Project Structure

```
backend/
├── bio_rag/                    # Main package
│   ├── __init__.py            # Package exports
│   ├── config.py              # Configuration management
│   ├── data_parsers.py        # STRING database parsers
│   ├── graph_builder.py       # LlamaIndex setup
│   ├── internet_search.py     # Future enhancement hooks
│   ├── rag_system.py          # Main RAG logic
│   ├── utils.py               # Utility functions
│   └── cli.py                 # Interactive interface
├── main.py                    # Entry point
├── requirements.txt           # Dependencies
├── README.md                  # This file
└── 05.py                      # Original monolithic version
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up Environment

Create a `.env` file with your OpenAI API key:
```bash
OPENAI_API_KEY=your_api_key_here
```

### 3. Prepare Data Files
visit STRING to download data
https://string-db.org/cgi/download?sessionId=bRhB5OTfIjaY&species_text=Homo+sapiens
Place your STRING database files in the `/backend/data/` directory:
- `9606.clusters.info.v12.0.txt`
- `9606.clusters.tree.v12.0.txt`
- `9606.clusters.proteins.v12.0.txt`
- `9606.protein.info.v12.0.txt`

### 4. Run the System

**Interactive Mode:**
```bash
python main.py --interactive
```

**Single Query:**
```bash
python main.py --query "protein kinases involved in cell cycle"
```

**With Custom Settings:**
```bash
python main.py --interactive --max-clusters 200 --verbose
```

## 🔧 Configuration Options

### Command Line Arguments

- `--interactive`: Run interactive CLI mode
- `--query "text"`: Run a single query and exit
- `--config [default|mock]`: Choose configuration preset
- `--max-clusters N`: Maximum clusters to embed
- `--max-proteins N`: Maximum proteins per cluster
- `--debug`: Enable detailed debug output
- `--verbose`: Enable verbose logging
- `--internet-search`: Enable internet search (placeholder)

## 💡 Usage Examples

### Interactive Mode Commands

```bash
# Regular queries
Find protein kinases involved in cell cycle regulation
What are the main metabolic enzyme clusters?
Tell me about cluster CL:39184
Proteins involved in DNA repair mechanisms

# Special commands
debug    # Test retrieval system
info     # Show system statistics
help     # Show help message
quit     # Exit
```

## 🏗️ Architecture Details

### Key Components

1. **Data Parsers** (`data_parsers.py`)
   - Parse STRING database files
   - Build enriched cluster records
   - Handle protein metadata

2. **Graph Builder** (`graph_builder.py`)
   - Create LlamaIndex vector stores
   - Set up recursive retriever
   - Handle index persistence

3. **RAG System** (`rag_system.py`)
   - Coordinate query processing
   - Manage retrieval pipeline
   - Handle error cases

4. **Utilities** (`utils.py`)
   - Intelligent cluster sampling
   - Text processing functions
   - Validation utilities

### Scalability Features

- **Intelligent Sampling**: Uses 40% importance + 40% diversity + 20% random sampling
- **Protein Truncation**: Limits proteins per cluster to avoid token overflow
- **Index Caching**: Persists embeddings to disk with change detection
- **Memory Management**: Loads full dataset but only embeds sampled subset
