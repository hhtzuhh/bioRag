# BioRAG System - Advanced Protein Cluster Retrieval

A modular, scalable bio-RAG system that uses **recursive retrieval** to handle megacontext scenarios for protein cluster analysis. Built with LlamaIndex and designed to scale to millions of proteins without context overflow.

## 🔄 Recursive Retrieval Architecture

The system implements a two-stage retrieval strategy:
1. **Stage 1**: Find the most semantically relevant CLUSTER SUMMARIES
2. **Stage 2**: Automatically fetch DETAILED PROTEINS only from those clusters

This prevents context overflow while maintaining comprehensive coverage.

## 📁 Project Structure

```
backend/setup/
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

Place your STRING database files in the `data/` directory:
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

### Configuration Presets

**Default Configuration:**
- Uses real STRING database files
- Embeds 500 clusters max
- 200 proteins per cluster max
- Production settings

**Mock Configuration:**
- Uses smaller test files
- Embeds 100 clusters max
- 50 proteins per cluster max
- Development/testing settings

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

### Programmatic Usage

```python
from bio_rag import BioRAGConfig, get_default_config
from bio_rag.rag_system import init_pipeline

# Initialize system
config = get_default_config()
bio_rag_system, cluster_records, query_engine = init_pipeline(config)

# Query the system
response = bio_rag_system("protein kinases in cancer")
print(response)
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

## 🔍 Debug Features

### Debug Mode
Enable with `--debug` flag to see:
- Retriever structure analysis
- Node relationship mapping
- Detailed error traces
- Query processing steps

### Debug Commands
- `debug`: Interactive retrieval testing
- `info`: System statistics and configuration

## 🌐 Future Enhancements

The system includes hooks for future internet search integration:

- **UniProt API**: Protein function and structure
- **PubMed API**: Recent publications
- **STRING API**: Protein interactions
- **Reactome API**: Biological pathways
- **PDB API**: 3D structures

## 📊 Performance Characteristics

- **Embeddings**: Only for sampled clusters (default: 500)
- **Memory**: Full dataset in RAM for fast lookups
- **Retrieval**: O(log n) for cluster selection + O(k) for proteins
- **Scaling**: Linear with cluster count, not protein count

## 🛠️ Development

### Adding New Features

1. **New Data Sources**: Extend `data_parsers.py`
2. **Custom Retrievers**: Modify `graph_builder.py` 
3. **Search Enhancement**: Implement in `internet_search.py`
4. **UI Changes**: Update `cli.py`

### Testing with Mock Data

```bash
python main.py --config mock --interactive
```

### Configuration Customization

```python
from bio_rag import BioRAGConfig

config = BioRAGConfig(
    llm_model="gpt-4",
    max_clusters=1000,
    max_proteins_per_cluster=100,
    similarity_top_k=10
)
```

## 📈 System Statistics

The system provides detailed statistics:
- Total clusters and proteins
- Sampling distribution
- Retriever configuration
- Performance metrics

Access via the `info` command in interactive mode.

## ⚠️ Troubleshooting

### Common Issues

1. **Missing Data Files**: Ensure STRING database files are in correct location
2. **API Key Issues**: Check `.env` file and OpenAI API key
3. **Memory Issues**: Reduce `max_clusters` or `max_proteins_per_cluster`
4. **Slow Performance**: Enable index caching (automatic)

### Error Handling

The system includes comprehensive error handling:
- File validation before processing
- Data integrity checks
- Graceful fallbacks for missing data
- Detailed error messages in debug mode 