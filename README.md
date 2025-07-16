# Cisco IOS RAG System - Complete Setup Guide

## Overview

This is a comprehensive end-to-end Retrieval-Augmented Generation (RAG) system specifically designed for Cisco IOS documentation. The system features:

- **Advanced Document Chunking**: Multiple strategies including context-aware, command-aware, token-based, and hybrid chunking
- **Knowledge Libraries**: Pre-built error pattern and best practices libraries
- **State Management**: Full recovery support with checkpointing
- **7.5K Memory Context**: Configured for Ollama models
- **Streamlit Interface**: User-friendly web interface

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit Web Interface                │
├─────────────────────────────────────────────────────────┤
│                    State Management                      │
│                 (Checkpoints & Recovery)                 │
├─────────────────────────────────────────────────────────┤
│                   Document Processing                    │
│     ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│     │   PDF    │→ │ Chunking │→ │ Embedding│          │
│     │ Extract  │  │ Strategies│  │          │          │
│     └──────────┘  └──────────┘  └──────────┘          │
├─────────────────────────────────────────────────────────┤
│                  Knowledge Libraries                     │
│     ┌──────────────┐        ┌─────────────────┐       │
│     │Error Patterns│        │ Best Practices  │       │
│     └──────────────┘        └─────────────────┘       │
├─────────────────────────────────────────────────────────┤
│                    Vector Database                       │
│                     (ChromaDB)                          │
├─────────────────────────────────────────────────────────┤
│                   RAG Query Engine                      │
│              (Ollama LLM + Embeddings)                  │
└─────────────────────────────────────────────────────────┘
```

## Prerequisites

- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- 10GB free disk space
- Internet connection for model downloads

## Installation Guide

### Step 1: Clone or Create Project Directory

```bash
mkdir cisco-rag-system
cd cisco-rag-system
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install Ollama

#### macOS
```bash
brew install ollama
```

#### Linux
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

#### Windows
Download from: https://ollama.ai/download/windows

### Step 5: Start Ollama Service

```bash
# In a separate terminal
ollama serve
```

### Step 6: Pull Required Model

```bash
# Recommended model with 7.5K context
ollama pull llama2

# Alternative models
ollama pull mistral
ollama pull neural-chat
```

## Running the Application

### Step 1: Save the Code

Save the main application code as `app.py` in your project directory.

### Step 2: Launch Streamlit

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Usage Guide

### 1. Initial Setup

1. **Check Ollama Status**: The sidebar will show if Ollama is installed and running
2. **Select Model**: Choose your preferred model (llama2 recommended)
3. **Configure Chunking**: Set your preferred chunking strategy and parameters

### 2. Processing PDF Documents

1. **Upload PDF**: 
   - Click "Upload & Process" tab
   - Upload your Cisco IOS documentation PDF
   - The system supports the official Cisco command reference

2. **Extract Text**:
   - Click "Extract Text" button
   - System will use multiple extraction methods for best results

3. **Create Chunks**:
   - Click "Create Chunks" button
   - Monitor chunk statistics

4. **Store in VectorDB**:
   - Click "Store in VectorDB" button
   - Chunks will be embedded and stored in ChromaDB

### 3. Loading Knowledge Libraries

Navigate to "Knowledge Libraries" tab:

1. **Error Pattern Library**:
   - Click "Load Error Pattern Library"
   - Contains 10 common Cisco IOS errors with solutions

2. **Best Practices Library**:
   - Click "Load Best Practices Library"
   - Contains 10 configuration best practices

### 4. Querying the System

1. Navigate to "Query System" tab
2. Enter your question about Cisco IOS
3. Click "Search" to get AI-powered answers with sources

Example queries:
- "How do I configure VLANs?"
- "What causes Invalid input detected error?"
- "Show me password security best practices"

### 5. State Management & Recovery

The system automatically saves state after each major operation.

**To recover from failure**:
1. Check sidebar for saved checkpoints
2. Select desired checkpoint
3. Click "Load Selected Checkpoint"
4. Continue from where you left off

**Manual state management**:
- Click "Save Current State" to create manual checkpoint
- States are saved in `./state` directory

## Advanced Features

### Chunking Strategies

1. **Context-Aware**: Preserves semantic boundaries (sections, chapters)
2. **Command-Aware**: Specifically extracts command blocks
3. **Token-Based**: Optimized for embedding models
4. **Hybrid**: Combines all strategies for comprehensive coverage

### Error Recovery

The system includes comprehensive error handling:
- Automatic state checkpointing
- Step-by-step failure recovery
- Processing history tracking
- Checkpoint management (keeps last 10)

### Memory Configuration

The system is configured for 7.5K token context:
```python
llm = Ollama(
    model=model_name,
    temperature=0.1,
    num_ctx=7500  # 7.5K context window
)
```

## Troubleshooting

### Common Issues

1. **Ollama not found**:
   - Ensure Ollama is installed and running
   - Check if `ollama serve` is running in background

2. **Model download fails**:
   - Check internet connection
   - Ensure sufficient disk space (models are 4-7GB)

3. **PDF extraction errors**:
   - Try different extraction methods
   - Ensure PDF is not encrypted or corrupted

4. **Memory errors**:
   - Reduce chunk size
   - Process smaller sections of PDF
   - Increase system RAM

### Debug Mode

Enable detailed logging:
```python
logging.basicConfig(level=logging.DEBUG)
```

## File Structure

```
cisco-rag-system/
├── app.py                 # Main application
├── requirements.txt       # Python dependencies
├── state/                # State management
│   ├── processing_state.json
│   └── checkpoints/
├── chroma_db/           # Vector database storage
├── temp_*.pdf           # Temporary PDF files
└── README.md           # This file
```

## Performance Optimization

1. **Chunk Size**: Start with 1000 characters, adjust based on results
2. **Overlap**: 200 characters recommended for context preservation
3. **Batch Processing**: System processes in batches to manage memory
4. **Caching**: ChromaDB persists embeddings for reuse

## Security Considerations

1. **Local Processing**: All data processed locally
2. **No External APIs**: Uses local Ollama instance
3. **State Files**: Contains processing metadata only
4. **Cleanup**: Remove temp files after processing

## Extending the System

### Adding New Error Patterns

Edit `KnowledgeLibraryBuilder.create_error_pattern_library()`:
```python
{
    "error": "Your error message",
    "cause": "Root cause",
    "solution": "How to fix",
    "example": "Configuration example"
}
```

### Adding New Best Practices

Edit `KnowledgeLibraryBuilder.create_best_practices_library()`:
```python
{
    "topic": "Practice topic",
    "practice": "Description",
    "implementation": "Config commands"
}
```

## Support

For issues:
1. Check the troubleshooting section
2. Review processing history in Analytics tab
3. Check state files for recovery options
4. Ensure all prerequisites are met

## License

This system is provided as-is for educational and professional use with Cisco documentation. 

### Output

The application generates a unique, timestamped folder for each run in the `OUTPUT-FOLDER` directory. This folder contains the generated knowledge libraries as structured JSON files, a copy of the fine-tuning guide, and a processing log.

For a detailed explanation of the output files and the application's complete workflow, please see `A_DETAIL_OUTPUT.txt`. 