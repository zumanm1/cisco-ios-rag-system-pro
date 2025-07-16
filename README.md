# 🔧 Cisco IOS RAG System Pro v2.0

An **enterprise-grade** tool that automatically discovers, downloads, and processes Cisco documentation to generate structured knowledge libraries for fine-tuning Large Language Models (LLMs) in network engineering contexts.

## 🚀 New Features in v2.0

### 🔍 **Automated Document Discovery**
- **Web Crawler**: Automatically finds Cisco documentation for 16+ networking topics
- **CCIE Book Discovery**: Locates study materials for Service Provider (10 books), Enterprise Infrastructure (12 books), and Security (9 books)
- **Multi-Platform Support**: Covers both IOS and IOS-XR documentation
- **Smart Filtering**: Targets relevant PDFs based on topic keywords

### 🔄 **Enhanced Batch Processing**
- **Multi-Document Processing**: Process dozens of documents simultaneously
- **Cross-Document Synthesis**: Creates enhanced knowledge by correlating information across multiple sources
- **Advanced Knowledge Libraries**: 
  - Error Patterns Library
  - Best Practices Library  
  - Command Syntax Library
  - Troubleshooting Scenarios Library
  - **NEW**: Cross-Reference Knowledge Library

### 🌐 **Comprehensive Topic Coverage**
#### Routing Protocols
- IPv4, IGP OSPF, BGP, MP-BGP, Route Reflector

#### MPLS & VPN Technologies  
- MPLS LDP, L2VPN, L3VPN

#### Switching & QoS
- Layer 2 Switching, Quality of Service

#### Services & Management
- Services, FTP, SSH, TFTP, SNMP, AAA, NetFlow

## 🛠️ Features

### **Three-Tab Interface**
1. **🔍 Document Discovery**: Find and download Cisco documentation automatically
2. **🏗️ Build Knowledgebase**: Process documents (single or batch) into structured libraries
3. **🚀 Fine-Tuning Guide**: Hardware-specific recommendations for model training

### **Advanced Processing Capabilities**
- **Multi-threaded Processing**: Real-time progress tracking with system monitoring
- **Intelligent Text Extraction**: Multiple PDF processing methods for maximum coverage
- **Vector Database Integration**: Uses Chroma for semantic search and knowledge retrieval
- **Output Management**: Organized folder structure with timestamped sessions

### **Enterprise Features**
- **Comprehensive Logging**: Full audit trail of processing activities
- **Error Handling**: Robust error recovery and detailed error reporting
- **Resource Monitoring**: Real-time CPU and memory usage tracking
- **Export Capabilities**: JSON reports and structured data export

## 📋 Prerequisites

- **Python 3.8+**
- **Ollama** (for local LLM inference)
- **4GB+ RAM** (8GB+ recommended for batch processing)
- **Stable internet connection** (for document discovery)

## 🔧 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/zumanm1/cisco-ios-rag-system-pro.git
cd cisco-ios-rag-system-pro
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install and configure Ollama:**
```bash
# Install Ollama (visit https://ollama.ai for platform-specific instructions)
ollama pull llama2  # or your preferred model
```

4. **Run the app**: `streamlit run app.py`
- Follow the on-screen instructions to upload your PDF and start processing.

### Output

The application generates a unique, timestamped folder for each run in the `OUTPUT-FOLDER` directory. This folder contains the generated knowledge libraries as structured JSON files, a copy of the fine-tuning guide, and a processing log.

For a detailed explanation of the output files and the application's complete workflow, please see `A_DETAIL_OUTPUT.txt`.

## 🎯 Enhanced Workflow

### **Discovery & Collection**
1. Select networking topics and CCIE categories
2. Configure discovery settings (max documents, auto-download)
3. Run automated discovery to find relevant documentation
4. Review and download selected documents

### **Batch Processing**
1. Process multiple documents simultaneously
2. Generate enhanced knowledge libraries with cross-document insights
3. Create comprehensive training datasets
4. Export structured results for model fine-tuning

### **Model Training**
1. Use generated JSON libraries as training data
2. Follow hardware-specific fine-tuning recommendations
3. Train specialized networking AI assistants

## 📊 Enhanced Output Structure

```
OUTPUT-FOLDER/
└── session_YYYYMMDD_HHMMSS/
    ├── batch_processing/
    │   ├── enhanced_libraries/
    │   │   ├── error_patterns_enhanced.json
    │   │   ├── best_practices_enhanced.json
    │   │   ├── command_syntax_enhanced.json
    │   │   ├── troubleshooting_enhanced.json
    │   │   └── cross_references_enhanced.json
    │   ├── document_metadata.json
    │   └── batch_report.json
    ├── downloaded_docs/
    ├── logs/
    └── discovery_reports/
```

## 🔍 What This App Does - Enhanced Version

**Input:** 
- Single Cisco IOS command guide PDF **OR**
- Multiple discovered/downloaded Cisco documentation

**Process:** 
1. **Discovery Phase**: Automatically finds relevant Cisco documentation across 16+ topics
2. **Collection Phase**: Downloads and organizes documents by topic and platform
3. **Analysis Phase**: Extracts and analyzes technical content using AI
4. **Synthesis Phase**: Creates enhanced knowledge libraries with cross-document insights
5. **Generation Phase**: Produces structured training datasets

**Output:** 
Five specialized JSON datasets plus enhanced cross-reference library:
- **Error Patterns Library** - Common mistakes and solutions
- **Best Practices Library** - Recommended implementation approaches  
- **Command Syntax Library** - Comprehensive syntax references
- **Troubleshooting Scenarios** - Diagnostic and resolution procedures
- **Cross-Reference Library** - Inter-topic relationships and dependencies

**Purpose:** Creates enterprise-grade training datasets that enable AI assistants to provide expert-level guidance across all major Cisco networking technologies.

## 🚀 Key Improvements in v2.0

- **10x More Content**: Processes multiple documents vs. single PDF
- **Enhanced Intelligence**: Cross-document correlation and synthesis
- **Automated Discovery**: No manual PDF hunting required
- **Comprehensive Coverage**: All major networking topics in one run
- **Enterprise Scale**: Handles dozens of documents efficiently
- **Better Training Data**: Richer, more diverse knowledge libraries

## 📝 License

This project is provided as-is for educational and professional use with Cisco documentation.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Transform your Cisco documentation into AI-ready knowledge with automated discovery and enterprise-scale processing!** 🚀 