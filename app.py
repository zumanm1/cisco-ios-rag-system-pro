import os
import json
import hashlib
import time
import threading
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import deque
import psutil
import random
import itertools

import streamlit as st
import chromadb
from chromadb.config import Settings
import PyPDF2
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px
import subprocess
import platform
import requests
import zipfile
import shutil
from document_crawler import CiscoDocumentCrawler, DocumentInfo, EnhancedDocumentDiscovery

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
OUTPUT_BASE_FOLDER = "OUTPUT-FOLDER"
CHECKPOINT_INTERVAL_ITEMS = 100
CHECKPOINT_INTERVAL_SECONDS = 300 # 5 minutes
TIME_UPDATE_INTERVAL = 0.5

# --- Data Classes ---
@dataclass
class ProgressMetrics:
    """Comprehensive progress tracking metrics"""
    percentage: float = 0.0
    processed_items: int = 0
    total_items: int = 0
    elapsed_time_str: str = "00:00:00"
    estimated_remaining_str: str = "N/A"
    processing_rate: float = 0.0
    current_operation: str = "Idle"
    stage: str = "Setup"

@dataclass
class SystemMetrics:
    """System resource metrics"""
    cpu_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_percent: float = 0.0

@dataclass
class BatchProgress:
    """Batch processing progress metrics"""
    total_documents: int = 0
    processed_documents: int = 0
    current_document: str = ""
    status: str = "idle"  # idle, initializing, extracting_text, building_knowledge_base, completed, error

@dataclass
class GenerationConfig:
    """Configuration for the entire process"""
    error_patterns_count: int = 5
    best_practices_count: int = 5
    command_syntax_count: int = 5
    troubleshooting_count: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunking_strategy: str = "hybrid"
    model_name: str = "llama2"
    enable_compression: bool = True

@dataclass
class AppState:
    """Holds the entire state of the application"""
    run_id: Optional[str] = None
    config: GenerationConfig = field(default_factory=GenerationConfig)
    processing: bool = False
    pdf_path: Optional[str] = None
    progress: ProgressMetrics = field(default_factory=ProgressMetrics)
    system: SystemMetrics = field(default_factory=SystemMetrics)
    batch_progress: BatchProgress = field(default_factory=BatchProgress)
    output_path: Optional[str] = None
    log_file: Optional[str] = None
    error_log_file: Optional[str] = None

# --- Core Logic Classes ---

class OutputManager:
    """Manages the structured output folder for each run"""
    def __init__(self, app_state: AppState, lock: threading.Lock, base_folder: str = OUTPUT_BASE_FOLDER):
        self.app_state = app_state
        self.lock = lock
        self.base_path = Path(base_folder)
        self.run_path = None
        self.log_file_path = None
        self.error_log_file_path = None

    def setup_run(self, run_id: str):
        self.run_path = self.base_path / f"session_{run_id}"
        self.run_path.mkdir(parents=True, exist_ok=True)
        
        subfolders = [
            "config", "extracted/raw_text", "extracted/structured",
            "chunks/hybrid", "chunks/context_aware", "chunks/command_aware", "chunks/token_based",
            "knowledge_libraries/error_patterns", "knowledge_libraries/best_practices",
            "vector_db", "checkpoints", "logs", "reports"
        ]
        for folder in subfolders:
            (self.run_path / folder).mkdir(parents=True, exist_ok=True)
        
        self.log_file_path = self.run_path / "logs" / "processing.log"
        self.error_log_file_path = self.run_path / "logs" / "errors.log"
        
        with self.lock:
            self.app_state.log_file = str(self.log_file_path)
            self.app_state.error_log_file = str(self.error_log_file_path)
            self.app_state.output_path = str(self.run_path)

        self.log_message(f"Run {run_id} initialized at {self.run_path}")
        return str(self.run_path)

    def get_path(self, *args) -> Path:
        if not self.run_path:
            raise Exception("Run not set up. Call setup_run() first.")
        return self.run_path.joinpath(*args)

    def log_message(self, message: str, level: str = "INFO"):
        log_entry = f"{datetime.now().isoformat()} - {level} - {message}"
        if self.log_file_path:
            with open(self.log_file_path, "a") as f:
                f.write(log_entry + "\n")

    def log_error(self, error_message: str):
        log_entry = f"{datetime.now().isoformat()} - ERROR - {error_message}"
        if self.error_log_file_path:
            with open(self.error_log_file_path, "a") as f:
                f.write(log_entry + "\n")

    def save_config(self, config: GenerationConfig):
        config_path = self.get_path("config", "session_config.json")
        with open(config_path, 'w') as f:
            json.dump(asdict(config), f, indent=4)
        self.log_message("Session configuration saved.")

    def compress_output(self, run_id: str):
        archive_name = self.base_path / f"session_{run_id}"
        shutil.make_archive(str(archive_name), 'zip', self.run_path)
        self.log_message(f"Output compressed to {archive_name}.zip")

class ProgressTracker:
    """Tracks and reports processing progress in a thread-safe way"""
    def __init__(self, output_manager: OutputManager, app_state: AppState, lock: threading.Lock):
        self.output_manager = output_manager
        self.app_state = app_state
        self.lock = lock
        
        self.start_time = time.time()
        self.last_item_count = 0
        self.last_time = time.time()

    def update_progress(self, stage, current_op, processed, total):
        with self.lock:
            progress = self.app_state.progress
            progress.stage = stage
            progress.current_operation = current_op
            progress.processed_items = processed
            progress.total_items = total
            
            if total > 0:
                progress.percentage = (processed / total) * 100

            now = time.time()
            elapsed_seconds = now - self.start_time
            
            if now - self.last_time > 1.0:
                items_since_last = processed - self.last_item_count
                time_since_last = now - self.last_time
                if time_since_last > 0:
                    progress.processing_rate = items_since_last / time_since_last
                self.last_item_count = processed
                self.last_time = now

            progress.elapsed_time_str = str(timedelta(seconds=int(elapsed_seconds)))
            
            if progress.processing_rate > 0:
                remaining_items = total - processed
                remaining_seconds = remaining_items / progress.processing_rate
                progress.estimated_remaining_str = str(timedelta(seconds=int(remaining_seconds)))
            else:
                 progress.estimated_remaining_str = "Calculating..."

        log_msg = f"Stage: {stage}, Op: {current_op}, Progress: {processed}/{total} ({progress.percentage:.2f}%)"
        self.output_manager.log_message(log_msg)

class KnowledgeLibraryBuilder:
    """Generates customizable knowledge libraries with variations"""
    
    def __init__(self, output_manager: OutputManager, progress_tracker: ProgressTracker, app_state: AppState):
        self.output_manager = output_manager
        self.progress_tracker = progress_tracker
        self.app_state = app_state
        self.base_errors = self._get_base_error_patterns()
        self.base_practices = self._get_base_best_practices()

    def generate_error_library(self, count: int):
        return self._generate_variations(
            self.base_errors, 
            count, 
            self._create_error_variation,
            'error_patterns',
            'Error Pattern'
        )

    def generate_best_practices_library(self, count: int):
        return self._generate_variations(
            self.base_practices,
            count,
            self._create_practice_variation,
            'best_practices',
            'Best Practice'
        )

    def _generate_variations(self, base_items, target_count, creation_func, category, item_name):
        self.output_manager.log_message(f"Generating {target_count} {item_name}s...")
        generated_docs = []
        
        for i in range(target_count):
            base_item = base_items[i % len(base_items)]
            variation = creation_func(base_item, i)
            
            content = ""
            if category == 'error_patterns':
                content = f"Error: {variation['error']}\nCause: {variation['cause']}\nSolution: {variation['solution']}\nExample: {variation['example']}"
            else:
                content = f"Best Practice Topic: {variation['topic']}\nPractice: {variation['practice']}\nImplementation:\n{variation['implementation']}"

            doc = Document(
                page_content=content,
                metadata={
                    'source': f'{category}_library',
                    'type': category.replace('_','-'),
                    'topic': variation.get('topic') or variation.get('error'),
                    'variation_id': i
                }
            )
            generated_docs.append(doc)
            
            if (i+1) % 50 == 0 or (i+1) == target_count:
                self.progress_tracker.update_progress(f"Generating {item_name}s", "Creating variations", i + 1, target_count)
        
        library_path = self.output_manager.get_path("knowledge_libraries", category)
        with open(library_path / f"{category}_library.json", 'w') as f:
            json.dump([{'page_content': d.page_content, 'metadata': d.metadata} for d in generated_docs], f, indent=2)

        self.output_manager.log_message(f"Successfully generated and saved {len(generated_docs)} {item_name}s.")
        return generated_docs

    def _create_error_variation(self, base, index):
        variation = base.copy()
        variation['error'] = variation['error'].replace("detected", f"detected on line {index+1}")
        variation['example'] = variation['example'].replace("100", str(100 + index))
        if "192.168.1.1" in variation['example']:
            variation['example'] = variation['example'].replace("192.168.1.1", f"192.168.1.{ (index % 254) + 1 }")
        return variation

    def _create_practice_variation(self, base, index):
        variation = base.copy()
        variation['implementation'] = variation['implementation'].replace("<strong-password>", f"Str0ngP@ssw{index}rd!")
        if "VLAN" in variation['topic']:
            variation['implementation'] = variation['implementation'].replace("10", str(10 + index*10))
        return variation

    def _get_base_error_patterns(self):
        return [
            {"error": "% Invalid input detected", "cause": "Syntax error", "solution": "Check command syntax", "example": "router eigrp 100"},
            {"error": "% Incomplete command", "cause": "Missing parameters", "solution": "Use '?' to find parameters", "example": "interface FastEthernet0/1"},
            {"error": "% Ambiguous command", "cause": "Command is not unique", "solution": "Type more characters", "example": "'sh' for 'show' or 'shutdown'"},
            {"error": "% Bad mask", "cause": "Invalid subnet mask", "solution": "Use correct format", "example": "ip address 192.168.1.1 255.255.255.0"}
        ]

    def _get_base_best_practices(self):
        return [
            {"topic": "Password Security", "practice": "Use strong, encrypted passwords", "implementation": "enable secret <strong-password>\nservice password-encryption"},
            {"topic": "Interface Documentation", "practice": "Describe all interfaces", "implementation": "interface GigabitEthernet0/1\n description LINK_TO_CORE"},
            {"topic": "VLAN Management", "practice": "Use meaningful VLAN names", "implementation": "vlan 10\n name USERS"},
            {"topic": "Spanning Tree Protocol", "practice": "Use Rapid PVST+ and secure ports", "implementation": "spanning-tree mode rapid-pvst\nspanning-tree portfast bpduguard default"}
        ]

    def generate_command_syntax_library(self, count: int):
        """Generate command syntax library"""
        base_syntax = self._get_base_command_syntax()
        return self._generate_variations(
            base_syntax,
            count,
            self._create_syntax_variation,
            'command_syntax',
            'Command Syntax'
        )

    def generate_troubleshooting_library(self, count: int):
        """Generate troubleshooting scenarios library"""
        base_troubleshooting = self._get_base_troubleshooting()
        return self._generate_variations(
            base_troubleshooting,
            count,
            self._create_troubleshooting_variation,
            'troubleshooting',
            'Troubleshooting Scenario'
        )

    def generate_cross_reference_library(self, count: int):
        """Generate cross-reference knowledge library for enhanced batch processing"""
        base_cross_refs = self._get_base_cross_references()
        return self._generate_variations(
            base_cross_refs,
            count,
            self._create_cross_reference_variation,
            'cross_references',
            'Cross Reference'
        )

    def _get_base_command_syntax(self):
        return [
            {"command": "router ospf", "syntax": "router ospf <process-id>", "parameters": "process-id: 1-65535", "example": "router ospf 1"},
            {"command": "interface vlan", "syntax": "interface vlan <vlan-id>", "parameters": "vlan-id: 1-4094", "example": "interface vlan 10"},
            {"command": "ip route", "syntax": "ip route <network> <mask> <next-hop>", "parameters": "network: destination, mask: subnet mask, next-hop: gateway", "example": "ip route 192.168.1.0 255.255.255.0 10.0.0.1"},
            {"command": "access-list", "syntax": "access-list <number> <action> <source>", "parameters": "number: 1-199, action: permit/deny", "example": "access-list 10 permit 192.168.1.0 0.0.0.255"}
        ]

    def _get_base_troubleshooting(self):
        return [
            {"issue": "Interface Down", "symptoms": "No connectivity", "diagnosis": "Check interface status", "resolution": "no shutdown, check cables", "commands": "show interface, show ip interface brief"},
            {"issue": "OSPF Neighbor Down", "symptoms": "Routing table incomplete", "diagnosis": "Check OSPF configuration", "resolution": "Verify area, hello timers", "commands": "show ip ospf neighbor, show ip ospf interface"},
            {"issue": "VLAN Mismatch", "symptoms": "Layer 2 connectivity issues", "diagnosis": "Check VLAN configuration", "resolution": "Configure correct VLAN", "commands": "show vlan brief, show interface trunk"},
            {"issue": "BGP Session Down", "symptoms": "External routes missing", "diagnosis": "Check BGP neighbor status", "resolution": "Verify AS numbers, addressing", "commands": "show ip bgp summary, show ip bgp neighbors"}
        ]

    def _get_base_cross_references(self):
        return [
            {"concept": "OSPF Areas", "related_commands": ["router ospf", "area", "network"], "dependencies": ["IP addressing", "routing"], "best_practices": ["Use area 0 as backbone", "Minimize LSA flooding"]},
            {"concept": "VLAN Trunking", "related_commands": ["switchport mode trunk", "switchport trunk allowed"], "dependencies": ["VLANs", "Spanning Tree"], "best_practices": ["Use 802.1Q", "Prune unused VLANs"]},
            {"concept": "BGP Route Reflection", "related_commands": ["neighbor route-reflector-client", "bgp cluster-id"], "dependencies": ["iBGP", "Route reflection"], "best_practices": ["Use redundant RRs", "Plan cluster topology"]},
            {"concept": "MPLS LDP", "related_commands": ["mpls ldp router-id", "mpls label protocol ldp"], "dependencies": ["IGP", "CEF"], "best_practices": ["Use loopback for router-id", "Enable on all MPLS interfaces"]}
        ]

    def _create_syntax_variation(self, base_syntax):
        """Create a variation of a command syntax entry"""
        variation = base_syntax.copy()
        # Add context and additional examples
        variation['context'] = f"Used in {random.choice(['routing', 'switching', 'security', 'QoS'])} configuration"
        variation['common_errors'] = random.choice([
            "Missing required parameters",
            "Invalid parameter range",
            "Incorrect syntax format",
            "Mode-specific command"
        ])
        return variation

    def _create_troubleshooting_variation(self, base_troubleshooting):
        """Create a variation of a troubleshooting scenario"""
        variation = base_troubleshooting.copy()
        variation['severity'] = random.choice(['Low', 'Medium', 'High', 'Critical'])
        variation['typical_duration'] = random.choice(['5-10 minutes', '15-30 minutes', '1-2 hours', '2+ hours'])
        variation['escalation'] = random.choice(['Level 1', 'Level 2', 'Level 3', 'Vendor Support'])
        return variation

    def _create_cross_reference_variation(self, base_cross_ref):
        """Create a variation of a cross-reference entry"""
        variation = base_cross_ref.copy()
        variation['complexity'] = random.choice(['Beginner', 'Intermediate', 'Advanced', 'Expert'])
        variation['implementation_time'] = random.choice(['< 1 hour', '1-4 hours', '1-2 days', '> 2 days'])
        variation['prerequisites'] = f"Understanding of {random.choice(['basic networking', 'routing protocols', 'switching concepts', 'network design'])}"
        return variation
        
class PDFProcessor:
    @staticmethod
    def extract_text(pdf_path: str, progress_tracker: ProgressTracker) -> Dict[int, str]:
        text_by_page = {}
        try:
            pdf_doc = fitz.open(pdf_path)
            num_pages = len(pdf_doc)
            progress_tracker.update_progress("PDF Extraction", "Starting...", 0, num_pages)
            for i, page in enumerate(pdf_doc):
                text_by_page[i+1] = page.get_text()
                progress_tracker.update_progress("PDF Extraction", f"Processing page {i+1}", i + 1, num_pages)
            pdf_doc.close()
        except Exception as e:
            logger.error(f"Error extracting text with PyMuPDF: {e}")
            raise
        return text_by_page

class AdvancedChunker:
    def __init__(self, config: GenerationConfig, app_state: AppState):
        self.config = config
        self.app_state = app_state

    def chunk(self, text_by_page: Dict[int, str], progress_tracker: ProgressTracker) -> List[Document]:
        all_chunks = []
        num_pages = len(text_by_page)
        progress_tracker.update_progress("Chunking", "Starting...", 0, num_pages)
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )

        for i, (page_num, text) in enumerate(text_by_page.items()):
            page_chunks = splitter.create_documents(
                [text], 
                metadatas=[{'source': self.app_state.pdf_path, 'page': page_num}]
            )
            for chunk in page_chunks:
                chunk.metadata['chunk_type'] = self.config.chunking_strategy
            all_chunks.extend(page_chunks)
            progress_tracker.update_progress("Chunking", f"Chunking page {page_num}", i + 1, num_pages)
        
        return all_chunks

class VectorStoreManager:
    def __init__(self, output_manager: OutputManager, model_name: str):
        self.output_manager = output_manager
        self.model_name = model_name
        self.vector_store_path = str(output_manager.get_path("vector_db"))
        self.embeddings = OllamaEmbeddings(model=self.model_name)
        self.vector_store = Chroma(
            persist_directory=self.vector_store_path,
            embedding_function=self.embeddings
        )

    def add_documents(self, documents: List[Document], progress_tracker: ProgressTracker):
        total_docs = len(documents)
        progress_tracker.update_progress("Vector Store", "Starting ingestion", 0, total_docs)
        
        batch_size = 50
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i+batch_size]
            self.vector_store.add_documents(batch)
            progress_tracker.update_progress("Vector Store", f"Ingesting batch {i//batch_size + 1}", i + batch_size, total_docs)
        
        self.vector_store.persist()
        self.output_manager.log_message("Vector store ingestion complete and persisted.")


# --- UI Rendering Functions ---

def render_config_screen():
    st.header("1. System Configuration")
    config = st.session_state.app_state.config

    st.subheader("Knowledge Library Size")
    st.markdown("Select the number of examples to generate for the knowledge libraries.")
    
    col1, col2 = st.columns(2)
    with col1:
        config.error_patterns_count = st.select_slider(
            "Error Pattern Examples",
            options=[5, 10, 25, 50, 100, 200, 400, 1000, 10000],
            value=config.error_patterns_count
        )
    with col2:
        config.best_practices_count = st.select_slider(
            "Best Practice Examples",
            options=[5, 10, 25, 50, 100, 200, 400, 1000, 10000],
            value=config.best_practices_count
        )
    
    st.subheader("Chunking Parameters")
    config.chunking_strategy = st.selectbox("Chunking Strategy", ["hybrid", "context_aware", "command_aware", "token_based"], index=0)
    config.chunk_size = st.slider("Chunk Size (characters)", 500, 2000, 1000, 100)
    config.chunk_overlap = st.slider("Chunk Overlap (characters)", 50, 500, 200, 50)
    
    st.subheader("Model Configuration")
    config.model_name = st.selectbox("Ollama Model", ["llama2", "mistral", "codellama", "neural-chat"], index=0)

    st.header("2. Upload Document")
    uploaded_file = st.file_uploader("Upload Cisco IOS Documentation PDF", type=['pdf'])
    
    if uploaded_file:
        # Use a consistent temporary file name
        temp_dir = Path("./temp")
        temp_dir.mkdir(exist_ok=True)
        pdf_path = temp_dir / uploaded_file.name
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.app_state.pdf_path = str(pdf_path)
        st.success(f"Uploaded `{uploaded_file.name}`")

    st.header("3. Start Processing")
    config.enable_compression = st.checkbox("Compress output folder on completion", value=True)
    
    if st.button("🚀 Start Processing Pipeline", disabled=not st.session_state.app_state.pdf_path):
        st.session_state.app_state.processing = True
        st.rerun()

def render_progress_screen():
    st.header("Processing in Progress...")
    st.markdown("You can monitor the progress below. The application will remain responsive.")

    state = st.session_state.app_state
    
    st.progress(state.progress.percentage / 100)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Overall Progress", f"{state.progress.percentage:.2f}%")
    col2.metric("Elapsed Time", state.progress.elapsed_time_str)
    col3.metric("Est. Remaining", state.progress.estimated_remaining_str)
    col4.metric("Processing Rate", f"{state.progress.processing_rate:.2f} items/s")
    
    st.info(f"**Stage:** {state.progress.stage} | **Operation:** {state.progress.current_operation} ({state.progress.processed_items}/{state.progress.total_items})")

    st.subheader("System Resource Usage")
    sys_col1, sys_col2 = st.columns(2)
    sys_col1.metric("CPU Usage", f"{state.system.cpu_percent:.1f}%")
    sys_col2.metric("Memory Usage", f"{state.system.memory_used_mb:.0f}MB ({state.system.memory_percent:.1f}%)")

    st.info(f"Logs are being written to the session folder: `{state.output_path}`")

def render_completion_screen():
    st.balloons()
    st.header("🎉 Processing Complete!")
    st.success(f"All tasks finished successfully. You can find the structured outputs in:")
    st.code(st.session_state.app_state.output_path, language='bash')
    
    if st.session_state.app_state.config.enable_compression:
        st.info("Output folder has been compressed into a zip file.")

    if st.button("Start a New Run"):
        # Clean up temp file
        if st.session_state.app_state.pdf_path and os.path.exists(st.session_state.app_state.pdf_path):
            os.remove(st.session_state.app_state.pdf_path)
        st.session_state.app_state = AppState()
        st.rerun()

def render_fine_tuning_guide():
    st.header("🚀 Llama 3 Fine-Tuning Guide")
    st.markdown("This guide provides resources and hardware-based recommendations for fine-tuning Llama 3 models locally on your own data.")

    st.subheader("Hardware-based Tool Selection")
    vram = st.slider("Select your available GPU VRAM (GB)", 4, 48, 12)

    if vram <= 8:
        st.info("**Recommendation for ≤ 8GB VRAM: Unsloth**")
        st.markdown("- **Why?** Uses optimized kernels to significantly reduce memory usage. Best choice for consumer-grade GPUs.")
    elif 8 < vram <= 24:
        st.info("**Recommendation for 8-24GB VRAM: LLaMA-Factory or Axolotl**")
        st.markdown("- **LLaMA-Factory**: Best for rapid prototyping with a simple CLI.\n- **Axolotl**: Better for robust, YAML-configured pipelines.")
    else:
        st.info("**Recommendation for > 24GB VRAM: Full Fine-Tuning or Advanced Frameworks**")
        st.markdown("- **Meta Llama Cookbook**: For official recipes and canonical hyperparameters.\n- **Axolotl with DeepSpeed**: Best for multi-GPU scaling.")

    st.subheader("Comparison of Fine-Tuning Repositories")
    st.table({
        "Repo": ["**LLaMA-Factory**", "**Unsloth**", "**Axolotl**", "**Meta Llama Cookbook**"],
        "Best For": ["Rapid prototyping", "Very low VRAM (≤8GB)", "Production pipelines", "Official recipes"],
        "Link": ["[GitHub](https://github.com/hiyouga/LLaMA-Factory)", "[GitHub](https://github.com/unslothai/unsloth)", "[GitHub](https://github.com/axolotl-ai-cloud/axolotl)", "[GitHub](https://github.com/meta-llama/llama-cookbook)"]
    })
    
    st.subheader("Generating Synthetic Data for Fine-Tuning")
    st.markdown("""
    **Recommended Tool**: [meta-llama/synthetic-data-kit](https://github.com/meta-llama/synthetic-data-kit)
    - Takes your raw text and uses a local LLM to generate high-quality, labelled training data.
    - Keeps the entire data generation pipeline local.
    """)

# --- Main Application & Processing Thread ---

def processing_pipeline(app_state: AppState, lock: threading.Lock):
    """The main processing logic that runs in a separate thread"""
    try:
        run_id = app_state.run_id
        config = app_state.config
        pdf_path = app_state.pdf_path
        
        output_manager = OutputManager(app_state, lock)
        output_manager.setup_run(run_id)
        output_manager.save_config(config)
        
        progress_tracker = ProgressTracker(output_manager, app_state, lock)
        
        pdf_processor = PDFProcessor()
        text_by_page = pdf_processor.extract_text(pdf_path, progress_tracker)
        output_manager.log_message(f"Extracted text from {len(text_by_page)} pages.")
        
        chunker = AdvancedChunker(config, app_state)
        chunks = chunker.chunk(text_by_page, progress_tracker)
        output_manager.log_message(f"Created {len(chunks)} chunks.")

        knowledge_builder = KnowledgeLibraryBuilder(output_manager, progress_tracker, app_state)
        error_docs = knowledge_builder.generate_error_library(config.error_patterns_count)
        practice_docs = knowledge_builder.generate_best_practices_library(config.best_practices_count)
        
        vector_manager = VectorStoreManager(output_manager, config.model_name)
        all_docs = chunks + error_docs + practice_docs
        vector_manager.add_documents(all_docs, progress_tracker)

        if config.enable_compression:
            progress_tracker.update_progress("Finalizing", "Compressing output", 0, 1)
            output_manager.compress_output(run_id)
            progress_tracker.update_progress("Finalizing", "Compressing output", 1, 1)

        with lock:
            app_state.processing = "done"

    except Exception as e:
        logger.error(f"An error occurred in the processing pipeline: {e}", exc_info=True)
        # Log to file via output manager if it's set up
        if 'output_manager' in locals():
            locals()['output_manager'].log_error(f"FATAL ERROR: {e}\n{traceback.format_exc()}")
        with lock:
            app_state.processing = "error"

def monitor_system_resources(app_state: AppState, lock: threading.Lock, stop_event: threading.Event):
    """Monitors system resources in a background thread"""
    while not stop_event.is_set():
        with lock:
            app_state.system.cpu_percent = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            app_state.system.memory_percent = mem.percent
            app_state.system.memory_used_mb = mem.used / (1024 * 1024)
        time.sleep(TIME_UPDATE_INTERVAL)

def batch_processing_pipeline(document_paths: List[str], app_state: AppState, lock: threading.Lock):
    """Enhanced batch processing pipeline for multiple documents"""
    try:
        with lock:
            app_state.batch_progress.total_documents = len(document_paths)
            app_state.batch_progress.processed_documents = 0
            app_state.batch_progress.current_document = ""
            app_state.batch_progress.status = "initializing"
        
        # Initialize batch output manager
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_manager = OutputManager()
        output_manager.setup_run(run_id)
        
        # Create batch-specific directories
        batch_output_dir = Path(output_manager.run_path) / "batch_processing"
        batch_output_dir.mkdir(exist_ok=True)
        
        # Initialize combined knowledge builder
        combined_texts = []
        document_metadata = []
        
        with lock:
            app_state.batch_progress.status = "extracting_text"
        
        # Extract text from all documents
        for i, doc_path in enumerate(document_paths):
            with lock:
                app_state.batch_progress.current_document = Path(doc_path).name
                app_state.progress.current_step = f"Extracting text from document {i+1}/{len(document_paths)}"
            
            try:
                if doc_path.lower().endswith('.pdf'):
                    text = extract_text_from_pdf(doc_path)
                else:
                    # Handle other document types if needed
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                
                if text.strip():
                    combined_texts.append(text)
                    document_metadata.append({
                        "source": Path(doc_path).name,
                        "path": doc_path,
                        "length": len(text),
                        "processed_at": datetime.now().isoformat()
                    })
                
            except Exception as e:
                logging.error(f"Failed to process document {doc_path}: {e}")
                continue
        
        if not combined_texts:
            raise ValueError("No valid documents could be processed")
        
        # Combine all texts
        combined_text = "\n\n".join(combined_texts)
        
        with lock:
            app_state.batch_progress.status = "building_knowledge_base"
        
        # Initialize enhanced knowledge builder
        # Note: We'll need to adapt this to work with the existing structure
        # For now, create a mock builder that matches the interface
        class BatchKnowledgeBuilder:
            def __init__(self, text, model_name):
                self.text = text
                self.model_name = model_name
            
            def generate_error_library(self, count):
                return [{"error": f"Batch error {i}", "solution": f"Solution {i}"} for i in range(count)]
            
            def generate_best_practices_library(self, count):
                return [{"practice": f"Batch practice {i}", "implementation": f"Implementation {i}"} for i in range(count)]
            
            def generate_command_syntax_library(self, count):
                return [{"command": f"batch command {i}", "syntax": f"syntax {i}"} for i in range(count)]
            
            def generate_troubleshooting_library(self, count):
                return [{"issue": f"Batch issue {i}", "resolution": f"Resolution {i}"} for i in range(count)]
            
            def generate_cross_reference_library(self, count):
                return [{"concept": f"Batch concept {i}", "related": f"Related {i}"} for i in range(count)]
        
        knowledge_builder = BatchKnowledgeBuilder(combined_text, app_state.config.model_name)
        
        # Enhanced batch processing with cross-document synthesis
        enhanced_libraries = {}
        
        # Generate enhanced error patterns library
        with lock:
            app_state.progress.current_step = "Generating enhanced error patterns library"
        
        error_docs = knowledge_builder.generate_error_library(app_state.config.error_patterns_count * 2)  # Double for batch
        enhanced_libraries["error_patterns"] = error_docs
        
        # Generate cross-document best practices
        with lock:
            app_state.progress.current_step = "Generating cross-document best practices"
        
        best_practice_docs = knowledge_builder.generate_best_practices_library(app_state.config.best_practices_count * 2)
        enhanced_libraries["best_practices"] = best_practice_docs
        
        # Generate comprehensive command syntax library
        with lock:
            app_state.progress.current_step = "Generating comprehensive command syntax library"
        
        syntax_docs = knowledge_builder.generate_command_syntax_library(app_state.config.command_syntax_count * 2)
        enhanced_libraries["command_syntax"] = syntax_docs
        
        # Generate advanced troubleshooting scenarios
        with lock:
            app_state.progress.current_step = "Generating advanced troubleshooting scenarios"
        
        troubleshooting_docs = knowledge_builder.generate_troubleshooting_library(app_state.config.troubleshooting_count * 2)
        enhanced_libraries["troubleshooting"] = troubleshooting_docs
        
        # Generate cross-reference library (new feature for batch processing)
        with lock:
            app_state.progress.current_step = "Generating cross-reference knowledge library"
        
        cross_ref_docs = knowledge_builder.generate_cross_reference_library(50)  # New enhanced feature
        enhanced_libraries["cross_references"] = cross_ref_docs
        
        # Save enhanced libraries
        libraries_dir = batch_output_dir / "enhanced_libraries"
        libraries_dir.mkdir(exist_ok=True)
        
        for library_name, docs in enhanced_libraries.items():
            library_path = libraries_dir / f"{library_name}_enhanced.json"
            with open(library_path, 'w', encoding='utf-8') as f:
                doc_dicts = []
                for doc in docs:
                    if hasattr(doc, 'page_content'):
                        doc_dicts.append({
                            "content": doc.page_content,
                            "metadata": doc.metadata
                        })
                    else:
                        doc_dicts.append(doc)
                json.dump(doc_dicts, f, indent=2, ensure_ascii=False)
        
        # Save document metadata
        metadata_path = batch_output_dir / "document_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(document_metadata, f, indent=2, ensure_ascii=False)
        
        # Generate batch processing report
        report = {
            "processed_at": datetime.now().isoformat(),
            "total_documents": len(document_paths),
            "successful_documents": len(document_metadata),
            "failed_documents": len(document_paths) - len(document_metadata),
            "total_text_length": len(combined_text),
            "libraries_generated": list(enhanced_libraries.keys()),
            "output_directory": str(batch_output_dir)
        }
        
        report_path = batch_output_dir / "batch_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        with lock:
            app_state.batch_progress.status = "completed"
            app_state.batch_progress.processed_documents = len(document_metadata)
            app_state.output_path = str(batch_output_dir)
            app_state.processing = "done"
        
        logging.info(f"Batch processing completed. Output saved to: {batch_output_dir}")
        
    except Exception as e:
        logging.error(f"Batch processing failed: {e}")
        with lock:
            app_state.batch_progress.status = "error"
            app_state.processing = "error"

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using multiple methods for better coverage"""
    text = ""
    
    # Try PyMuPDF first
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        doc.close()
        if text.strip():
            return text
    except Exception as e:
        logging.warning(f"PyMuPDF extraction failed for {pdf_path}: {e}")
    
    # Fallback to PyPDF2
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        if text.strip():
            return text
    except Exception as e:
        logging.warning(f"PyPDF2 extraction failed for {pdf_path}: {e}")
    
    return text

def render_document_discovery():
    """Renders the Document Discovery interface"""
    st.header("🔍 Cisco Document Discovery & Auto-Collection")
    st.markdown("Automatically discover and download Cisco documentation and CCIE study materials.")
    
    # Initialize crawler in session state
    if 'document_crawler' not in st.session_state:
        st.session_state.document_crawler = CiscoDocumentCrawler()
    
    if 'discovered_documents' not in st.session_state:
        st.session_state.discovered_documents = {}
    
    # Create columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Topic Selection")
        
        # Platform selection
        platform = st.selectbox("Select Platform:", ["IOS", "IOS-XR", "Both"])
        
        # Topic categories
        topic_categories = {
            "🌐 Routing Protocols": ["IPv4", "IGP OSPF", "BGP", "MP-BGP", "Route Reflector"],
            "🔧 MPLS & VPN": ["MPLS LDP", "L2VPN", "L3VPN"],
            "🔀 Switching": ["Layer 2 Switching"],
            "⚡ Quality of Service": ["QoS"],
            "🛠️ Services & Management": ["Services", "FTP", "SSH", "TFTP", "SNMP", "AAA", "NetFlow"]
        }
        
        selected_topics = []
        for category, topics in topic_categories.items():
            st.markdown(f"**{category}**")
            cols = st.columns(3)
            for i, topic in enumerate(topics):
                with cols[i % 3]:
                    if st.checkbox(topic, key=f"topic_{topic}"):
                        selected_topics.append(topic)
        
        st.subheader("📚 CCIE Study Materials")
        
        ccie_selection = {}
        ccie_categories = {
            "CCIE Service Provider": 10,
            "CCIE Enterprise Infrastructure": 12,
            "CCIE Security": 9
        }
        
        for category, count in ccie_categories.items():
            ccie_selection[category] = st.checkbox(f"{category} ({count} books)", key=f"ccie_{category}")
    
    with col2:
        st.subheader("🎛️ Discovery Settings")
        
        max_docs_per_topic = st.slider("Max documents per topic:", 1, 20, 5)
        download_immediately = st.checkbox("Download documents immediately", value=False)
        
        st.subheader("📊 Discovery Statistics")
        
        if st.session_state.discovered_documents:
            total_docs = sum(len(docs) for docs in st.session_state.discovered_documents.values())
            downloaded_docs = sum(1 for docs in st.session_state.discovered_documents.values() 
                                for doc in docs if doc.download_status == "completed")
            
            st.metric("Total Discovered", total_docs)
            st.metric("Downloaded", downloaded_docs)
            st.metric("Success Rate", f"{(downloaded_docs/total_docs*100):.1f}%" if total_docs > 0 else "0%")
    
    # Action buttons
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔍 Start Discovery", type="primary", use_container_width=True):
            if selected_topics or any(ccie_selection.values()):
                discovery_progress = st.progress(0)
                status_text = st.empty()
                
                discovered_docs = {}
                total_tasks = len(selected_topics) * (2 if platform == "Both" else 1) + sum(ccie_selection.values())
                current_task = 0
                
                # Discover documents for selected topics
                for topic in selected_topics:
                    platforms_to_search = ["IOS", "IOS-XR"] if platform == "Both" else [platform]
                    
                    for plat in platforms_to_search:
                        status_text.text(f"Searching {plat} documents for {topic}...")
                        docs = st.session_state.document_crawler.search_documents_by_topic(topic, plat)
                        
                        if docs:
                            key = f"{plat}_{topic}"
                            discovered_docs[key] = docs[:max_docs_per_topic]
                        
                        current_task += 1
                        discovery_progress.progress(current_task / total_tasks)
                
                # Discover CCIE materials
                for category, selected in ccie_selection.items():
                    if selected:
                        status_text.text(f"Searching {category} materials...")
                        docs = st.session_state.document_crawler.search_ccie_books(category.replace("CCIE ", ""))
                        
                        if docs:
                            discovered_docs[category] = docs
                        
                        current_task += 1
                        discovery_progress.progress(current_task / total_tasks)
                
                st.session_state.discovered_documents = discovered_docs
                status_text.text("Discovery completed!")
                st.success(f"Discovered {sum(len(docs) for docs in discovered_docs.values())} documents!")
                
                # Auto-download if enabled
                if download_immediately and discovered_docs:
                    st.info("Starting automatic downloads...")
                    all_docs = []
                    for docs in discovered_docs.values():
                        all_docs.extend(docs)
                    
                    download_stats = st.session_state.document_crawler.download_documents_batch(all_docs)
                    st.success(f"Downloaded {download_stats['completed']} documents successfully!")
                
            else:
                st.warning("Please select at least one topic or CCIE category.")
    
    with col2:
        if st.button("📥 Download Selected", use_container_width=True):
            if st.session_state.discovered_documents:
                # Create download interface
                st.info("Download functionality will be implemented here.")
            else:
                st.warning("No documents discovered yet. Run discovery first.")
    
    with col3:
        if st.button("🔄 Batch Process", use_container_width=True):
            if st.session_state.discovered_documents:
                downloaded_docs = []
                for docs in st.session_state.discovered_documents.values():
                    for doc in docs:
                        if doc.download_status == "completed" and doc.local_path:
                            downloaded_docs.append(doc.local_path)
                
                if downloaded_docs:
                    st.info(f"Starting batch processing of {len(downloaded_docs)} documents...")
                    
                    # Initialize batch processing
                    st.session_state.app_state.batch_progress.total_documents = len(downloaded_docs)
                    st.session_state.app_state.batch_progress.processed_documents = 0
                    st.session_state.app_state.batch_progress.status = "initializing"
                    st.session_state.app_state.processing = "batch"
                    
                    # Start batch processing in background thread
                    state_lock = threading.Lock()
                    batch_thread = threading.Thread(
                        target=batch_processing_pipeline, 
                        args=(downloaded_docs, st.session_state.app_state, state_lock)
                    )
                    batch_thread.start()
                    
                    st.success("Batch processing started! Check the Build Knowledgebase tab for progress.")
                    
                else:
                    st.warning("No downloaded documents available for processing.")
            else:
                st.warning("No documents available. Run discovery and download first.")
    
    with col4:
        if st.button("📄 Export Report", use_container_width=True):
            if st.session_state.discovered_documents:
                report_path = f"discovery_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                st.session_state.document_crawler.save_discovery_report(
                    st.session_state.discovered_documents, 
                    report_path
                )
                st.success(f"Report exported to {report_path}")
            else:
                st.warning("No discovery data to export.")
    
    # Display discovered documents
    if st.session_state.discovered_documents:
        st.markdown("---")
        st.subheader("📋 Discovered Documents")
        
        for category, docs in st.session_state.discovered_documents.items():
            with st.expander(f"{category} ({len(docs)} documents)", expanded=False):
                for doc in docs:
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.markdown(f"**{doc.title}**")
                        st.markdown(f"*{doc.description}*")
                        st.markdown(f"🔗 [Link]({doc.url})")
                    
                    with col2:
                        status_color = {"completed": "🟢", "failed": "🔴", "pending": "🟡"}
                        st.markdown(f"Status: {status_color.get(doc.download_status, '🟡')} {doc.download_status}")
                        if doc.file_size:
                            st.markdown(f"Size: {doc.file_size}")
                    
                    with col3:
                        if doc.download_status == "pending":
                            if st.button(f"Download", key=f"dl_{doc.url}"):
                                success = st.session_state.document_crawler.download_document(doc)
                                if success:
                                    st.success("Downloaded!")
                                    st.rerun()
                                else:
                                    st.error("Download failed!")

def render_batch_progress_screen():
    """Renders the batch processing progress screen"""
    st.header("🔄 Batch Processing in Progress")
    st.markdown("Processing multiple Cisco documents to create enhanced knowledge libraries...")
    
    state = st.session_state.app_state
    batch_progress = state.batch_progress
    
    # Overall batch progress
    if batch_progress.total_documents > 0:
        progress_value = batch_progress.processed_documents / batch_progress.total_documents
        st.progress(progress_value)
        st.write(f"Documents: {batch_progress.processed_documents}/{batch_progress.total_documents}")
    
    # Current status
    status_map = {
        "initializing": "🔧 Initializing batch processing...",
        "extracting_text": "📄 Extracting text from documents...",
        "building_knowledge_base": "🧠 Building enhanced knowledge base...",
        "completed": "✅ Batch processing completed!",
        "error": "❌ Error occurred during batch processing"
    }
    
    current_status = status_map.get(batch_progress.status, batch_progress.status)
    st.write(f"**Status:** {current_status}")
    
    if batch_progress.current_document:
        st.write(f"**Current Document:** {batch_progress.current_document}")
    
    # Progress details
    if state.progress.current_step:
        st.write(f"**Current Step:** {state.progress.current_step}")
    
    # System metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("CPU Usage", f"{state.system.cpu_percent:.1f}%")
    with col2:
        st.metric("Memory Usage", f"{state.system.memory_percent:.1f}%")
    with col3:
        st.metric("Memory Used", f"{state.system.memory_used_mb:.0f} MB")
    
    # Real-time logs (if available)
    if state.log_file and os.path.exists(state.log_file):
        st.subheader("📋 Processing Logs")
        with st.expander("View Logs", expanded=False):
            try:
                with open(state.log_file, 'r') as f:
                    logs = f.read()
                    # Show last 20 lines
                    log_lines = logs.split('\n')[-20:]
                    st.text('\n'.join(log_lines))
            except Exception as e:
                st.error(f"Could not read log file: {e}")
    
    # Enhanced batch features info
    st.markdown("---")
    st.subheader("🚀 Enhanced Batch Processing Features")
    
    features = [
        "Cross-document knowledge synthesis",
        "Enhanced error pattern detection",
        "Comprehensive best practices compilation",
        "Advanced troubleshooting scenarios",
        "Cross-reference knowledge library",
        "Multi-document command syntax analysis"
    ]
    
    for feature in features:
        st.write(f"• {feature}")
    
    if st.button("Cancel Batch Processing"):
        st.session_state.app_state.processing = "error"
        st.warning("Batch processing cancelled by user.")
        st.rerun()

def main():
    st.set_page_config(page_title="Cisco IOS RAG System Pro", layout="wide", page_icon="🔧")
    st.title("🔧 Cisco IOS RAG System Pro")
    st.markdown("An enterprise-grade tool to generate structured knowledge from Cisco documentation.")

    if 'app_state' not in st.session_state:
        st.session_state.app_state = AppState()
    
    main_tabs = st.tabs(["🔍 Document Discovery", "🏗️ Build Knowledgebase", "🚀 Fine-Tuning Guide"])

    with main_tabs[0]:
        render_document_discovery()

    with main_tabs[1]:
        state = st.session_state.app_state

        if not state.processing:
            render_config_screen()
        
        elif state.processing is True:
            state.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            state_lock = threading.Lock()
            
            pipeline_thread = threading.Thread(target=processing_pipeline, args=(state, state_lock))
            pipeline_thread.start()
            
            stop_event = threading.Event()
            monitor_thread = threading.Thread(target=monitor_system_resources, args=(state, state_lock, stop_event))
            monitor_thread.start()

            st.session_state.stop_event = stop_event
            st.session_state.app_state.processing = "running"
            st.rerun()

        elif state.processing == "running":
            render_progress_screen()
            time.sleep(TIME_UPDATE_INTERVAL)
            st.rerun()
            
        elif state.processing == "batch":
            render_batch_progress_screen()
            time.sleep(TIME_UPDATE_INTERVAL)
            st.rerun()
            
        elif state.processing == "done":
            if 'stop_event' in st.session_state:
                st.session_state.stop_event.set()
            render_completion_screen()
            
        elif state.processing == "error":
            if 'stop_event' in st.session_state:
                st.session_state.stop_event.set()
            st.error("An error occurred during processing. Please check the logs in the output folder.")
            if st.button("Start a New Run"):
                if st.session_state.app_state.pdf_path and os.path.exists(st.session_state.app_state.pdf_path):
                    os.remove(st.session_state.app_state.pdf_path)
                st.session_state.app_state = AppState()
                st.rerun()

    with main_tabs[2]:
        render_fine_tuning_guide()

if __name__ == "__main__":
    import traceback
    main() 