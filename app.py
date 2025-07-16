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
class GenerationConfig:
    """Configuration for the entire process"""
    error_patterns_count: int = 5
    best_practices_count: int = 5
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

def main():
    st.set_page_config(page_title="Cisco IOS RAG System Pro", layout="wide", page_icon="🔧")
    st.title("🔧 Cisco IOS RAG System Pro")
    st.markdown("An enterprise-grade tool to generate structured knowledge from Cisco documentation.")

    if 'app_state' not in st.session_state:
        st.session_state.app_state = AppState()
    
    main_tabs = st.tabs(["🏗️ Build Knowledgebase", "🚀 Fine-Tuning Guide"])

    with main_tabs[0]:
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

    with main_tabs[1]:
        render_fine_tuning_guide()

if __name__ == "__main__":
    import traceback
    main() 