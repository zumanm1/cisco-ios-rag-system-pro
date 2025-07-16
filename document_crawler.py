"""
Cisco Document Crawler Module
Automatically discovers and downloads Cisco documentation and CCIE books
"""

import requests
from bs4 import BeautifulSoup
import re
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import concurrent.futures
from fake_useragent import UserAgent
import json
from tqdm import tqdm

@dataclass
class DocumentInfo:
    title: str
    url: str
    file_size: Optional[str]
    description: str
    category: str
    platform: str  # IOS, IOS-XR, etc.
    topic: str
    download_status: str = "pending"
    local_path: Optional[str] = None

class CiscoDocumentCrawler:
    """Advanced web crawler for Cisco documentation and CCIE materials"""
    
    def __init__(self, download_dir: str = "downloaded_docs"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        self.ua = UserAgent()
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.ua.random})
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Topic configurations
        self.ios_topics = {
            "IPv4": ["ipv4", "ip routing", "static routing", "route maps"],
            "Layer 2 Switching": ["vlan", "stp", "spanning tree", "layer 2", "switching"],
            "IGP OSPF": ["ospf", "open shortest path first", "igp", "interior gateway"],
            "MPLS LDP": ["mpls", "label distribution protocol", "ldp", "mpls vpn"],
            "BGP": ["bgp", "border gateway protocol", "ebgp", "ibgp"],
            "MP-BGP": ["mp-bgp", "multiprotocol bgp", "vpnv4", "vpnv6"],
            "Route Reflector": ["route reflector", "rr", "bgp rr", "cluster"],
            "L2VPN": ["l2vpn", "layer 2 vpn", "pseudowire", "vpls"],
            "L3VPN": ["l3vpn", "layer 3 vpn", "vrf", "mpls vpn"],
            "QoS": ["qos", "quality of service", "traffic shaping", "policing"],
            "Services": ["service", "embedded event manager", "eem"],
            "FTP": ["ftp", "file transfer protocol", "tftp"],
            "SSH": ["ssh", "secure shell", "telnet", "remote access"],
            "TFTP": ["tftp", "trivial file transfer", "file transfer"],
            "SNMP": ["snmp", "simple network management", "monitoring"],
            "AAA": ["aaa", "authentication", "authorization", "accounting"],
            "NetFlow": ["netflow", "flow monitoring", "traffic analysis"]
        }
        
        self.ccie_categories = {
            "Service Provider": {
                "count": 10,
                "keywords": ["ccie service provider", "ccie sp", "service provider guide", "sp workbook"]
            },
            "Enterprise Infrastructure": {
                "count": 12,
                "keywords": ["ccie enterprise", "ccie ei", "enterprise infrastructure", "ei workbook"]
            },
            "Security": {
                "count": 9,
                "keywords": ["ccie security", "security guide", "firewall", "asa configuration"]
            }
        }
        
        # Common Cisco documentation sources
        self.cisco_sources = [
            "https://www.cisco.com/c/en/us/support/docs/",
            "https://www.cisco.com/c/en/us/td/docs/",
            "https://www.cisco.com/c/en/us/products/",
            "https://developer.cisco.com/docs/",
        ]
        
        # Additional sources for CCIE materials
        self.ccie_sources = [
            "https://www.ciscopress.com/",
            "https://www.packtpub.com/",
            "https://www.oreilly.com/",
            "https://www.informit.com/",
        ]
    
    def search_documents_by_topic(self, topic: str, platform: str = "IOS") -> List[DocumentInfo]:
        """Search for documents related to a specific topic"""
        self.logger.info(f"Searching for {platform} documents on {topic}")
        
        if topic not in self.ios_topics:
            self.logger.warning(f"Topic '{topic}' not found in predefined topics")
            return []
        
        keywords = self.ios_topics[topic]
        documents = []
        
        for keyword in keywords:
            docs = self._search_cisco_official(keyword, topic, platform)
            documents.extend(docs)
            time.sleep(1)  # Rate limiting
        
        # Remove duplicates based on URL
        unique_docs = {}
        for doc in documents:
            if doc.url not in unique_docs:
                unique_docs[doc.url] = doc
        
        return list(unique_docs.values())
    
    def search_ccie_books(self, category: str) -> List[DocumentInfo]:
        """Search for CCIE books in a specific category"""
        self.logger.info(f"Searching for CCIE {category} books")
        
        if category not in self.ccie_categories:
            self.logger.warning(f"CCIE category '{category}' not found")
            return []
        
        config = self.ccie_categories[category]
        keywords = config["keywords"]
        target_count = config["count"]
        
        documents = []
        
        for keyword in keywords:
            docs = self._search_ccie_materials(keyword, category)
            documents.extend(docs)
            if len(documents) >= target_count:
                break
            time.sleep(1)
        
        # Return up to target count
        return documents[:target_count]
    
    def _search_cisco_official(self, keyword: str, topic: str, platform: str) -> List[DocumentInfo]:
        """Search Cisco's official documentation"""
        documents = []
        
        # Search Cisco documentation
        search_url = f"https://www.cisco.com/c/en/us/support/docs/index.html"
        
        try:
            response = self.session.get(search_url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for PDF links
                pdf_links = soup.find_all('a', href=re.compile(r'\.pdf$', re.I))
                
                for link in pdf_links:
                    href = link.get('href')
                    if not href:
                        continue
                    
                    full_url = urljoin(search_url, href)
                    title = link.get_text(strip=True) or link.get('title', 'Cisco Documentation')
                    
                    # Check if the link is relevant to our keyword
                    if any(kw.lower() in title.lower() for kw in keyword.split()):
                        doc = DocumentInfo(
                            title=title,
                            url=full_url,
                            file_size=None,
                            description=f"Official Cisco documentation for {topic}",
                            category="Official Documentation",
                            platform=platform,
                            topic=topic
                        )
                        documents.append(doc)
                        
        except Exception as e:
            self.logger.error(f"Error searching Cisco docs for '{keyword}': {e}")
        
        return documents
    
    def _search_ccie_materials(self, keyword: str, category: str) -> List[DocumentInfo]:
        """Search for CCIE study materials"""
        documents = []
        
        # Search various sources for CCIE materials
        search_engines = [
            f"https://www.google.com/search?q={keyword.replace(' ', '+')}+filetype:pdf",
            f"https://www.bing.com/search?q={keyword.replace(' ', '+')}+filetype:pdf"
        ]
        
        for search_url in search_engines:
            try:
                response = self.session.get(search_url, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract search results (this would need to be adapted based on search engine)
                    results = soup.find_all('a', href=re.compile(r'\.pdf$', re.I))
                    
                    for result in results[:3]:  # Limit results per search
                        href = result.get('href')
                        if href and 'pdf' in href.lower():
                            title = result.get_text(strip=True) or f"CCIE {category} Study Material"
                            
                            doc = DocumentInfo(
                                title=title,
                                url=href,
                                file_size=None,
                                description=f"CCIE {category} study material",
                                category=f"CCIE {category}",
                                platform="Multi-platform",
                                topic=keyword
                            )
                            documents.append(doc)
                            
            except Exception as e:
                self.logger.error(f"Error searching for CCIE materials '{keyword}': {e}")
            
            time.sleep(2)  # Rate limiting between searches
        
        return documents
    
    def download_document(self, doc: DocumentInfo) -> bool:
        """Download a single document"""
        try:
            self.logger.info(f"Downloading: {doc.title}")
            
            response = self.session.get(doc.url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Generate filename
            filename = self._generate_filename(doc)
            filepath = self.download_dir / filename
            
            # Download with progress
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                if total_size > 0:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            
            doc.local_path = str(filepath)
            doc.download_status = "completed"
            doc.file_size = self._format_file_size(filepath.stat().st_size)
            
            self.logger.info(f"Successfully downloaded: {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download {doc.title}: {e}")
            doc.download_status = "failed"
            return False
    
    def download_documents_batch(self, documents: List[DocumentInfo], max_workers: int = 3) -> Dict[str, int]:
        """Download multiple documents concurrently"""
        stats = {"completed": 0, "failed": 0, "total": len(documents)}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_doc = {executor.submit(self.download_document, doc): doc for doc in documents}
            
            for future in concurrent.futures.as_completed(future_to_doc):
                doc = future_to_doc[future]
                try:
                    success = future.result()
                    if success:
                        stats["completed"] += 1
                    else:
                        stats["failed"] += 1
                except Exception as e:
                    self.logger.error(f"Error downloading {doc.title}: {e}")
                    stats["failed"] += 1
        
        return stats
    
    def discover_all_documents(self) -> Dict[str, List[DocumentInfo]]:
        """Discover all documents for all topics and CCIE categories"""
        all_documents = {}
        
        # IOS documents
        self.logger.info("Discovering IOS documents...")
        for topic in self.ios_topics.keys():
            docs = self.search_documents_by_topic(topic, "IOS")
            if docs:
                all_documents[f"IOS_{topic}"] = docs
        
        # IOS-XR documents
        self.logger.info("Discovering IOS-XR documents...")
        for topic in self.ios_topics.keys():
            docs = self.search_documents_by_topic(topic, "IOS-XR")
            if docs:
                all_documents[f"IOS-XR_{topic}"] = docs
        
        # CCIE books
        self.logger.info("Discovering CCIE books...")
        for category in self.ccie_categories.keys():
            docs = self.search_ccie_books(category)
            if docs:
                all_documents[f"CCIE_{category}"] = docs
        
        return all_documents
    
    def _generate_filename(self, doc: DocumentInfo) -> str:
        """Generate a safe filename for the document"""
        # Clean title for filename
        safe_title = re.sub(r'[^\w\s-]', '', doc.title)
        safe_title = re.sub(r'[-\s]+', '_', safe_title)
        
        # Add platform and topic prefix
        prefix = f"{doc.platform}_{doc.topic}".replace(" ", "_").replace("-", "_")
        
        return f"{prefix}_{safe_title}.pdf"
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def save_discovery_report(self, documents: Dict[str, List[DocumentInfo]], filepath: str):
        """Save document discovery report as JSON"""
        report = {}
        
        for category, docs in documents.items():
            report[category] = []
            for doc in docs:
                report[category].append({
                    "title": doc.title,
                    "url": doc.url,
                    "file_size": doc.file_size,
                    "description": doc.description,
                    "category": doc.category,
                    "platform": doc.platform,
                    "topic": doc.topic,
                    "download_status": doc.download_status,
                    "local_path": doc.local_path
                })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Discovery report saved to: {filepath}")

# Alternative document sources for better discovery
class EnhancedDocumentDiscovery:
    """Enhanced document discovery using multiple strategies"""
    
    def __init__(self):
        self.cisco_config_guides = {
            "IOS": [
                "https://www.cisco.com/c/en/us/td/docs/ios-xml/ios/fundamentals/configuration/",
                "https://www.cisco.com/c/en/us/td/docs/ios-xml/ios/iproute_bgp/configuration/",
                "https://www.cisco.com/c/en/us/td/docs/ios-xml/ios/iproute_ospf/configuration/",
            ],
            "IOS-XR": [
                "https://www.cisco.com/c/en/us/td/docs/iosxr/",
                "https://www.cisco.com/c/en/us/td/docs/routers/asr9000/software/",
            ]
        }
    
    def get_official_cisco_pdfs(self, platform: str, topic: str) -> List[str]:
        """Get direct links to official Cisco PDF documentation"""
        if platform not in self.cisco_config_guides:
            return []
        
        pdf_urls = []
        for base_url in self.cisco_config_guides[platform]:
            # This would be implemented to crawl the actual Cisco documentation tree
            # and find relevant PDFs based on the topic
            pass
        
        return pdf_urls 