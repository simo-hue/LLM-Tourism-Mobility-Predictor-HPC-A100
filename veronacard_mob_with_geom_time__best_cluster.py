# ============= IMPORT =============
import argparse
import json
import logging
import math
import os
import queue
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from threading import Lock, Semaphore
from typing import Dict, Any, List, Optional, Tuple, Set, Union

# Third-party imports
import numpy as np
import pandas as pd
import requests
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ============= CONFIGURATION =============
class Config:
    """Centralized configuration to avoid global variables"""
    
    # Model configuration
    MODEL_NAME = "llama3.1:8b" #llama3.1:8b - qwen2.5:7b - qwen2.5:14b - mixtral:8x7b - mistral:7b - deepseek-coder_33b
    modello = MODEL_NAME.replace(":", "_")
    strategia = "with_geom_time_best_cluster"
    
    TOP_K = 5  # Number of POI predictions
    
    # HPC optimization parameters - OPTIMIZED FOR 4x A100
    MAX_CONCURRENT_REQUESTS = 12  # 4 GPUs × 4 requests per GPU (OPTIMIZED)
    REQUEST_TIMEOUT = 600  # Increased for stability (was 300)
    BATCH_SAVE_INTERVAL = 1000  # Save results every N cards
    HEALTH_CHECK_INTERVAL = 600  # Check host health every N seconds
    
    # Retry and failure handling
    MAX_RETRIES_PER_REQUEST = 10
    MAX_CONSECUTIVE_FAILURES = 50
    BACKOFF_BASE = 2
    BACKOFF_MAX = 120
    CIRCUIT_BREAKER_THRESHOLD = 100
    
    # 503 specific handling
    RETRY_ON_503_WAIT = 60  # ✅ NUOVO: attesa specifica per 503
    MAX_503_RETRIES = 20    # ✅ NUOVO: retry dedicati per 503
    
    # Anchor rule for POI selection
    DEFAULT_ANCHOR_RULE = "middle"
    
    # Parallelism - OPTIMIZED FOR A100 64GB CAPACITY
    MAX_CONCURRENT_PER_GPU = 3
    
    # File paths
    OLLAMA_PORT_FILE = "ollama_ports.txt"
    LOG_DIR = Path(__file__).resolve().parent / "logs"
    RESULTS_DIR = Path(__file__).resolve().parent / f"results/{DEFAULT_ANCHOR_RULE}/{modello}/{strategia}/"
    DATA_DIR = Path(__file__).resolve().parent / "data" / "verona"
    POI_FILE = DATA_DIR / "vc_site.csv"

# ============= LOGGING SETUP =============
def setup_logging() -> logging.Logger:
    """Configure logging with file and console output"""
    Config.LOG_DIR.mkdir(exist_ok=True)
    log_file = Config.LOG_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Create formatter without special characters
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Configure logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Remove any existing handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
# Initialize logger
logger = setup_logging()

# ============= THREAD-SAFE STATISTICS =============
class Statistics:
    """Thread-safe statistics tracking"""
    
    def __init__(self):
        self._lock = Lock()
        self._data = {
            'total_processed': 0,
            'total_errors': 0,
            'consecutive_failures': 0,
            'last_success_time': time.time(),
            'host_failures': {},
            'circuit_breaker_active': False
        }
    
    def increment_processed(self):
        with self._lock:
            self._data['total_processed'] += 1
            self._data['consecutive_failures'] = 0
            self._data['last_success_time'] = time.time()
    
    def increment_errors(self):
        with self._lock:
            self._data['total_errors'] += 1
            self._data['consecutive_failures'] += 1
    
    def get_stats(self) -> dict:
        with self._lock:
            return self._data.copy()
    
    def get_success_rate(self) -> float:
        with self._lock:
            total = self._data['total_processed'] + self._data['total_errors']
            if total == 0:
                return 0.0
            return (self._data['total_processed'] / total) * 100
    
    def reset_consecutive_failures(self):
        with self._lock:
            self._data['consecutive_failures'] = 0
            
    @property
    def consecutive_failures(self) -> int:
        with self._lock:
            return self._data['consecutive_failures']
    
    @property
    def circuit_breaker_active(self) -> bool:
        with self._lock:
            return self._data['circuit_breaker_active']
    
    def set_circuit_breaker(self, active: bool):
        with self._lock:
            self._data['circuit_breaker_active'] = active

# Global instances
stats = Statistics()
write_lock = Lock()  # Global lock for file writing

# ============= CIRCUIT BREAKER =============
class CircuitBreaker:
    """
    Circuit Breaker pattern implementation for handling cascading failures.
    
    States:
    - CLOSED: Normal operation
    - OPEN: Failures exceeded threshold, rejecting requests
    - HALF_OPEN: Testing if service recovered
    """
    
    def __init__(self, failure_threshold: int = 10, timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure = None
        self.state = "CLOSED"
        self._lock = Lock()
    
    @contextmanager
    def call(self):
        """Context manager for circuit breaker protected calls"""
        with self._lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure > self.timeout:
                    self.state = "HALF_OPEN"
                    logger.info("Circuit breaker: Attempting HALF_OPEN state")
                else:
                    raise Exception("Circuit breaker OPEN - system paused")
        
        try:
            yield
            with self._lock:
                if self.state == "HALF_OPEN":
                    self.reset()
        except Exception as e:
            self.record_failure()
            raise e
    
    def record_failure(self):
        """Record a failure and potentially open the circuit"""
        with self._lock:
            self.failures += 1
            self.last_failure = time.time()
            
            if self.failures >= self.failure_threshold:
                self.state = "OPEN"
                logger.error(f"Circuit breaker OPEN after {self.failures} failures")
                stats.set_circuit_breaker(True)
    
    def reset(self):
        """Reset circuit breaker to closed state"""
        with self._lock:
            self.failures = 0
            self.state = "CLOSED"
            stats.set_circuit_breaker(False)
            logger.info("Circuit breaker RESET to CLOSED state")

# ============= HOST HEALTH MONITORING =============
class HostHealthMonitor:
    """
    Monitors health status of Ollama hosts.
    
    Features:
    - Health checks with configurable intervals
    - Response time tracking
    - Performance-based host selection
    - Automatic failover
    """
    
    def __init__(self, hosts: List[str]):
        self.hosts = hosts if hosts else []
        self.health_status = {host: True for host in self.hosts}
        self.last_check = {host: 0 for host in self.hosts}
        self.response_times = {host: [] for host in self.hosts}
        self._lock = Lock()
        self._max_response_history = 5
        self._round_robin_index = 0  
        
        # Log di warning per inizializzazione vuota
        if not hosts:
            logger.warning("HostHealthMonitor initialized with empty host list")
    
    def is_healthy(self, host: str) -> bool:
        """Check if a specific host is healthy"""
        with self._lock:
            return self.health_status.get(host, False)
    
    def get_healthy_hosts(self) -> List[str]:
        """Get list of all healthy hosts"""
        if not self.hosts:
            logger.warning("No hosts available in health monitor")
            return []
        
        with self._lock:
            healthy = [host for host, healthy in self.health_status.items() if healthy]
            if not healthy:
                # Tentativo di recovery: ricontrolla tutti gli host
                logger.warning("No healthy hosts found, attempting recovery...")
                for host in self.hosts:
                    if self._quick_health_check(host):
                        self.health_status[host] = True
                        healthy.append(host)
                        
            return healthy
    
    def _quick_health_check(self, host: str) -> bool:
        """Quick health check without updating response times"""
        try:
            resp = requests.get(f"{host}/api/tags", timeout=2)
            return resp.status_code == 200
        except:
            return False
    
    def check_health(self, host: str) -> bool:
        """
        Perform health check on a specific host.
        Uses lightweight endpoint to minimize overhead.
        """
        try:
            start_time = time.time()
            resp = requests.get(
                f"{host}/api/tags", 
                timeout=3,
                headers={'Accept': 'application/json'}
            )
            response_time = time.time() - start_time
            
            with self._lock:
                # Track response times (keep only recent ones)
                self.response_times[host].append(response_time)
                if len(self.response_times[host]) > self._max_response_history:
                    self.response_times[host].pop(0)
                
                is_healthy = resp.status_code == 200
                self.health_status[host] = is_healthy
                self.last_check[host] = int(time.time())
                
                if is_healthy:
                    logger.debug(f"Host {host} healthy (response time: {response_time:.2f}s)")
                else:
                    logger.warning(f"Host {host} unhealthy (status: {resp.status_code})")
                
                return is_healthy
                
        except Exception as e:
            logger.warning(f"Health check failed for {host}: {e}")
            with self._lock:
                self.health_status[host] = False
            return False
    
    def get_round_robin_host(self) -> Optional[str]:
        """
        Select host using round-robin algorithm to ensure even distribution
        """
        healthy_hosts = self.get_healthy_hosts()
        if not healthy_hosts:
            return None
        
        with self._lock:
            # Use modulo to cycle through healthy hosts
            host = healthy_hosts[self._round_robin_index % len(healthy_hosts)]
            self._round_robin_index += 1
            logger.debug(f"Round-robin selected: {host} (index: {self._round_robin_index})")
            return host
    
    def get_best_host(self) -> Optional[str]:
        """
        Select the best performing host based on response times and health.
        Uses sophisticated scoring algorithm.
        """
        healthy_hosts = self.get_healthy_hosts()
        if not healthy_hosts:
            return None
        
        if len(healthy_hosts) == 1:
            return healthy_hosts[0]
        
        with self._lock:
            host_scores = []
            
            for host in healthy_hosts:
                recent_times = self.response_times.get(host, [1.0])
                if not recent_times:
                    recent_times = [1.0]
                
                # Calculate average response time
                avg_time = sum(recent_times) / len(recent_times)
                
                # Calculate trend (positive = getting slower)
                if len(recent_times) > 1:
                    trend = recent_times[-1] - recent_times[0]
                else:
                    trend = 0
                
                # Score calculation (lower is better)
                # Penalize hosts with increasing response times
                score = avg_time + (trend * 2)
                host_scores.append((host, score))
            
            # Return host with lowest score
            best_host = min(host_scores, key=lambda x: x[1])[0]
            logger.debug(f"Performance-based selected: {best_host}")
            return best_host
    
    def get_balanced_host(self) -> Optional[str]:
        """
        Balanced host selection: 70% round-robin, 30% performance-based
        This ensures good distribution while still considering performance
        """
        import random
        
        healthy_hosts = self.get_healthy_hosts()
        if not healthy_hosts:
            return None
        
        # 70% delle volte usa round-robin per distribuzione uniforme
        # 30% usa performance-based per ottimizzazione
        if random.random() < 0.7:
            host = self.get_round_robin_host()
            logger.debug("Using round-robin selection")
            return host
        else:
            host = self.get_best_host()
            logger.debug("Using performance-based selection")
            return host
    
    def get_host_stats(self) -> Dict[str, Dict]:
        """Get statistics for all hosts"""
        with self._lock:
            stats = {}
            for host in self.hosts:
                recent_times = self.response_times.get(host, [])
                stats[host] = {
                    'healthy': self.health_status.get(host, False),
                    'last_check': self.last_check.get(host, 0),
                    'avg_response_time': sum(recent_times) / len(recent_times) if recent_times else 0,
                    'recent_requests': len(recent_times)
                }
            return stats
    
    def reset_round_robin(self):
        """Reset round-robin counter (useful for testing)"""
        with self._lock:
            self._round_robin_index = 0
            logger.debug("Round-robin index reset to 0")

# ============= OLLAMA CONNECTION MANAGEMENT =============
class OllamaConnectionManager:
    """Manages Ollama connections and API interactions"""
    
    def __init__(self):
        self.hosts: List[str] = []
        self.rate_limiter: Semaphore = Semaphore(1)  # Default semaforo con 1 permit
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=Config.CIRCUIT_BREAKER_THRESHOLD,
            timeout=300
        )
        self.health_monitor: HostHealthMonitor = HostHealthMonitor([])  # Lista vuota iniziale
        
    def setup_connections(self) -> List[str]:
        """Setup Ollama connections from port configuration file"""
        try:
            with open(Config.OLLAMA_PORT_FILE, "r") as f:
                ports_str = f.read().strip()
            
            if "," in ports_str:
                # Multi-GPU configuration
                ports = [p.strip() for p in ports_str.split(",")]
                self.hosts = [f"http://127.0.0.1:{port}" for port in ports]
                logger.info(f"Multi-GPU configuration: {len(self.hosts)} instances")
                
                self.rate_limiter = Semaphore(len(self.hosts) * Config.MAX_CONCURRENT_PER_GPU) 
                
            else:
                # Single GPU fallback
                self.hosts = [f"http://127.0.0.1:{ports_str}"]
                logger.info(f"Single GPU configuration: {self.hosts[0]}")
                self.rate_limiter = Semaphore(1)
            
            # RE-inizializza health monitor con hosts corretti
            self.health_monitor = HostHealthMonitor(self.hosts)
            
            return self.hosts
            
        except FileNotFoundError:
            raise RuntimeError(f"Configuration file {Config.OLLAMA_PORT_FILE} not found")
        except Exception as e:
            raise RuntimeError(f"Failed to setup Ollama connections: {e}")
    
    def wait_for_services(self, max_attempts: int = 30, wait_interval: int = 3) -> bool:
        """Wait for all Ollama services to be ready"""
        logger.info("Waiting for Ollama services to start...")
        
        for i, host in enumerate(self.hosts):
            logger.info(f"Checking host {i+1}/{len(self.hosts)}: {host}")
            
            for attempt in range(1, max_attempts + 1):
                try:
                    response = requests.get(
                        f"{host}/api/tags",
                        timeout=10,
                        headers={'Accept': 'application/json'}
                    )
                    
                    if response.status_code == 200:
                        logger.info(f"Host {host} is ready")
                        break
                        
                except requests.exceptions.RequestException as e:
                    logger.debug(f"Attempt {attempt}/{max_attempts} failed: {e}")
                    
                if attempt < max_attempts:
                    time.sleep(wait_interval)
                else:
                    logger.error(f"Host {host} failed to respond after {max_attempts} attempts")
                    return False
        
        logger.info("All Ollama services are ready!")
        return True
    
    def test_model_availability(self, model: str = Config.MODEL_NAME) -> bool:
        """Test if the specified model is available on all hosts"""
        working_hosts = 0
        
        for host in self.hosts:
            try:
                # Check available models
                resp = requests.get(f"{host}/api/tags", timeout=10)
                if resp.status_code != 200:
                    logger.error(f"Failed to get models from {host}")
                    continue
                
                models = [m.get('name', '') for m in resp.json().get('models', [])]
                if model not in models:
                    logger.error(f"Model {model} not found on {host}")
                    continue
                
                # Test inference capability
                test_payload = {
                    "model": model,
                    "prompt": "Hi",
                    "stream": False,
                    "options": {
                        "num_predict": 1,
                        "temperature": 0
                    }
                }
                
                test_resp = requests.post(
                    f"{host}/api/generate",
                    json=test_payload,
                    timeout=60
                )
                
                if test_resp.status_code == 200:
                    data = test_resp.json()
                    if data.get("done") and data.get("response"):
                        working_hosts += 1
                        logger.info(f"Model {model} working on {host}")
                    else:
                        logger.error(f"Model test failed on {host}: incomplete response")
                else:
                    logger.error(f"Model test failed on {host}: HTTP {test_resp.status_code}")
                    
            except Exception as e:
                logger.error(f"Error testing {host}: {e}")
        
        logger.info(f"{working_hosts}/{len(self.hosts)} hosts have working model")
        return working_hosts > 0
    
    def get_chat_completion(self, prompt: str, model: str = Config.MODEL_NAME) -> Optional[str]:
        """Get chat completion with load balancing and error handling"""
        
        # Check circuit breaker
        if stats.circuit_breaker_active:
            logger.warning("Circuit breaker active - skipping request")
            return None
        
        # Controllo di sicurezza
        if self.rate_limiter is None:
            logger.error("Rate limiter not initialized - call setup_connections() first")
            return None
            
        # Acquire rate limiting semaphore
        if not self.rate_limiter.acquire(blocking=True, timeout=30):
            logger.warning("Rate limit timeout - system overloaded")
            return None
        
        try:
            with self.circuit_breaker.call():
                return self._make_request_with_retry(prompt, model)
        except Exception as e:
            logger.error(f"Request failed completely: {e}")
            return None
        finally:
            # Controllo sicurezza anche nel finally
            if self.rate_limiter is not None:
                self.rate_limiter.release()
    
    def _make_request_with_retry(self, prompt: str, model: str) -> Optional[str]:
        """Make request with exponential backoff retry logic and improved load balancing"""
        
        if self.health_monitor is None:
            logger.error("Health monitor not initialized")
            return None
        
        service_unavailable_count = 0  # Counter per 503
        host_usage_count = {}  # Track usage per host for this request
        
        for attempt in range(1, Config.MAX_RETRIES_PER_REQUEST + 1):
            # Select host using round-robin for better load distribution
            healthy_hosts = self.health_monitor.get_healthy_hosts()
            
            if not healthy_hosts:
                logger.error("No healthy hosts available")
                # Try to recover all hosts
                for h in self.hosts:
                    self.health_monitor.check_health(h)
                
                healthy_hosts = self.health_monitor.get_healthy_hosts()
                if not healthy_hosts:
                    # If all hosts are down, wait and retry
                    if service_unavailable_count > 0:
                        logger.warning(f"All hosts down after {service_unavailable_count} 503 errors, waiting 2 minutes...")
                        time.sleep(120)
                        continue
                    raise Exception("All hosts are down")
            
            # Improved host selection: round-robin with fallback to performance-based
            host = None
            if hasattr(self.health_monitor, '_round_robin_index'):
                # Use round-robin if available
                with self.health_monitor._lock:
                    idx = self.health_monitor._round_robin_index % len(healthy_hosts)
                    host = healthy_hosts[idx]
                    self.health_monitor._round_robin_index += 1
            else:
                # Fallback to least used host in this request
                if host_usage_count:
                    # Find host with minimum usage in this request
                    min_usage = min(host_usage_count.values())
                    least_used_hosts = [h for h, count in host_usage_count.items() 
                                    if count == min_usage and h in healthy_hosts]
                    if least_used_hosts:
                        host = random.choice(least_used_hosts)
                
                # If still no host selected, use best performing one
                if not host:
                    host = self.health_monitor.get_best_host()
            
            if not host:
                logger.error("Failed to select any host")
                continue
                
            # Track host usage for this request
            host_usage_count[host] = host_usage_count.get(host, 0) + 1
            
            # Log host selection for debugging
            logger.debug(f"Attempt {attempt}: Selected host {host} (usage: {host_usage_count[host]})")
            
            try:
                # Prepare optimized payload
                payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "system", 
                            "content": "You are a JSON-only responder. Always output valid JSON."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    "stream": False,
                    "format": "json",
                    "options": {
                        "num_ctx": 2048,      
                        "num_predict": 256,   # Reduced for faster completion
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "num_thread": 32,     
                        "num_batch": 1024,    
                        "repeat_penalty": 1.1,
                        "stop": ["<|im_end|>", "<|endoftext|>"],  
                    }
                }
                
                start_time = time.time()
                resp = requests.post(
                    f"{host}/api/chat",
                    json=payload,
                    timeout=Config.REQUEST_TIMEOUT,
                    headers={'Content-Type': 'application/json'}
                )
                response_time = time.time() - start_time
                
                # Handle 503 Service Unavailable specifically
                if resp.status_code == 503:
                    service_unavailable_count += 1
                    logger.warning(f"503 Service Unavailable from {host} (count: {service_unavailable_count})")
                    
                    if service_unavailable_count <= Config.MAX_503_RETRIES:
                        wait_time = Config.RETRY_ON_503_WAIT * min(service_unavailable_count, 5)
                        logger.info(f"Waiting {wait_time}s for model to load on {host}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Too many 503 errors ({service_unavailable_count}) from {host}")
                        with self.health_monitor._lock:
                            self.health_monitor.health_status[host] = False
                        continue
                
                resp.raise_for_status()
                response_data = resp.json()
                
                if not response_data.get("done", False):
                    logger.warning(f"Incomplete response from {host}")
                    continue
                
                content = response_data.get("message", {}).get("content", "")
                if content:
                    stats.increment_processed()
                    logger.debug(f"SUCCESS: Got response from {host} in {response_time:.2f}s")
                    
                    # Update response time tracking for this host
                    with self.health_monitor._lock:
                        if host not in self.health_monitor.response_times:
                            self.health_monitor.response_times[host] = []
                        self.health_monitor.response_times[host].append(response_time)
                        if len(self.health_monitor.response_times[host]) > self.health_monitor._max_response_history:
                            self.health_monitor.response_times[host].pop(0)
                    
                    service_unavailable_count = 0  # Reset 503 counter on success
                    return content
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on {host} (attempt {attempt})")
                with self.health_monitor._lock:
                    self.health_monitor.health_status[host] = False
            except requests.exceptions.HTTPError as e:
                if "503" not in str(e):  # Log non-503 HTTP errors
                    logger.error(f"HTTP Error on {host}: {e}")
                with self.health_monitor._lock:
                    self.health_monitor.health_status[host] = False
                stats.increment_errors()
            except Exception as exc:
                logger.error(f"Unexpected error on {host}: {exc}")
                with self.health_monitor._lock:
                    self.health_monitor.health_status[host] = False
                stats.increment_errors()
            
            # Exponential backoff between retries, but shorter for 503 errors
            if attempt < Config.MAX_RETRIES_PER_REQUEST:
                if service_unavailable_count > 0:
                    # Shorter backoff for 503 errors since we already waited above
                    backoff_time = min(5, Config.BACKOFF_BASE ** (attempt - service_unavailable_count))
                else:
                    backoff_time = min(Config.BACKOFF_BASE ** attempt, Config.BACKOFF_MAX)
                
                logger.debug(f"Backing off for {backoff_time}s before retry {attempt + 1}")
                time.sleep(backoff_time)
        
        logger.error(f"All {Config.MAX_RETRIES_PER_REQUEST} attempts failed")
        return None
 
# ============= GEOGRAPHIC UTILITIES =============
def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance in kilometers between two geographic points using Haversine formula.
    
    The Haversine formula determines the great-circle distance between two points
    on a sphere given their longitudes and latitudes.
    
    Args:
        lat1, lon1: Latitude and longitude of first point
        lat2, lon2: Latitude and longitude of second point
        
    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth's radius in kilometers
    
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Calculate differences
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Haversine formula
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c

# ============= DATA LOADING FUNCTIONS =============
class DataLoader:
    """Handles loading and preprocessing of tourist visit data"""
    
    @staticmethod
    def load_pois(filepath: Path) -> DataFrame:
        """
        Load Points of Interest (POI) data with coordinates.
        
        Args:
            filepath: Path to POI CSV file
            
        Returns:
            DataFrame with columns: name_short, latitude, longitude
        """
        df = pd.read_csv(
            filepath, 
            usecols=["name_short", "latitude", "longitude"],
            dtype={
                "name_short": "category",
                "latitude": np.float32,
                "longitude": np.float32
            }
        )
        logger.info(f"Loaded {len(df)} POIs from {filepath.name}")
        return df
    
    @staticmethod
    def load_visits(filepath: Path) -> DataFrame:
        """
        Load tourist visit data and convert to standardized format.
        
        Args:
            filepath: Path to visits CSV file
            
        Returns:
            DataFrame with columns: timestamp, card_id, name_short
        """
        df = pd.read_csv(
            filepath,
            usecols=[0, 1, 2, 4],  # Select specific columns by position
            names=["data", "ora", "name_short", "card_id"],
            header=0,
            dtype={"card_id": "category", "name_short": "category"}
        )
        
        # Combine date and time into single timestamp
        df["timestamp"] = pd.to_datetime(
            df["data"] + " " + df["ora"], 
            format="%d-%m-%y %H:%M:%S"
        )
        
        logger.info(f"Loaded {len(df)} visits from {filepath.name}")
        
        # Return only needed columns, sorted by timestamp
        return (df[["timestamp", "card_id", "name_short"]]
                .sort_values("timestamp")
                .reset_index(drop=True))
    
    @staticmethod
    def merge_visits_pois(visits_df: DataFrame, pois_df: DataFrame) -> DataFrame:
        """
        Merge visits with POI data to filter out invalid visits.
        
        Args:
            visits_df: DataFrame with visit records
            pois_df: DataFrame with POI information
            
        Returns:
            DataFrame with only valid visits (matching POIs)
        """
        # Inner join keeps only visits to valid POIs
        merged = visits_df.merge(
            pois_df[["name_short"]], 
            on="name_short", 
            how="inner"
        )
        
        logger.info(f"Valid visits after merge: {len(merged)}")
        return merged.sort_values("timestamp").reset_index(drop=True)
    
    @staticmethod
    def filter_multi_visit_cards(df: DataFrame) -> DataFrame:
        """
        Filter to keep only cards that visited multiple distinct POIs.
        
        This ensures we have meaningful sequences for prediction.
        
        Args:
            df: DataFrame with visit records
            
        Returns:
            DataFrame with only multi-visit cards
        """
        # Count unique POIs per card
        unique_pois_per_card = df.groupby("card_id")["name_short"].nunique()
        
        # Keep cards with more than one unique POI
        valid_cards = unique_pois_per_card[unique_pois_per_card > 1].index
        
        logger.info(f"Multi-visit cards: {len(valid_cards)} / {df.card_id.nunique()}")
        
        return df[df["card_id"].isin(valid_cards)].reset_index(drop=True)
    
    @staticmethod
    def create_user_poi_matrix(df: DataFrame) -> DataFrame:
        """
        Create user-POI interaction matrix for clustering.
        
        Args:
            df: DataFrame with visit records
            
        Returns:
            Crosstab matrix of card_id x POI visits
        """
        return pd.crosstab(df["card_id"], df["name_short"])

# ============= PROMPT GENERATION =============
class PromptBuilder:
    """Handles prompt generation for LLM predictions"""
    
    @staticmethod
    def get_anchor_index(seq_len: int, rule: str | int, is_full_sequence: bool = False) -> int:
        """
        Determine anchor POI index based on specified rule.
        
        The anchor POI is used as the "current location" in the prompt.
        
        Args:
            seq_len: Length of sequence 
            rule: Selection strategy:
                - "penultimate": Last element of prefix (when is_full_sequence=False)
                - "first": Index 0
                - "middle": seq_len // 2 (when is_full_sequence=True, operates on full sequence)
                - int: Explicit index (negative allowed)
            is_full_sequence: Whether seq_len refers to full sequence (True) or prefix (False)
                
        Returns:
            0-based index of anchor POI
            
        Raises:
            ValueError: If rule is invalid or index out of range
        """
        if rule == "penultimate":
            if is_full_sequence:
                idx = seq_len - 2  # Second to last in full sequence
            else:
                idx = seq_len - 1  # Last in prefix
        elif rule == "first":
            idx = 0
        elif rule == "middle":
            if is_full_sequence:
                # For middle rule on full sequence, target will be element after anchor
                # So anchor should be at position that allows for a valid target
                idx = (seq_len - 1) // 2  # Ensures target index < seq_len
            else:
                idx = seq_len // 2
        elif isinstance(rule, int):
            idx = rule if rule >= 0 else seq_len + rule
        else:
            raise ValueError(f"Invalid anchor_rule: '{rule}'")
        
        max_idx = seq_len - 1 if is_full_sequence else seq_len - 1
        if not (0 <= idx < seq_len):
            raise ValueError(f"Anchor index {idx} out of range for sequence length {seq_len}")
        
        # For middle rule with full sequence, ensure there's space for target
        if rule == "middle" and is_full_sequence and idx >= seq_len - 1:
            raise ValueError(f"Middle anchor index {idx} doesn't allow for target in sequence of length {seq_len}")
            
        return idx
    
    @staticmethod
    def get_nearby_pois(
        current_poi: str, 
        pois_df: pd.DataFrame,
        visited_pois: List[str],
        max_pois: int = 10,
        max_distance: float = 5.0
    ) -> List[Dict[str, Any]]:
        """
        Find POIs near the current location.
        
        Args:
            current_poi: Name of current POI
            pois_df: DataFrame with all POIs
            visited_pois: List of already visited POIs to exclude
            max_pois: Maximum number of POIs to return
            max_distance: Maximum distance in km
            
        Returns:
            List of nearby POIs with distances
        """
        current_poi_row = pois_df[pois_df["name_short"] == current_poi]
        if current_poi_row.empty:
            return []
        
        current_lat = current_poi_row["latitude"].iloc[0]
        current_lon = current_poi_row["longitude"].iloc[0]
        
        nearby_pois = []
        
        for _, row in pois_df.iterrows():
            poi_name = row["name_short"]
            
            # Skip if already visited or is current POI
            if poi_name in visited_pois or poi_name == current_poi:
                continue
            
            distance = calculate_distance(
                current_lat, current_lon,
                row["latitude"], row["longitude"]
            )
            
            # Only include if within max distance
            if distance <= max_distance:
                nearby_pois.append({
                    "name": poi_name,
                    "distance": distance
                })
        
        # Sort by distance and limit results
        nearby_pois.sort(key=lambda x: x["distance"])
        return nearby_pois[:max_pois]
    
    @staticmethod
    def extract_temporal_features(visits_df: pd.DataFrame, card_id: str, anchor_idx: int) -> Dict[str, Any]:
        """
        Extract temporal features for enhanced context.

        Args:
            visits_df: DataFrame with visit records
            card_id: Card identifier
            anchor_idx: Index of anchor visit in sequence

        Returns:
            Dictionary with temporal features
        """
        visits = visits_df[visits_df["card_id"] == card_id].sort_values("timestamp")

        if len(visits) <= anchor_idx:
            return {}

        anchor_visit = visits.iloc[anchor_idx]
        timestamp = anchor_visit["timestamp"]

        # Extract time features
        hour = timestamp.hour
        day_name = timestamp.strftime("%A")
        is_weekend = timestamp.weekday() >= 5

        # Categorize time periods for tourism context
        if 6 <= hour <= 11:
            time_period = "Morning"
        elif 12 <= hour <= 17:
            time_period = "Afternoon"
        elif 18 <= hour <= 22:
            time_period = "Evening"
        else:
            time_period = "Late/Early"

        # Extract user's typical visit times (based on all visits)
        all_hours = visits["timestamp"].dt.hour.tolist()
        typical_hours = [h for h in all_hours if all_hours.count(h) > 1][:3]  # Top 3 frequent hours

        return {
            "day": day_name[:3],  # Mon, Tue, Wed etc.
            "time_period": time_period,
            "hour": hour,
            "minute": timestamp.minute,
            "is_weekend": is_weekend,
            "typical_hours": typical_hours,
            "timestamp": timestamp  # Keep full timestamp for seasonality
        }

    @staticmethod
    def compute_poi_peak_hours(df: pd.DataFrame) -> Dict[str, List[int]]:
        """
        Calculate top 3 peak hours for each POI.

        This should be called ONCE during initialization, before processing cards.

        Args:
            df: DataFrame with 'name_short' and 'timestamp' columns

        Returns:
            Dictionary mapping POI name to list of peak hours
            Example: {'Arena': [10, 11, 12], 'Casa Giulietta': [12, 13, 14], ...}
        """
        poi_peak_hours = {}

        for poi in df['name_short'].unique():
            poi_visits = df[df['name_short'] == poi]

            if len(poi_visits) > 0:
                # Get top 3 most frequent hours
                hour_counts = poi_visits['timestamp'].dt.hour.value_counts()
                top_hours = hour_counts.head(3).index.tolist()
                poi_peak_hours[poi] = sorted(top_hours)
            else:
                poi_peak_hours[poi] = []

        return poi_peak_hours

    @staticmethod
    def extract_seasonality_features(timestamp: pd.Timestamp) -> Dict[str, str]:
        """
        Extract seasonal and weekly patterns.

        Args:
            timestamp: Current timestamp

        Returns:
            Dictionary with 'season', 'tourist_intensity', 'day_type'
        """
        month = timestamp.month
        day_of_week = timestamp.dayofweek
        is_weekend = day_of_week >= 5

        # Seasonal classification (Northern Hemisphere)
        if month in [12, 1, 2]:
            season = "winter"
        elif month in [3, 4, 5]:
            season = "spring"
        elif month in [6, 7, 8]:
            season = "summer"
        else:
            season = "autumn"

        # Tourist intensity (Verona specific)
        if month in [7, 8, 12]:  # Peak season (summer holidays + Christmas)
            tourist_intensity = "high"
        elif month in [4, 5, 6, 9, 10]:  # Shoulder season
            tourist_intensity = "medium"
        else:  # January-March, November
            tourist_intensity = "low"

        # Day type
        day_type = "weekend" if is_weekend else "weekday"

        return {
            'season': season,
            'tourist_intensity': tourist_intensity,
            'day_type': day_type
        }

    @staticmethod
    def format_poi_timing_context(
        current_hour: int,
        nearby_pois: List[Dict[str, Any]],
        poi_peak_hours: Dict[str, List[int]],
        max_pois: int = 3
    ) -> str:
        """
        Generate timing hints for nearby POIs based on peak hours.

        Args:
            current_hour: Current hour of the day (0-23)
            nearby_pois: List of dicts with 'name' and 'distance' keys
            poi_peak_hours: Pre-computed peak hours dictionary
            max_pois: Maximum number of POIs to include

        Returns:
            Formatted string for prompt
            Example: "Timing: Duomo (peak now), Torre Lamberti (peak 15h)"
        """
        poi_timing = []

        for poi_info in nearby_pois[:max_pois]:
            poi_name = poi_info['name']
            peak_hours = poi_peak_hours.get(poi_name, [])

            if not peak_hours:
                continue

            # Check if current hour is a peak hour
            if current_hour in peak_hours:
                timing_hint = "peak now"
            else:
                # Find next peak hour
                future_peaks = [h for h in peak_hours if h > current_hour]
                if future_peaks:
                    next_peak = min(future_peaks)
                    timing_hint = f"peak {next_peak}h"
                else:
                    # Next peak is tomorrow (use first peak of the day)
                    next_peak = min(peak_hours)
                    timing_hint = f"peak {next_peak}h"

            poi_timing.append(f"{poi_name} ({timing_hint})")

        if poi_timing:
            return "Timing: " + ", ".join(poi_timing)
        else:
            return ""

    @staticmethod
    def create_prompt(
        df: pd.DataFrame,
        user_clusters: pd.DataFrame,
        cluster_preferences: Dict[int, List[str]],
        pois_df: pd.DataFrame,
        poi_peak_hours: Dict[str, List[int]],
        card_id: str,
        top_k: int = Config.TOP_K,
        anchor_rule: Union[str, int] = Config.DEFAULT_ANCHOR_RULE
    ) -> str:
        """
        Create optimized prompt for POI prediction.
        
        This method generates a concise prompt that includes:
        - User's cluster (tourist type)
        - Visit history
        - Current location
        - Nearby POIs with distances
        
        Args:
            df: Visit data
            user_clusters: Cluster assignments
            pois_df: POI information
            card_id: Card to predict for
            top_k: Number of predictions requested
            anchor_rule: Rule for selecting current POI
            
        Returns:
            Formatted prompt string
            
        Raises:
            ValueError: If sequence too short or POI not found
        """
        # Get visit sequence for this card
        visits = df[df["card_id"] == card_id].sort_values("timestamp")
        seq = visits["name_short"].tolist()
        
        if len(seq) < 3:
            raise ValueError("Sequence too short (minimum 3 visits required)")
        
        # Handle different logic for middle rule vs others
        if anchor_rule == "middle":
            # For middle rule: anchor is middle of full sequence, target is next element
            anchor_idx = PromptBuilder.get_anchor_index(len(seq), anchor_rule, is_full_sequence=True)
            current_poi = seq[anchor_idx]
            target = seq[anchor_idx + 1]  # Element immediately after anchor
            history = seq[:anchor_idx]  # Elements before anchor
        else:
            # Original logic for other rules
            target = seq[-1]  # Last visit (to predict)
            prefix = seq[:-1]  # All except last
            
            # Determine current POI using anchor rule
            anchor_idx = PromptBuilder.get_anchor_index(len(prefix), anchor_rule)
            current_poi = prefix[anchor_idx]
            history = [p for i, p in enumerate(prefix) if i != anchor_idx]
        
        # Get user's cluster
        cluster_id = user_clusters.loc[
            user_clusters["card_id"] == card_id, "cluster"
        ].values[0]

        # Extract temporal features
        temporal_info = PromptBuilder.extract_temporal_features(df, card_id, anchor_idx)

        # Get nearby POIs
        nearby_pois = PromptBuilder.get_nearby_pois(
            current_poi, pois_df, history, max_pois=10
        )

        # Format POI list with distances
        pois_list = ", ".join([
            f"{poi['name']} ({poi['distance']:.1f}km)"
            for poi in nearby_pois
        ])

        # Build temporal context string with Peak Hours and Seasonality
        temporal_context = ""
        if temporal_info:
            time_parts = []
            # Format: Mon 14:30 (usual 10h,14h,18h)
            time_parts.append(f"{temporal_info['day']} {temporal_info['hour']:02d}:{temporal_info['minute']:02d}")
            if temporal_info['typical_hours']:
                typical_str = ",".join([f"{h}h" for h in temporal_info['typical_hours']])
                time_parts.append(f"(usual {typical_str})")

            # Add seasonality context
            if 'timestamp' in temporal_info:
                seasonality = PromptBuilder.extract_seasonality_features(temporal_info['timestamp'])
                season_str = f"{seasonality['season'].capitalize()} {seasonality['day_type']}, {seasonality['tourist_intensity']} season"
                time_parts.append(f"[{season_str}]")

            temporal_context = f"Time: {' '.join(time_parts)}\n            "

            # Add POI peak hours timing
            poi_timing = PromptBuilder.format_poi_timing_context(
                temporal_info['hour'],
                nearby_pois,
                poi_peak_hours,
                max_pois=3
            )
            if poi_timing:
                temporal_context += f"{poi_timing}\n            "

        # Get cluster preferences and format as ranking (Strategy 4: Ranking Order)
        cluster_top_pois = cluster_preferences.get(cluster_id, [])
        cluster_ranking = " > ".join(cluster_top_pois[0]) if cluster_top_pois else "none"

        # Create enhanced prompt with temporal context and cluster preferences
        return f"""
            Tourist cluster {cluster_id} (top cluster preference: {cluster_ranking}).
            {temporal_context}Visited: {', '.join(history) if history else 'none'}
            Current: {current_poi}
            Nearby POIs: {pois_list}

            Suggest {top_k} most likely next POIs considering cluster preference, peak hours, seasonality, time and distances.
            Reply ONLY JSON with this format: {{"prediction": ["poi1", "poi2", ...], "reason": "brief explanation"}}"""

# ============= CHECKPOINT MANAGEMENT =============
class CheckpointManager:
    """Manages checkpoint files for resumable processing"""
    
    def __init__(self, visits_path: Path, out_dir: Path):
        self.visits_path = visits_path
        self.out_dir = out_dir
        self.checkpoint_file = out_dir / f"{visits_path.stem}_checkpoint.txt"
        self._completed_cards: Set[str] = set()
        self._load_checkpoint()
    
    def _load_checkpoint(self):
        """Load completed cards from checkpoint file"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    self._completed_cards = {line.strip() for line in f if line.strip()}
                logger.info(f"Loaded {len(self._completed_cards)} completed cards from checkpoint")
            except Exception as e:
                logger.warning(f"Error loading checkpoint: {e}")
                self._completed_cards = set()
    
    def is_completed(self, card_id: str) -> bool:
        """Check if a card has been processed"""
        return card_id in self._completed_cards
    
    def mark_completed(self, card_id: str):
        """Mark a card as completed and update checkpoint file"""
        self._completed_cards.add(card_id)
        try:
            with open(self.checkpoint_file, 'a') as f:
                f.write(f"{card_id}\n")
        except Exception as e:
            logger.warning(f"Error updating checkpoint: {e}")
    
    def get_completed_count(self) -> int:
        """Get number of completed cards"""
        return len(self._completed_cards)
    
    @staticmethod
    def should_skip_file(visits_path: Path, out_dir: Path, append: bool = False) -> bool:
        """
        Check if a file should be skipped (already fully processed).
        
        Args:
            visits_path: Path to visits file
            out_dir: Output directory
            append: Whether in append mode
            
        Returns:
            True if file should be skipped
        """
        if not append:
            return False
        
        checkpoint = CheckpointManager(visits_path, out_dir)
        completed_count = checkpoint.get_completed_count()
        
        if completed_count == 0:
            return False
        
        # Quick check - if we have many completed cards, it's likely done
        # For exact check, would need to load and process the file
        logger.info(f"File {visits_path.stem} has {completed_count} completed cards")
        
        # Conservative approach - don't skip unless explicitly verified
        return False

# ============= RESULTS MANAGEMENT =============
class ResultsManager:
    """Handles saving and managing prediction results"""
    
    def __init__(self, visits_path: Path, out_dir: Path, append: bool = False):
        self.visits_path = visits_path
        self.out_dir = out_dir
        self.append = append
        self.output_file = self._get_output_file()
        self.write_header = not (append and self.output_file.exists())
        self._buffer: List[Dict] = []
        self._write_lock = Lock()
    
    def _get_output_file(self) -> Path:
        """Determine output file path"""
        if self.append:
            # Look for existing output files
            pattern = f"{self.visits_path.stem}_pred_*.csv"
            existing_files = list(self.out_dir.glob(pattern))
            
            if existing_files:
                # Use the most recent file
                return max(existing_files, key=lambda p: p.stat().st_mtime)
        
        # Create new file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.out_dir / f"{self.visits_path.stem}_pred_{timestamp}.csv"
    
    def add_result(self, result: Dict):
        """Add a result to the buffer"""
        self._buffer.append(result)
        
        # Save if buffer is full
        if len(self._buffer) >= Config.BATCH_SAVE_INTERVAL:
            self.save_batch()
    
    def save_batch(self):
        """Save buffered results to file"""
        if not self._buffer:
            return
        
        with self._write_lock:
            try:
                df_batch = pd.DataFrame(self._buffer)
                
                mode = 'w' if self.write_header else 'a'
                df_batch.to_csv(
                    self.output_file,
                    mode=mode,
                    header=self.write_header,
                    index=False,
                    encoding='utf-8'
                )
                
                logger.debug(f"Saved batch of {len(self._buffer)} results")
                self.write_header = False
                self._buffer.clear()
                
            except Exception as e:
                logger.error(f"Error saving batch: {e}")
                # Create backup
                self._save_backup()
    
    def _save_backup(self):
        """Save backup of current buffer"""
        try:
            backup_file = (self.out_dir / 
                          f"backup_{self.visits_path.stem}_{int(time.time())}.json")
            with open(backup_file, 'w') as f:
                json.dump(self._buffer, f)
            logger.info(f"Backup saved to {backup_file}")
        except Exception as e:
            logger.error(f"Failed to save backup: {e}")
    
    def finalize(self):
        """Save any remaining results"""
        if self._buffer:
            self.save_batch()

# ============= CARD PROCESSING =============
class CardProcessor:
    """Processes individual tourist cards for POI prediction"""
    
    def __init__(
        self,
        filtered_df: DataFrame,
        user_clusters: DataFrame,
        cluster_preferences: Dict[int, List[str]],
        pois_df: DataFrame,
        poi_peak_hours: Dict[str, List[int]],
        ollama_manager: OllamaConnectionManager,
        checkpoint_manager: CheckpointManager,
        results_manager: ResultsManager
    ):
        self.filtered_df = filtered_df
        self.user_clusters = user_clusters
        self.cluster_preferences = cluster_preferences
        self.pois_df = pois_df
        self.poi_peak_hours = poi_peak_hours
        self.ollama_manager = ollama_manager
        self.checkpoint_manager = checkpoint_manager
        self.results_manager = results_manager
    
    def process_card(self, card_id: str) -> Optional[Dict]:
        """
        Process a single card to predict next POI visit.
        
        Args:
            card_id: Card identifier to process
            
        Returns:
            Dictionary with prediction results or None if error
        """
        start_time = time.time()
        
        try:
            # Skip if already processed
            if self.checkpoint_manager.is_completed(card_id):
                logger.debug(f"Card {card_id} already processed - skipping")
                return None
            
            # Get visit sequence
            seq = (self.filtered_df[self.filtered_df.card_id == card_id]
                   .sort_values("timestamp")["name_short"]
                   .tolist())
            
            if len(seq) < 3:
                logger.debug(f"Card {card_id} has insufficient visits ({len(seq)})")
                return None
            
            # Create prompt (which now handles the logic internally)
            try:
                prompt = PromptBuilder.create_prompt(
                    self.filtered_df,
                    self.user_clusters,
                    self.cluster_preferences,
                    self.pois_df,
                    self.poi_peak_hours,
                    card_id,
                    top_k=Config.TOP_K,
                    anchor_rule=Config.DEFAULT_ANCHOR_RULE
                )
            except Exception as e:
                logger.warning(f"Error creating prompt for {card_id}: {e}")
                return None
            
            # Get LLM prediction
            response = self.ollama_manager.get_chat_completion(prompt)
            
            # Extract sequence components based on anchor rule (for result record)
            if Config.DEFAULT_ANCHOR_RULE == "middle":
                # For middle rule: anchor is middle of full sequence, target is next element
                anchor_idx = PromptBuilder.get_anchor_index(len(seq), Config.DEFAULT_ANCHOR_RULE, is_full_sequence=True)
                current_poi = seq[anchor_idx]
                target = seq[anchor_idx + 1]  # Element immediately after anchor
                history_list = seq[:anchor_idx]  # Elements before anchor
            else:
                # Original logic for other rules
                target = seq[-1]
                prefix = seq[:-1]
                anchor_idx = PromptBuilder.get_anchor_index(len(prefix), Config.DEFAULT_ANCHOR_RULE)
                history_list = [p for i, p in enumerate(prefix) if i != anchor_idx]
                current_poi = prefix[anchor_idx]
            
            result = {
                "card_id": card_id,
                "cluster": self._get_user_cluster(card_id),
                "history": str(history_list),
                "current_poi": current_poi,
                "prediction": None,
                "ground_truth": target,
                "reason": None,
                "hit": False,
                "processing_time": time.time() - start_time,
                "status": "success" if response else "failed"
            }
            
            # Parse response if available
            if response:
                try:
                    parsed = json.loads(response)
                    predictions = parsed.get("prediction", [])
                    if not isinstance(predictions, list):
                        predictions = [predictions]
                    
                    result["prediction"] = str(predictions)
                    result["reason"] = parsed.get("reason", "")[:200]  # Limit length
                    result["hit"] = target in predictions
                    
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON response for {card_id}")
                    result["prediction"] = "PARSE_ERROR"
                    result["status"] = "parse_error"
                except Exception as e:
                    logger.warning(f"Error parsing response for {card_id}: {e}")
                    result["prediction"] = "PROCESSING_ERROR"
                    result["status"] = "processing_error"
            
            # Mark as completed and save result
            self.checkpoint_manager.mark_completed(card_id)
            self.results_manager.add_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Fatal error processing card {card_id}: {e}")
            return {
                "card_id": card_id,
                "cluster": None,
                "history": None,
                "current_poi": None,
                "prediction": "FATAL_ERROR",
                "ground_truth": None,
                "reason": str(e)[:200],
                "hit": False,
                "processing_time": time.time() - start_time,
                "status": "fatal_error"
            }
    
    def _get_user_cluster(self, card_id: str) -> Optional[int]:
        """Get cluster ID for a card"""
        try:
            return int(self.user_clusters[
                self.user_clusters.card_id == card_id
            ]["cluster"].iloc[0])
        except Exception:
            return None

# ============= MAIN PROCESSING PIPELINE =============
class VisitFileProcessor:
    """Orchestrates the complete processing pipeline for a visits file"""
    
    def __init__(self, ollama_manager: OllamaConnectionManager):
        if ollama_manager.rate_limiter is None:
            raise ValueError("OllamaConnectionManager not properly initialized - rate_limiter is None")
        if ollama_manager.health_monitor is None:
            raise ValueError("OllamaConnectionManager not properly initialized - health_monitor is None")
        if not ollama_manager.hosts:
            raise ValueError("OllamaConnectionManager has no hosts configured")
            
        self.ollama_manager = ollama_manager
        Config.RESULTS_DIR.mkdir(exist_ok=True)
    
    def process_file(
        self,
        visits_path: Path,
        poi_path: Path,
        max_users: Optional[int] = None,
        force: bool = False,
        append: bool = False
    ) -> None:
        """
        Process a single visits file to generate POI predictions.
        
        This method orchestrates the complete pipeline:
        1. Load and preprocess data
        2. Perform clustering
        3. Process cards in parallel
        4. Save results with checkpointing
        
        Args:
            visits_path: Path to visits CSV file
            poi_path: Path to POI CSV file
            max_users: Maximum number of users to process (None for all)
            force: Force reprocessing even if output exists
            append: Resume from previous run
        """
        # Check if file should be skipped
        if not force and CheckpointManager.should_skip_file(
            visits_path, Config.RESULTS_DIR, append
        ):
            logger.info(f"Skipping {visits_path.name} - already processed")
            return
        
        logger.info(f"Processing {visits_path.name}")
        
        # Initialize managers
        checkpoint_manager = CheckpointManager(visits_path, Config.RESULTS_DIR)
        results_manager = ResultsManager(visits_path, Config.RESULTS_DIR, append)
        
        try:
            # Load and preprocess data
            logger.info("Loading and preprocessing data...")
            pois_df = DataLoader.load_pois(poi_path)
            visits_df = DataLoader.load_visits(visits_path)
            merged_df = DataLoader.merge_visits_pois(visits_df, pois_df)
            filtered_df = DataLoader.filter_multi_visit_cards(merged_df)
            
            # Perform clustering
            logger.info("Performing user clustering...")
            user_poi_matrix = DataLoader.create_user_poi_matrix(filtered_df)
            
            # K-means clustering with standardization
            scaler = StandardScaler()
            scaled_matrix = scaler.fit_transform(user_poi_matrix)
            
            # Run clustering and keep model to extract centroids
            kmeans_model = KMeans(
                n_clusters=7,
                random_state=42,
                n_init=10
            )
            clusters = kmeans_model.fit_predict(scaled_matrix)

            # Extract centroids and create cluster preferences (Strategy 4: Ranking Order)
            centroids_scaled = kmeans_model.cluster_centers_
            centroids_original = scaler.inverse_transform(centroids_scaled)

            centroids_df = pd.DataFrame(
                centroids_original,
                columns=user_poi_matrix.columns,
                index=range(7)
            )

            # Create cluster preferences dictionary (top 5 POI per cluster)
            cluster_preferences = {}
            for cluster_id in range(7):
                top_pois = centroids_df.iloc[cluster_id].nlargest(5).index.tolist()
                cluster_preferences[cluster_id] = top_pois

            logger.info("Cluster preferences extracted:")
            for cid, pois in cluster_preferences.items():
                logger.info(f"  Cluster {cid}: {' > '.join(pois)}")

            # Pre-compute POI peak hours
            logger.info("Computing POI peak hours...")
            poi_peak_hours = PromptBuilder.compute_poi_peak_hours(filtered_df)
            logger.info(f"Peak hours computed for {len(poi_peak_hours)} POIs")

            # Log sample peak hours
            sample_pois = list(poi_peak_hours.keys())[:5]
            for poi in sample_pois:
                logger.info(f"  {poi}: peak hours = {poi_peak_hours[poi]}")

            user_clusters = pd.DataFrame({
                "card_id": user_poi_matrix.index,
                "cluster": clusters
            })

            # Select cards to process
            eligible_cards = self._get_eligible_cards(filtered_df)
            
            if max_users is not None:
                cards_to_process = random.sample(
                    eligible_cards, 
                    min(max_users, len(eligible_cards))
                )
            else:
                cards_to_process = eligible_cards
            
            # Filter out already processed cards if in append mode
            if append:
                cards_to_process = [
                    card for card in cards_to_process 
                    if not checkpoint_manager.is_completed(card)
                ]
            
            logger.info(f"Processing {len(cards_to_process)} cards")
            
            if not cards_to_process:
                logger.info("No cards to process")
                return
            
            # Create card processor
            card_processor = CardProcessor(
                filtered_df,
                user_clusters,
                cluster_preferences,
                pois_df,
                poi_peak_hours,
                self.ollama_manager,
                checkpoint_manager,
                results_manager
            )
            
            # Process cards in parallel
            self._process_cards_parallel(card_processor, cards_to_process)
            
            # Finalize results
            results_manager.finalize()
            
            # Log summary statistics
            success_rate = stats.get_success_rate()
            logger.info(f"Completed processing {visits_path.name}")
            logger.info(f"Success rate: {success_rate:.1f}%")
            
        except Exception as e:
            logger.error(f"Error processing {visits_path.name}: {e}")
            raise
    
    def _get_eligible_cards(self, filtered_df: DataFrame) -> List[str]:
        """Get cards with sufficient visits for prediction"""
        card_visit_counts = filtered_df.groupby("card_id").size()
        eligible = card_visit_counts[card_visit_counts >= 3].index.tolist()
        return eligible
    
    def _process_cards_parallel(
        self, 
        card_processor: CardProcessor, 
        cards_to_process: List[str]
    ) -> None:
        """Process cards in parallel with progress tracking"""
        
        # Calculate optimal number of workers
        n_healthy_hosts = len(self.ollama_manager.health_monitor.get_healthy_hosts())
    
        # OPTIMIZED: Use full concurrency potential
        optimal_workers = min(Config.MAX_CONCURRENT_REQUESTS, len(cards_to_process))  # Use global limit instead of per-GPU
        
        logger.info(f"Using {optimal_workers} workers for {n_healthy_hosts} healthy hosts")
        
        # OPTIMIZED: Reduced stabilization time for faster startup
        logger.info("Waiting 60s for model stabilization...")
        time.sleep(60)  
        
        # ✅ NUOVO: Test pre-processing per verificare che tutto sia OK
        logger.info("Running pre-flight check on all hosts...")
        for host in self.ollama_manager.hosts:
            try:
                test_resp = requests.post(
                    f"{host}/api/chat",
                    json={
                        "model": Config.MODEL_NAME,
                        "messages": [{"role": "user", "content": "test"}],
                        "stream": False,
                        "options": {"num_predict": 1, "temperature": 0}
                    },
                    timeout=60
                )
                if test_resp.status_code == 200:
                    logger.info(f"✅ Pre-flight check passed for {host}")
                else:
                    logger.warning(f"⚠️ Pre-flight check failed for {host}: {test_resp.status_code}")
            except Exception as e:
                logger.warning(f"Pre-flight check error for {host}: {e}")
        
        # Process cards with thread pool
        with ThreadPoolExecutor(
            max_workers=optimal_workers,
            thread_name_prefix="CardWorker"
        ) as executor:
            
            # Submit all tasks
            futures = {
                executor.submit(card_processor.process_card, card_id): card_id
                for card_id in cards_to_process
            }
            
            # Process results with progress bar
            with tqdm(
                total=len(cards_to_process),
                desc="Processing cards",
                unit="card"
            ) as pbar:
                
                for future in as_completed(futures):
                    card_id = futures[future]
                    
                    try:
                        result = future.result(timeout=Config.REQUEST_TIMEOUT + 60)
                        
                        if result and result.get('status') == 'fatal_error':
                            logger.warning(f"Fatal error for card {card_id}")
                        
                        # Check circuit breaker
                        if stats.consecutive_failures >= Config.CIRCUIT_BREAKER_THRESHOLD:
                            logger.error("Too many consecutive failures - pausing")
                            time.sleep(60)  # Pause for 1 minute
                            stats.reset_consecutive_failures()
                    
                    except TimeoutError:
                        logger.error(f"Timeout processing card {card_id}")
                    except Exception as e:
                        logger.error(f"Error processing card {card_id}: {e}")
                    
                    pbar.update(1)
    
    def process_all_files(
        self,
        max_users: Optional[int] = None,
        force: bool = False,
        append: bool = False,
        single_file: Optional[str] = None
    ) -> None:
        """
        Process all visit files or a single specified file.
        
        Args:
            max_users: Maximum users per file
            force: Force reprocessing
            append: Resume from previous runs
            single_file: Process only this file (if specified)
        """
        poi_path = Config.POI_FILE
        
        if not poi_path.exists():
            raise RuntimeError(f"POI file not found: {poi_path}")
        
        if single_file:
            # Process single file
            target_path = self._resolve_file_path(single_file)
            self.process_file(target_path, poi_path, max_users, force, append)
        else:
            # Process all visit files
            visit_files = self._find_visit_files()
            
            if not visit_files:
                raise RuntimeError("No visit files found")
            
            logger.info(f"Found {len(visit_files)} files to process")
            
            processed = 0
            skipped = 0
            
            for visit_file in sorted(visit_files):
                try:
                    if not force and CheckpointManager.should_skip_file(
                        visit_file, Config.RESULTS_DIR, append
                    ):
                        skipped += 1
                        continue
                    
                    self.process_file(visit_file, poi_path, max_users, force, append)
                    processed += 1
                    
                except Exception as e:
                    logger.error(f"Error processing {visit_file.name}: {e}")
                    continue
            
            # Summary statistics
            logger.info("\n" + "=" * 70)
            logger.info("PROCESSING SUMMARY:")
            logger.info(f"  Total files: {len(visit_files)}")
            logger.info(f"  Processed: {processed}")
            logger.info(f"  Skipped: {skipped}")
            logger.info(f"  Efficiency: {skipped/len(visit_files)*100:.1f}% files avoided")
            logger.info("=" * 70)
    
    def _find_visit_files(self) -> List[Path]:
        """Find all visit CSV files (excluding POI file)"""
        visit_files = []
        
        for csv_path in Config.DATA_DIR.rglob("*.csv"):
            # Skip POI file and backup files
            if (csv_path.name.lower() != "vc_site.csv" and 
                "backup" not in str(csv_path).lower()):
                visit_files.append(csv_path)
        
        return visit_files
    
    def _resolve_file_path(self, file_path: str) -> Path:
        """Resolve file path from string input"""
        target = Path(file_path)
        
        # Try different path resolutions
        if not target.is_absolute():
            if not target.exists():
                # Try relative to data directory
                target = Config.DATA_DIR / file_path
                if not target.exists():
                    # Try just filename in data directory
                    target = Config.DATA_DIR / Path(file_path).name
        
        if not target.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if target.suffix.lower() != '.csv':
            raise ValueError(f"File must be CSV: {target}")
        
        if target.name.lower() == 'vc_site.csv':
            raise ValueError("Cannot process POI file")
        
        return target

# ============= MAIN ENTRY POINT =============
def main():
    """Main entry point for the application"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="VeronaCard Tourist Behavior Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Process all files:
    python %(prog)s
    
  Process with max 100 users per file:
    python %(prog)s --max-users 100
    
  Process single file:
    python %(prog)s --file data/verona/visits_2014.csv
    
  Resume from previous run:
    python %(prog)s --append
    
  Force reprocessing:
    python %(prog)s --force
        """
    )
    
    parser.add_argument(
        "--file",
        type=str,
        help="Process only this specific file"
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=None,
        help="Maximum number of users to process per file"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if output exists"
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Resume from previous run (append mode)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.force and args.append:
        parser.error("Cannot use both --force and --append")
    
    # Setup GPU environment if not already set
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Use all 4 A100 GPUs
    
    try:
        # Inizializzazione passo-passo con controlli
        logger.info("Initializing Ollama connection manager...")
        ollama_manager = OllamaConnectionManager()
        
        # Setup connections DEVE essere chiamato
        logger.info("Setting up connections...")
        hosts = ollama_manager.setup_connections()
        if not hosts:
            raise RuntimeError("No hosts configured")
            
        # Verifica inizializzazione corretta
        if ollama_manager.rate_limiter is None:
            raise RuntimeError("Rate limiter not properly initialized")
        if ollama_manager.health_monitor is None:
            raise RuntimeError("Health monitor not properly initialized")
            
        logger.info(f"Initialized with {len(hosts)} hosts")
        
        # Wait for services to be ready
        if not ollama_manager.wait_for_services():
            raise RuntimeError("Ollama services failed to start")
        
        # Create and run processor
        processor = VisitFileProcessor(ollama_manager)
        
        processor.process_all_files(
            max_users=args.max_users,
            force=args.force,
            append=args.append,
            single_file=args.file
        )
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()