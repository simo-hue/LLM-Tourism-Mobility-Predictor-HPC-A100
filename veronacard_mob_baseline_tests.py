# ============= IMPORT =============
import argparse
import json
import logging
import math
import os
import random
import sys
import threading
from threading import RLock
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set, Union

# Third-party imports
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ============= CONFIGURATION =============
class Config:
    """Centralized configuration to avoid global variables"""
    
    # Model configuration
    TOP_K = 5  # Number of POI predictions
    
    # Optimization parameters
    MAX_CONCURRENT_WORKERS = max(1, os.cpu_count() // 2) # Usa metà delle CPU
    BATCH_SAVE_INTERVAL = 1000  # Save results every N cards
    
    # Anchor rule for POI selection
    DEFAULT_ANCHOR_RULE = "middle"
    
    # File paths
    LOG_DIR = Path(__file__).resolve().parent / "logs"
    # NOTA: Aggiorna questo percorso se vuoi separare i risultati
    RESULTS_DIR = Path(__file__).resolve().parent / f"results/{DEFAULT_ANCHOR_RULE}/baseline_heuristics"
    DATA_DIR = Path(__file__).resolve().parent / "data" / "verona"
    POI_FILE = DATA_DIR / "vc_site.csv"

# ============= STRATEGY PREDICTOR =============
class StrategyPredictor:
    """
    Contiene la logica per le strategie di predizione euristiche.
    """
    
    def __init__(self, pois_df: DataFrame, cluster_preferences: Dict[int, List[str]], absolute_popularity_list: List[str]):
        self.pois_df = pois_df
        self.all_poi_names = set(pois_df['name_short'].unique())
        self.cluster_preferences = cluster_preferences
        self.absolute_popularity_list = absolute_popularity_list
        logger.info(f"StrategyPredictor initialized with {len(self.all_poi_names)} POIs.")

    def predict(
        self,
        strategy: str,
        current_poi: str,
        cluster_id: int,
        visited_pois: List[str],
        top_k: int
    ) -> Tuple[List[str], str]:
        """
        Esegue la logica di predizione basata sulla strategia.
        
        Returns:
            Una tupla (predictions_list, reason_string)
        """
        start_time = time.time()
        
        # Set di POI da escludere (visitati + corrente)
        exclude_set = set(visited_pois)
        exclude_set.add(current_poi)
        
        # Calcola POI disponibili
        available_pois = self.all_poi_names - exclude_set
        if not available_pois:
            logger.warning("No available POIs left to predict.")
            return [], f"strategy_{strategy}_no_options"

        predictions = []
        reason = f"strategy_{strategy}"

        try:
            if strategy == 'random':
                # --- 1. Strategia RANDOM ---
                predictions = random.sample(
                    list(available_pois), 
                    min(top_k, len(available_pois))
                )
                
            elif strategy == 'nearest':
                # --- 2. Strategia NEAREST ---
                # Usiamo la logica esistente di PromptBuilder
                nearby_pois_dicts = PromptBuilder.get_nearby_pois(
                    current_poi, 
                    self.pois_df, 
                    list(exclude_set), # Passa tutti i POI da escludere
                    max_pois=top_k
                )
                predictions = [poi['name'] for poi in nearby_pois_dicts]

            elif strategy == 'popular':
                # --- 3. Strategia POPULAR ---
                # Prendi le preferenze per questo cluster
                popular_for_cluster = self.cluster_preferences.get(cluster_id, [])
                
                # Filtra per quelli disponibili e non visitati
                predictions = [
                    poi for poi in popular_for_cluster 
                    if poi in available_pois
                ]
                
                # Assicurati di avere il numero giusto di predizioni
                if len(predictions) < top_k:
                    # Se non bastano, riempi con POI a caso tra quelli disponibili
                    remaining_needed = top_k - len(predictions)
                    remaining_available = list(available_pois - set(predictions))
                    
                    if remaining_available:
                        predictions.extend(random.sample(
                            remaining_available,
                            min(remaining_needed, len(remaining_available))
                        ))
                
                predictions = predictions[:top_k]
            
            elif strategy == 'absolute_top_k':
                # Filtra la nostra lista di popolarità globale pre-calcolata,
                # tenendo solo i POI che sono in 'available_pois'
                # (available_pois esclude già i POI visitati e quello corrente)
                predictions = [
                    poi for poi in self.absolute_popularity_list
                    if poi in available_pois
                ]
                
                # Prendi i primi top_k risultati
                predictions = predictions[:top_k]
            
            elif strategy == 'absolute':
                # Filtra la nostra lista di popolarità globale pre-calcolata,
                # tenendo solo i POI che sono in 'available_pois'
                # (available_pois esclude già i POI visitati e quello corrente)
                predictions = [
                    poi for poi in self.absolute_popularity_list
                    if poi in available_pois
                ]
                
                # Prendi il primo
                predictions = predictions[0]
            
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
                
            processing_time = (time.time() - start_time) * 1000 # in ms
            logger.debug(f"Strategy '{strategy}' computed in {processing_time:.2f} ms")
            
            return predictions, reason

        except Exception as e:
            logger.error(f"Error during strategy '{strategy}' prediction: {e}")
            return [], f"strategy_{strategy}_error"

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
        unique_pois_per_card = df.groupby("card_id", observed=True)["name_short"].nunique()
        
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
    """
    Classe ridotta a funzioni utility per anchor e geo-distanza.
    """
    
    @staticmethod
    def get_anchor_index(seq_len: int, rule: str | int, is_full_sequence: bool = False) -> int:
        """
        Determina l'indice dell'anchor POI (logica invariata).
        ... (il contenuto di questa funzione rimane IDENTICO a prima) ...
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
                idx = (seq_len - 1) // 2  
            else:
                idx = seq_len // 2
        elif isinstance(rule, int):
            idx = rule if rule >= 0 else seq_len + rule
        else:
            raise ValueError(f"Invalid anchor_rule: '{rule}'")
        
        max_idx = seq_len - 1 if is_full_sequence else seq_len - 1
        if not (0 <= idx < seq_len):
            raise ValueError(f"Anchor index {idx} out of range for sequence length {seq_len}")
        
        if rule == "middle" and is_full_sequence and idx >= seq_len - 1:
            raise ValueError(f"Middle anchor index {idx} doesn't allow for target in sequence of length {seq_len}")
            
        return idx

    @staticmethod
    def get_nearby_pois(
        current_poi: str, 
        pois_df: pd.DataFrame,
        visited_pois_to_exclude: List[str], # Nome parametro aggiornato
        max_pois: int = 10,
        max_distance: float = 5.0
    ) -> List[Dict[str, Any]]:
        """
        Trova POI vicini (logica quasi invariata).
        Usato dalla strategia 'nearest'.
        """
        current_poi_row = pois_df[pois_df["name_short"] == current_poi]
        if current_poi_row.empty:
            return []
        
        current_lat = current_poi_row["latitude"].iloc[0]
        current_lon = current_poi_row["longitude"].iloc[0]
        
        nearby_pois = []
        exclude_set = set(visited_pois_to_exclude) # Usa il set per efficienza
        
        for _, row in pois_df.iterrows():
            poi_name = row["name_short"]
            
            # Skip se già visitato O è il POI corrente
            if poi_name in exclude_set:
                continue
            
            distance = calculate_distance(
                current_lat, current_lon,
                row["latitude"], row["longitude"]
            )
            
            if distance <= max_distance:
                nearby_pois.append({
                    "name": poi_name,
                    "distance": distance
                })
        
        nearby_pois.sort(key=lambda x: x["distance"])
        return nearby_pois[:max_pois]

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
        self._write_lock = RLock()
    
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
        """Add a result to the buffer (thread-safe)"""
        # --- AGGIUNGI QUESTO BLOCCO ---
        with self._write_lock:
            self._buffer.append(result)
            
            # Save if buffer is full
            if len(self._buffer) >= Config.BATCH_SAVE_INTERVAL:
                # Questo ora è sicuro perché RLock permette chiamate annidate
                self.save_batch()

    def save_batch(self):
        """Save buffered results to file"""
        # (Questa funzione è già corretta, 
        #  perché ha già "with self._write_lock:")
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
        with self._write_lock:
            if self._buffer:
                self.save_batch()

# ============= CARD PROCESSING =============
class CardProcessor:
    """Processes individual tourist cards for POI prediction"""
    
    def __init__(
        self,
        filtered_df: DataFrame,
        user_clusters: DataFrame,
        predictor: StrategyPredictor, # NUOVO
        strategy: str, # NUOVO
        checkpoint_manager: CheckpointManager,
        results_manager: ResultsManager
        # RIMOSSO: cluster_preferences, pois_df, poi_peak_hours, ollama_manager
    ):
        self.filtered_df = filtered_df
        self.user_clusters = user_clusters
        self.predictor = predictor
        self.strategy = strategy
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
            
            cluster_id = self._get_user_cluster(card_id)
            if cluster_id is None:
                logger.warning(f"Card {card_id} has no cluster. Skipping.")
                return None

            predictions_list, reason_str = self.predictor.predict(
                strategy=self.strategy,
                current_poi=current_poi,
                cluster_id=cluster_id,
                visited_pois=history_list,
                top_k=Config.TOP_K
            )
            
            hit = target in predictions_list
            
            result = {
                "card_id": card_id,
                "cluster": cluster_id,
                "history": str(history_list),
                "current_poi": current_poi,
                "prediction": str(predictions_list), # Salva la lista
                "ground_truth": target,
                "reason": reason_str, # Salva la strategia usata
                "hit": hit,
                "processing_time": time.time() - start_time,
                "status": "success" # È sempre successo se non lancia eccezioni
            }
            
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
    
    def __init__(self):
        # RIMOSSO: ollama_manager
        # Creiamo la directory dei risultati (aggiornata in Config)
        Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Results will be saved to: {Config.RESULTS_DIR}")
        
    def process_file(
        self,
        visits_path: Path,
        poi_path: Path,
        strategy: str,
        max_users: Optional[int] = None,
        force: bool = False,
        append: bool = False
    ) -> None:
        """
        P
        """
        # 1. Definisci la directory di strategia SUBITO
        strategy_results_dir = Config.RESULTS_DIR / strategy
        strategy_results_dir.mkdir(parents=True, exist_ok=True)

        # 2. Usa la directory corretta per il CONTROLLO SKIP
        if not force and CheckpointManager.should_skip_file(
            visits_path, strategy_results_dir, append 
        ):
            logger.info(f"Skipping {visits_path.name} for strategy '{strategy}' - already processed")
            return

        logger.info(f"Processing {visits_path.name} with strategy '{strategy}'")

        # 3. Usa la directory corretta per i MANAGER
        checkpoint_manager = CheckpointManager(visits_path, strategy_results_dir)
        results_manager = ResultsManager(visits_path, strategy_results_dir, append)
        
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

            logger.info("Cluster preferences extracted (for 'popular' strategy).")
           
           # Calclo valori globali di popolarità
            logger.info("Calculating absolute POI popularity...")
            # Calcoliamo la popolarità globale da tutti i visite filtrate
            absolute_poi_popularity = filtered_df['name_short'].value_counts().index.tolist()
            logger.info(f"Top 3 absolute POIs: {absolute_poi_popularity[:3]}")
           
            user_clusters = pd.DataFrame({
                "card_id": user_poi_matrix.index,
                "cluster": clusters
            })
            
            predictor = StrategyPredictor(pois_df, cluster_preferences, absolute_poi_popularity)

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
                predictor, # Passa il nuovo predictor
                strategy,  # Passa la strategia scelta
                checkpoint_manager,
                results_manager
                # RIMOSSO: cluster_preferences (ora è dentro il predictor)
                # RIMOSSO: pois_df (ora è dentro il predictor)
                # RIMOSSO: poi_peak_hours
                # RIMOSSO: ollama_manager
            )
            
            # Process cards in parallel
            self._process_cards_parallel(card_processor, cards_to_process)
            
            # Finalize results
            results_manager.finalize()
            
            # Log summary statistics
            logger.info(f"Completed processing {visits_path.name} for strategy '{strategy}'")
        except Exception as e:
            logger.error(f"Error processing {visits_path.name}: {e}")
            raise
    
    def _get_eligible_cards(self, filtered_df: DataFrame) -> List[str]:
        """Get cards with sufficient visits for prediction"""
        card_visit_counts = filtered_df.groupby("card_id", observed=True).size()
        eligible = card_visit_counts[card_visit_counts >= 3].index.tolist()
        return eligible

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
    
    def _process_cards_parallel(
        self, 
        card_processor: CardProcessor, 
        cards_to_process: List[str]
    ) -> None:
        """Process cards in parallel with progress tracking"""
        
        # Usa la nuova config basata su CPU
        optimal_workers = min(Config.MAX_CONCURRENT_WORKERS, len(cards_to_process))
        
        logger.info(f"Using {optimal_workers} CPU workers")
        
        # RIMOSSO: Attesa stabilizzazione modello
        # RIMOSSO: Pre-flight check
        
        # Process cards with thread pool
        with ThreadPoolExecutor(
            max_workers=optimal_workers,
            thread_name_prefix="CardWorker"
        ) as executor:
            
            futures = {
                executor.submit(card_processor.process_card, card_id): card_id
                for card_id in cards_to_process
            }
            
            with tqdm(
                total=len(cards_to_process),
                desc=f"Processing cards ({card_processor.strategy})", # Aggiungi strategia
                unit="card"
            ) as pbar:
                
                for future in as_completed(futures):
                    card_id = futures[future]
                    
                    try:
                        # Il timeout non è più critico, ma lo teniamo per sicurezza
                        result = future.result(timeout=60) 
                        
                        if result and result.get('status') == 'fatal_error':
                            logger.warning(f"Fatal error for card {card_id}")
                        
                        # RIMOSSO: Controllo Circuit Breaker / consecutive_failures
                    
                    except TimeoutError:
                        logger.error(f"Timeout processing card {card_id} (CPU task)")
                    except Exception as e:
                        logger.error(f"Error processing card {card_id}: {e}")
                    
                    pbar.update(1)
                   
    def process_all_files(
        self,
        strategy: str,
        max_users: Optional[int] = None,
        force: bool = False,
        append: bool = False,
        single_file: Optional[str] = None
    ) -> None:
        """
        Process all visit files or a single specified file.
        
        """
        poi_path = Config.POI_FILE
        
        strategy_results_dir = Config.RESULTS_DIR / strategy
        
        if not poi_path.exists():
            raise RuntimeError(f"POI file not found: {poi_path}")
        
        if single_file:
            target_path = self._resolve_file_path(single_file)
            self.process_file(target_path, poi_path, strategy, max_users, force, append)
        else:
            visit_files = self._find_visit_files()
            
            if not visit_files:
                raise RuntimeError("No visit files found")
            
            logger.info(f"Found {len(visit_files)} files to process")
            
            processed = 0
            skipped = 0
            
            for visit_file in sorted(visit_files):
                try:
                    # Passiamo la directory giusta a CheckpointManager
                    if not force and CheckpointManager.should_skip_file(
                        visit_file, strategy_results_dir, append
                    ):
                        skipped += 1
                        continue
                    
                    # Passa la strategia
                    self.process_file(visit_file, poi_path, strategy, max_users, force, append)
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

# ============= MAIN ENTRY POINT =============
def main():
    """Main entry point for the application"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="VeronaCard POI Prediction Baseline Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Run 'nearest' strategy on all files:
    python %(prog)s --strategy nearest
    
  Run 'random' strategy with max 100 users per file:
    python %(prog)s --strategy random --max-users 100
    
  Run 'popular' strategy on a single file:
    python %(prog)s --strategy popular --file data/verona/visits_2014.csv
        """
    )
    
    # NUOVO: Argomento Strategy (obbligatorio)
    parser.add_argument(
        "--strategy",
        type=str,
        required=True, # Rendiamo la strategia obbligatoria
        choices=['random', 'nearest', 'popular', 'absolute'], # Valori consentiti
        help="The prediction strategy to use: random, nearest, popular"
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

    try:
        # Log della strategia scelta
        logger.info(f"Starting baseline test with strategy: {args.strategy.upper()}")
        
        # Crea e avvia il processore (ora molto più semplice)
        processor = VisitFileProcessor()
        
        processor.process_all_files(
            strategy=args.strategy, # Passiamo la strategia
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
        # Aggiungi traceback per un debug migliore
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()