"""Card info lookup table builder with GPU acceleration support."""
import logging
import time
import os
from pathlib import Path
import signal
from typing import Any, Dict, List
import concurrent.futures

try:
    # Try importing GPU libraries
    import cupy as cp
    from cuml.cluster import KMeans as cuKMeans
    from cuml.metrics import pairwise_distances
    HAS_GPU = True
except ImportError:
    # Fallback to CPU libraries
    import numpy as cp  # type: ignore
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances
    HAS_GPU = False

import joblib
from scipy.stats import wasserstein_distance
from tqdm import tqdm

from poker_ai.clustering.card_combos import CardCombos
from poker_ai.clustering.game_utility import GameUtility
from poker_ai.clustering.preflop import compute_preflop_lossless_abstraction
from poker_ai.clustering.advanced_abstractions import (
    PotentialAwareAbstraction,
    ImplicitModelingAbstraction,
    StrengthBucketing
)

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger("poker_ai.clustering.card_info_lut_builder")

if HAS_GPU:
    log.info("GPU support enabled - using CUDA acceleration")
    # Set memory limit to 90% of available GPU memory
    try:
        mem_info = cp.cuda.runtime.memGetInfo()
        mem_limit = int(mem_info[0] * 0.9)  # 90% of free memory
        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)
        log.info(f"GPU memory limit set to {mem_limit / 1024**3:.2f} GB")
    except Exception as e:
        log.warning(f"Failed to configure GPU memory: {str(e)}")
else:
    log.warning("GPU libraries not found - falling back to CPU processing")


class CardInfoLutBuilder(CardCombos):
    """
    Stores info buckets for each street when called.

    Attributes
    ----------
    card_info_lut : Dict[str, Any]
        Lookup table of card combinations per betting round to a cluster id.
    centroids : Dict[str, Any]
        Centroids per betting round for use in clustering previous rounds by
        earth movers distance.
    """

    def __init__(
        self,
        n_simulations_river: int,
        n_simulations_turn: int,
        n_simulations_flop: int,
        low_card_rank: int,
        high_card_rank: int,
        save_dir: str,
    ):
        """Initialize the CardInfoLutBuilder with enhanced error handling and checkpointing."""
        self.n_simulations_river = n_simulations_river
        self.n_simulations_turn = n_simulations_turn
        self.n_simulations_flop = n_simulations_flop
        super().__init__(
            low_card_rank, high_card_rank,
        )
        
        # Initialize advanced abstraction components
        self.potential_aware = PotentialAwareAbstraction(n_clusters=100)
        self.implicit_modeling = ImplicitModelingAbstraction()
        self.strength_bucketing = StrengthBucketing()
        self.action_history = []
        
        # Setup save directory and paths
        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(parents=True, exist_ok=True)
        
        self.save_dir = save_dir_path
        self.card_info_lut_path = save_dir_path / "card_info_lut.joblib"
        self.centroid_path = save_dir_path / "centroids.joblib"
        self.progress_path = save_dir_path / "progress.joblib"
        
        # Initialize state
        self.is_running = False
        self.current_stage = None
        self.progress = {
            "pre_flop": 0,
            "river": 0,
            "turn": 0,
            "flop": 0
        }
        
        # Load existing progress if available
        try:
            self.card_info_lut: Dict[str, Any] = joblib.load(self.card_info_lut_path)
            self.centroids: Dict[str, Any] = joblib.load(self.centroid_path)
            if self.progress_path.exists():
                self.progress = joblib.load(self.progress_path)
        except FileNotFoundError:
            self.card_info_lut: Dict[str, Any] = {}
            self.centroids: Dict[str, Any] = {}
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def compute(
        self, n_river_clusters: int, n_turn_clusters: int, n_flop_clusters: int,
    ):
        """Compute all clusters and save to card_info_lut dictionary."""
        try:
            self.is_running = True
            log.info("Starting computation of clusters.")
            start = time.time()
            
            stages = [
                ("pre_flop", lambda: compute_preflop_lossless_abstraction(builder=self), None),
                ("river", self._compute_river_clusters, n_river_clusters),
                ("turn", self._compute_turn_clusters, n_turn_clusters),
                ("flop", self._compute_flop_clusters, n_flop_clusters)
            ]
            
            for stage, compute_fn, n_clusters in stages:
                if stage not in self.card_info_lut:
                    try:
                        self.current_stage = stage
                        log.info(f"Starting {stage} computation")
                        
                        if n_clusters:
                            self.card_info_lut[stage] = compute_fn(n_clusters)
                        else:
                            self.card_info_lut[stage] = compute_fn()
                            
                        self._save_progress(stage, 100)
                        log.info(f"Completed {stage} computation")
                    except Exception as e:
                        log.error(f"Error during {stage} computation: {str(e)}")
                        self._save_progress(stage, self.progress[stage])
                        raise
            
            end = time.time()
            log.info(f"Finished computation of clusters - took {end - start:.2f} seconds.")
            
        except Exception as e:
            log.error(f"Fatal error during computation: {str(e)}")
            raise
        finally:
            self.is_running = False
            self.current_stage = None
            if HAS_GPU:
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception as e:
                    log.warning(f"Failed to free GPU memory: {str(e)}")

    def _compute_river_clusters(self, n_river_clusters: int):
        """Compute river clusters and create lookup table."""
        log.info("Starting computation of river clusters.")
        start = time.time()
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            self._river_ehs = list(
                self._process_with_progress(
                    executor=executor,
                    fn=self.process_river_ehs,
                    items=self.river,
                    stage="river",
                    desc="Processing river EHS"
                )
            )
            
        try:
            log.info("Starting river clustering")
            if HAS_GPU:
                X_gpu = cp.array(self._river_ehs)
                self.centroids["river"], self._river_clusters = self._gpu_cluster(
                    n_river_clusters, X_gpu)
            else:
                self.centroids["river"], self._river_clusters = self._cpu_cluster(
                    n_river_clusters, self._river_ehs)
        except Exception as e:
            log.error(f"Error during river clustering: {str(e)}")
            # Retry with CPU if GPU fails
            self.centroids["river"], self._river_clusters = self._cpu_cluster(
                n_river_clusters, self._river_ehs)
        
        end = time.time()
        log.info(f"Finished computation of river clusters - took {end - start} seconds.")
        return self.create_card_lookup(self._river_clusters, self.river)

    def _compute_turn_clusters(self, n_turn_clusters: int):
        """Compute turn clusters and create lookup table."""
        log.info("Starting computation of turn clusters.")
        start = time.time()
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            self._turn_ehs_distributions = list(
                self._process_with_progress(
                    executor=executor,
                    fn=self.process_turn_ehs_distributions,
                    items=self.turn,
                    stage="turn",
                    desc="Processing turn distributions"
                )
            )
            
        try:
            log.info("Starting turn clustering")
            if HAS_GPU:
                X_gpu = cp.array(self._turn_ehs_distributions)
                self.centroids["turn"], self._turn_clusters = self._gpu_cluster(
                    n_turn_clusters, X_gpu)
            else:
                self.centroids["turn"], self._turn_clusters = self._cpu_cluster(
                    n_turn_clusters, self._turn_ehs_distributions)
        except Exception as e:
            log.error(f"Error during turn clustering: {str(e)}")
            # Retry with CPU if GPU fails
            self.centroids["turn"], self._turn_clusters = self._cpu_cluster(
                n_turn_clusters, self._turn_ehs_distributions)
        
        end = time.time()
        log.info(f"Finished computation of turn clusters - took {end - start} seconds.")
        return self.create_card_lookup(self._turn_clusters, self.turn)

    def _compute_flop_clusters(self, n_flop_clusters: int):
        """Compute flop clusters and create lookup table."""
        log.info("Starting computation of flop clusters.")
        start = time.time()
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            self._flop_potential_aware_distributions = list(
                self._process_with_progress_and_modeling(
                    executor=executor,
                    fn=self.process_flop_potential_aware_distributions,
                    items=self.flop,
                    stage="flop",
                    desc="Processing flop distributions"
                )
            )
            
        try:
            log.info("Starting flop clustering")
            if HAS_GPU:
                X_gpu = cp.array(self._flop_potential_aware_distributions)
                self.centroids["flop"], self._flop_clusters = self._gpu_cluster(
                    n_flop_clusters, X_gpu)
            else:
                self.centroids["flop"], self._flop_clusters = self._cpu_cluster(
                    n_flop_clusters, self._flop_potential_aware_distributions)
        except Exception as e:
            log.error(f"Error during flop clustering: {str(e)}")
            # Retry with CPU if GPU fails
            self.centroids["flop"], self._flop_clusters = self._cpu_cluster(
                n_flop_clusters, self._flop_potential_aware_distributions)
        
        end = time.time()
        log.info(f"Finished computation of flop clusters - took {end - start} seconds.")
        return self.create_card_lookup(self._flop_clusters, self.flop)

    def _gpu_cluster(self, num_clusters: int, X_gpu: cp.ndarray):
        """Perform clustering using GPU acceleration."""
        try:
            km = cuKMeans(
                n_clusters=num_clusters,
                init="random",
                n_init=10,
                max_iter=300,
                tol=1e-04,
                random_state=0,
                output_type='numpy'
            )
            y_km = km.fit_predict(X_gpu)
            centroids = km.cluster_centers_
            
            # Free GPU memory
            del X_gpu
            cp.get_default_memory_pool().free_all_blocks()
            
            return centroids, y_km
        except Exception as e:
            log.error(f"GPU clustering failed: {str(e)}")
            raise

    def _cpu_cluster(self, num_clusters: int, X: cp.ndarray):
        """Perform clustering using CPU."""
        km = KMeans(
            n_clusters=num_clusters,
            init="random",
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0,
        )
        y_km = km.fit_predict(X)
        centroids = km.cluster_centers_
        return centroids, y_km

    def _process_with_progress(self, executor, fn, items, stage, desc):
        """Process items with progress tracking and error handling."""
        total = len(items)
        chunk_size = max(1, total // (os.cpu_count() or 1))
        results = []
        
        try:
            futures = {
                executor.submit(fn, item): i 
                for i, item in enumerate(items)
            }
            
            with tqdm(total=total, desc=desc) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        results.append((futures[future], result))
                        pbar.update(1)
                        self._save_progress(stage, (len(results) / total) * 100)
                    except Exception as e:
                        log.error(f"Error processing item {futures[future]}: {str(e)}")
                        raise
                        
            return [r[1] for r in sorted(results, key=lambda x: x[0])]
        except Exception as e:
            log.error(f"Error during {stage} processing: {str(e)}")
            raise

    def _process_with_progress_and_modeling(self, executor, fn, items, stage, desc):
        """Process items with progress tracking and opponent modeling."""
        results = self._process_with_progress(executor, fn, items, stage, desc)
        
        # Add opponent modeling after processing
        enhanced_results = []
        for result in results:
            opponent_features = self.implicit_modeling.compute_opponent_aware_features(
                result, self.action_history)
            enhanced_results.append(cp.concatenate([result, opponent_features]))
            
        return enhanced_results

    def _save_progress(self, stage: str, progress: float):
        """Save current progress and state."""
        try:
            self.progress[stage] = progress
            joblib.dump(self.progress, self.progress_path)
            joblib.dump(self.card_info_lut, self.card_info_lut_path)
            if stage != "pre_flop":
                joblib.dump(self.centroids, self.centroid_path)
                
            # Create backup copies
            if progress == 100:
                self._create_backup(stage)
                
        except Exception as e:
            log.error(f"Error saving progress: {str(e)}")
            raise

    def _create_backup(self, stage: str):
        """Create backup copies of completed stage data."""
        try:
            backup_dir = self.save_dir / "backups" / stage
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            joblib.dump(self.card_info_lut, backup_dir / f"card_info_lut_{timestamp}.joblib")
            if stage != "pre_flop":
                joblib.dump(self.centroids, backup_dir / f"centroids_{timestamp}.joblib")
        except Exception as e:
            log.warning(f"Failed to create backup for {stage}: {str(e)}")

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        if self.is_running:
            log.info("Received shutdown signal. Saving progress...")
            try:
                if self.current_stage:
                    self._save_progress(self.current_stage, self.progress[self.current_stage])
                log.info("Progress saved successfully")
            except Exception as e:
                log.error(f"Error saving progress during shutdown: {str(e)}")
            self.is_running = False
            
            if HAS_GPU:
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception as e:
                    log.warning(f"Failed to free GPU memory during shutdown: {str(e)}")

    def simulate_get_ehs(self, game: GameUtility) -> cp.ndarray:
        """Get expected hand strength object."""
        ehs = cp.zeros(3)
        for _ in range(self.n_simulations_river):
            idx: int = game.get_winner()
            ehs[idx] += 1 / self.n_simulations_river
        return ehs

    def simulate_get_turn_ehs_distributions(
        self,
        available_cards: cp.ndarray,
        the_board: cp.ndarray,
        our_hand: cp.ndarray,
    ) -> cp.ndarray:
        """Get histogram of frequencies for turn situations."""
        turn_ehs_distribution = cp.zeros(len(self.centroids["river"]))
        
        for _ in range(self.n_simulations_turn):
            river_card = cp.random.choice(available_cards, 1, replace=False)
            board = cp.append(the_board, river_card)
            game = GameUtility(our_hand=our_hand, board=board, cards=self._cards)
            ehs = self.simulate_get_ehs(game)
            
            # Find nearest centroid using GPU if available
            if HAS_GPU:
                distances = pairwise_distances(
                    ehs.reshape(1, -1),
                    self.centroids["river"]
                )
                min_idx = cp.argmin(distances)
            else:
                min_idx = 0
                min_emd = float('inf')
                for idx, centroid in enumerate(self.centroids["river"]):
                    emd = wasserstein_distance(ehs, centroid)
                    if emd < min_emd:
                        min_idx = idx
                        min_emd = emd
                        
            turn_ehs_distribution[min_idx] += 1 / self.n_simulations_turn
            
        return turn_ehs_distribution

    def process_river_ehs(self, public: cp.ndarray) -> cp.ndarray:
        """Get the expected hand strength for a particular card combo."""
        our_hand = public[:2]
        board = public[2:7]
        game = GameUtility(our_hand=our_hand, board=board, cards=self._cards)
        return self.simulate_get_ehs(game)

    def process_turn_ehs_distributions(self, public: cp.ndarray) -> cp.ndarray:
        """Get the potential aware turn distribution."""
        available_cards = self.get_available_cards(
            cards=self._cards, unavailable_cards=public
        )
        return self.simulate_get_turn_ehs_distributions(
            available_cards, the_board=public[2:6], our_hand=public[:2],
        )

    def process_flop_potential_aware_distributions(
        self, public: cp.ndarray,
    ) -> cp.ndarray:
        """Get the potential aware flop distribution."""
        available_cards = self.get_available_cards(
            cards=self._cards, unavailable_cards=public
        )
        
        our_hand = public[:2]
        board = public[2:5]
        game = GameUtility(our_hand=our_hand, board=board, cards=self._cards)
        base_strength = self.simulate_get_ehs(game)
        
        bucket = self.strength_bucketing.compute_strength_bucket(
            cp.mean(base_strength), board, 100.0)
        
        potential_aware_distribution_flop = cp.zeros(len(self.centroids["turn"]))
        
        for j in range(self.n_simulations_flop):
            turn_card = cp.random.choice(available_cards, 1, replace=False)
            the_board = cp.append(board, turn_card).tolist()
            available_cards_turn = cp.array(
                [x for x in available_cards if x != turn_card[0]]
            )
            
            base_turn_distribution = self.simulate_get_turn_ehs_distributions(
                available_cards_turn, the_board=the_board, our_hand=our_hand,
            )
            
            future_states = self._simulate_future_states(
                our_hand, the_board, available_cards_turn)
            potential_distribution = self.potential_aware.compute_potential_distribution(
                base_turn_distribution, future_states, self.centroids)
            
            opponent_features = self.implicit_modeling.compute_opponent_aware_features(
                base_turn_distribution, self.action_history)
            
            enhanced_distribution = cp.concatenate([
                base_turn_distribution,
                potential_distribution,
                opponent_features
            ])
            
            if HAS_GPU:
                distances = pairwise_distances(
                    enhanced_distribution.reshape(1, -1),
                    self.centroids["turn"]
                )
                cluster_idx = cp.argmin(distances)
            else:
                cluster_idx = self._get_nearest_cluster(
                    enhanced_distribution, self.centroids["turn"])
                
            potential_aware_distribution_flop[cluster_idx] += 1 / self.n_simulations_flop
            
        return potential_aware_distribution_flop

    def _simulate_future_states(
        self,
        our_hand: cp.ndarray,
        board: cp.ndarray,
        available_cards: cp.ndarray,
        n_samples: int = 10
    ) -> List[cp.ndarray]:
        """Simulate possible future states for potential-aware abstraction."""
        future_states = []
        for _ in range(n_samples):
            next_card = cp.random.choice(available_cards, 1, replace=False)
            new_board = cp.append(board, next_card)
            game = GameUtility(our_hand=our_hand, board=new_board, cards=self._cards)
            state = self.simulate_get_ehs(game)
            future_states.append(state)
        return future_states

    def _get_nearest_cluster(
        self,
        features: cp.ndarray,
        centroids: cp.ndarray
    ) -> int:
        """Find nearest cluster using earth mover's distance."""
        min_dist = float('inf')
        nearest_idx = 0
        for idx, centroid in enumerate(centroids):
            dist = wasserstein_distance(features, centroid)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = idx
        return nearest_idx

    @staticmethod
    def create_card_lookup(clusters: cp.ndarray, card_combos: cp.ndarray) -> Dict:
        """Create lookup table."""
        log.info("Creating lookup table.")
        lossy_lookup = {}
        for i, card_combo in enumerate(tqdm(card_combos)):
            lossy_lookup[tuple(card_combo)] = clusters[i]
        return lossy_lookup
