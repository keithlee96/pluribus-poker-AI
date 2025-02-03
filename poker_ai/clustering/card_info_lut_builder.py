import logging
import time
from pathlib import Path
from typing import Any, Dict, List
import concurrent.futures

import joblib
import numpy as np
from tqdm import tqdm

from poker_ai.clustering.card_combos import CardCombos
from poker_ai.clustering.game_utility import GameUtility
from poker_ai.clustering.preflop import compute_preflop_lossless_abstraction
from poker_ai.clustering.advanced_abstractions import AdvancedAbstractions, RealTimeSolver


log = logging.getLogger("poker_ai.clustering.runner")


class CardInfoLutBuilder(CardCombos):
    """Stores info buckets for each street with advanced abstractions."""

    def __init__(
        self,
        n_simulations_river: int,
        n_simulations_turn: int,
        n_simulations_flop: int,
        low_card_rank: int,
        high_card_rank: int,
        save_dir: str,
    ):
        self.n_simulations_river = n_simulations_river
        self.n_simulations_turn = n_simulations_turn
        self.n_simulations_flop = n_simulations_flop
        super().__init__(low_card_rank, high_card_rank)
        
        # Initialize advanced abstraction techniques
        self.abstractions = {
            "river": AdvancedAbstractions(n_clusters=n_simulations_river),
            "turn": AdvancedAbstractions(n_clusters=n_simulations_turn),
            "flop": AdvancedAbstractions(n_clusters=n_simulations_flop),
        }
        
        self.card_info_lut_path: Path = Path(save_dir) / "card_info_lut.joblib"
        self.centroid_path: Path = Path(save_dir) / "centroids.joblib"
        try:
            self.card_info_lut: Dict[str, Any] = joblib.load(self.card_info_lut_path)
            self.centroids: Dict[str, Any] = joblib.load(self.centroid_path)
        except FileNotFoundError:
            self.centroids: Dict[str, Any] = {}
            self.card_info_lut: Dict[str, Any] = {}

    def compute(
        self, n_river_clusters: int, n_turn_clusters: int, n_flop_clusters: int,
    ):
        """Compute all clusters using advanced abstraction techniques."""
        log.info("Starting computation of clusters with advanced abstractions.")
        start = time.time()
        
        # Preflop remains lossless
        if "pre_flop" not in self.card_info_lut:
            self.card_info_lut["pre_flop"] = compute_preflop_lossless_abstraction(
                builder=self
            )
            joblib.dump(self.card_info_lut, self.card_info_lut_path)
            
        # Use advanced abstractions for postflop streets
        if "river" not in self.card_info_lut:
            self.card_info_lut["river"] = self._compute_river_clusters(
                n_river_clusters,
            )
            joblib.dump(self.card_info_lut, self.card_info_lut_path)
            joblib.dump(self.centroids, self.centroid_path)
            
        if "turn" not in self.card_info_lut:
            self.card_info_lut["turn"] = self._compute_turn_clusters(n_turn_clusters)
            joblib.dump(self.card_info_lut, self.card_info_lut_path)
            joblib.dump(self.centroids, self.centroid_path)
            
        if "flop" not in self.card_info_lut:
            self.card_info_lut["flop"] = self._compute_flop_clusters(n_flop_clusters)
            joblib.dump(self.card_info_lut, self.card_info_lut_path)
            joblib.dump(self.centroids, self.centroid_path)
            
        end = time.time()
        log.info(f"Finished computation of clusters - took {end - start} seconds.")

    def _compute_river_clusters(self, n_river_clusters: int):
        """Compute river clusters using hierarchical clustering."""
        log.info("Starting computation of river clusters.")
        start = time.time()
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            self._river_ehs = list(
                tqdm(
                    executor.map(
                        self.process_river_ehs,
                        self.river,
                        chunksize=len(self.river) // 160,
                    ),
                    total=len(self.river),
                )
            )
            
        # Use hierarchical clustering instead of KMeans
        self._river_clusters = self.abstractions["river"].hierarchical_cluster(
            np.array(self._river_ehs)
        )
        
        end = time.time()
        log.info(f"Finished computation of river clusters - took {end - start} seconds.")
        return self.create_card_lookup(self._river_clusters, self.river)

    def _compute_turn_clusters(self, n_turn_clusters: int):
        """Compute turn clusters with multi-street potential awareness."""
        log.info("Starting computation of turn clusters.")
        start = time.time()
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            self._turn_distributions = list(
                tqdm(
                    executor.map(
                        self.process_turn_distributions,
                        self.turn,
                        chunksize=len(self.turn) // 160,
                    ),
                    total=len(self.turn),
                )
            )
            
        # Use hierarchical clustering
        self._turn_clusters = self.abstractions["turn"].hierarchical_cluster(
            np.array(self._turn_distributions)
        )
        
        end = time.time()
        log.info(f"Finished computation of turn clusters - took {end - start} seconds.")
        return self.create_card_lookup(self._turn_clusters, self.turn)

    def _compute_flop_clusters(self, n_flop_clusters: int):
        """Compute flop clusters with multi-street potential awareness."""
        log.info("Starting computation of flop clusters.")
        start = time.time()
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            self._flop_distributions = list(
                tqdm(
                    executor.map(
                        self.process_flop_distributions,
                        self.flop,
                        chunksize=len(self.flop) // 160,
                    ),
                    total=len(self.flop),
                )
            )
            
        # Use hierarchical clustering
        self._flop_clusters = self.abstractions["flop"].hierarchical_cluster(
            np.array(self._flop_distributions)
        )
        
        end = time.time()
        log.info(f"Finished computation of flop clusters - took {end - start} seconds.")
        return self.create_card_lookup(self._flop_clusters, self.flop)

    def process_river_ehs(self, public: np.ndarray) -> np.ndarray:
        """Get expected hand strength with opponent modeling."""
        our_hand = public[:2]
        board = public[2:7]
        game = GameUtility(our_hand=our_hand, board=board, cards=self._cards)
        return self.simulate_get_ehs(game)

    def process_turn_distributions(self, public: np.ndarray) -> np.ndarray:
        """Process turn with multi-street potential awareness."""
        available_cards = self.get_available_cards(
            cards=self._cards, unavailable_cards=public
        )
        
        # Use advanced abstraction for multi-street potential
        return self.abstractions["turn"].multi_street_potential(
            our_hand=public[:2],
            board=public[2:6],
            available_cards=available_cards,
            n_samples=self.n_simulations_turn
        )

    def process_flop_distributions(self, public: np.ndarray) -> np.ndarray:
        """Process flop with multi-street potential awareness."""
        available_cards = self.get_available_cards(
            cards=self._cards, unavailable_cards=public
        )
        
        # Use advanced abstraction for multi-street potential
        return self.abstractions["flop"].multi_street_potential(
            our_hand=public[:2],
            board=public[2:5],
            available_cards=available_cards,
            n_samples=self.n_simulations_flop
        )

    @staticmethod
    def get_available_cards(
        cards: np.ndarray, unavailable_cards: np.ndarray
    ) -> np.ndarray:
        """Get available cards using set operations for speed."""
        unavailable_cards = set(unavailable_cards.tolist())
        return np.array([c for c in cards if c not in unavailable_cards])

    @staticmethod
    def create_card_lookup(clusters: np.ndarray, card_combos: np.ndarray) -> Dict:
        """Create lookup table mapping card combinations to cluster IDs."""
        log.info("Creating lookup table.")
        return {
            tuple(card_combo): cluster_id
            for card_combo, cluster_id in zip(card_combos, clusters)
        }
