import logging
from typing import List, Union
from itertools import combinations
import operator
import multiprocessing as mp
from functools import partial
import math

import numpy as np
from tqdm import tqdm

from poker_ai.poker.card import Card
from poker_ai.poker.deck import get_all_suits


log = logging.getLogger("poker_ai.clustering.runner")


def process_chunk(args):
    """Process a chunk of combinations in parallel."""
    chunk_data, publics, chunk_id, total_chunks = args
    our_cards = []
    
    # Convert publics to a set of frozensets for faster lookup
    public_sets = {frozenset(combo) for combo in publics}
    
    for combos in chunk_data:
        # Convert combos to a set for faster intersection check
        combo_set = frozenset(combos)
        for public_set in public_sets:
            # Use set operations instead of np.isin for better performance
            if not combo_set.intersection(public_set):
                # Only create numpy array at the end
                our_cards.append(list(combo_set) + list(public_set))
    
    return our_cards


class CardCombos:
    """This class stores combinations of cards (histories) per street."""

    def __init__(
        self, low_card_rank: int, high_card_rank: int,
    ):
        super().__init__()
        # Sort for caching.
        suits: List[str] = sorted(list(get_all_suits()))
        ranks: List[int] = sorted(list(range(low_card_rank, high_card_rank + 1)))
        self._cards = [Card(rank, suit) for suit in suits for rank in ranks]
        self.starting_hands = self.get_card_combos(2)
        self.flop = self.create_info_combos(
            self.starting_hands, self.get_card_combos(3)
        )
        log.info("created flop")
        self.turn = self.create_info_combos(
            self.starting_hands, self.get_card_combos(4)
        )
        log.info("created turn")
        self.river = self.create_info_combos(
            self.starting_hands, self.get_card_combos(5)
        )
        log.info("created river")

    def get_card_combos(self, num_cards: int) -> List[List[Card]]:
        """
        Get the card combinations for a given street.

        Parameters
        ----------
        num_cards : int
            Number of cards you want returned

        Returns
        -------
            Combos of cards (Card) -> List[List[Card]]
        """
        # Use list comprehension instead of np.array for better performance
        return [list(c) for c in combinations(self._cards, num_cards)]

    def create_info_combos(
        self, start_combos: Union[List, np.ndarray], publics: Union[List, np.ndarray]
    ) -> np.ndarray:
        """Combinations of private info(hole cards) and public info (board).

        Uses the logic that a AsKsJs on flop with a 10s on turn is the same
        as AsKs10s on flop and Js on turn. That logic is used within the
        literature as well as the logic where those two are different.

        Parameters
        ----------
        start_combos : Union[List, np.ndarray]
            Starting combination of cards (beginning with hole cards)
        publics : Union[List, np.ndarray]
            Public cards being added
        Returns
        -------
            Combinations of private information (hole cards) and public
            information (board)
        """
        # Get the number of cards in each public combo
        num_public_cards = len(publics[0]) if publics else 0
        
        if num_public_cards == 3:
            betting_stage = "flop"
            # For flop, use more chunks but smaller size
            n_chunks = mp.cpu_count() * 8  # Increased parallelization
        elif num_public_cards == 4:
            betting_stage = "turn"
            # For turn, use more chunks for better distribution
            n_chunks = mp.cpu_count() * 6
        elif num_public_cards == 5:
            betting_stage = "river"
            # For river, use more chunks for better distribution
            n_chunks = mp.cpu_count() * 6
        else:
            betting_stage = "unknown"
            n_chunks = mp.cpu_count()

        # Calculate optimal chunk size
        chunk_size = max(1, math.ceil(len(start_combos) / n_chunks))
        
        # Split work into chunks
        chunks = []
        for i in range(0, len(start_combos), chunk_size):
            chunk = start_combos[i:i + chunk_size]
            chunk_id = len(chunks)
            total_chunks = (len(start_combos) + chunk_size - 1) // chunk_size
            chunks.append((chunk, publics, chunk_id, total_chunks))

        # Process chunks in parallel
        with mp.Pool(mp.cpu_count()) as pool:
            all_results = []
            
            with tqdm(
                total=len(start_combos),
                dynamic_ncols=True,
                desc=f"Creating {betting_stage} info combos",
            ) as pbar:
                # Process chunks in parallel with better memory handling
                for chunk_result in pool.imap_unordered(process_chunk, chunks, chunksize=1):
                    all_results.extend(chunk_result)
                    # Update progress based on actual combinations processed
                    pbar.update(len(chunk_result) // len(publics))

        # Convert to numpy array only at the end
        return np.array(all_results)
