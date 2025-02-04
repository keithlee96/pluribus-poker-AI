from typing import List, Tuple, Dict, Optional

import numpy as np

from poker_ai.poker.evaluation import Evaluator
from poker_ai.poker.card import Card


class GameUtility:
    """This class takes care of some game related functions."""

    def __init__(self, our_hand: np.ndarray, board: np.ndarray, cards: np.ndarray):
        self._evaluator = Evaluator()
        unavailable_cards = np.concatenate([board, our_hand], axis=0)
        self.available_cards = np.array(
            [c for c in cards if c not in unavailable_cards]
        )
        self.our_hand = our_hand
        self.board = board
        self._cards = cards
        self._hand_cache = {}
        self._board_texture_cache = None
        
        # Track game state
        self.pot_size = 0
        self.action_history: List[Tuple[int, int]] = []

    def evaluate_hand(self, hand: np.ndarray) -> int:
        """
        Evaluate a hand.

        Parameters
        ----------
        hand : np.ndarray
            Hand to evaluate.

        Returns
        -------
            Evaluation of hand
        """
        # Use caching for performance
        hand_key = tuple(hand)
        if hand_key in self._hand_cache:
            return self._hand_cache[hand_key]
            
        try:
            result = self._evaluator.evaluate(
                board=self.board.astype(np.int64).tolist(),
                cards=hand.astype(np.int64).tolist(),
            )
            self._hand_cache[hand_key] = result
            return result
        except Exception as e:
            # Handle evaluation errors gracefully
            raise ValueError(f"Hand evaluation failed: {str(e)}")

    def get_winner(self) -> int:
        """Get the winner.

        Returns
        -------
            int of win (0), lose (1) or tie (2) - this is an index in the
            expected hand strength array
        """
        try:
            our_hand_rank = self.evaluate_hand(self.our_hand)
            opp_hand_rank = self.evaluate_hand(self.opp_hand)
            
            if our_hand_rank > opp_hand_rank:
                return 0
            elif our_hand_rank < opp_hand_rank:
                return 1
            else:
                return 2
        except Exception as e:
            # Log error and return tie as fallback
            print(f"Winner evaluation failed: {str(e)}")
            return 2

    def get_hand_strength(self) -> float:
        """
        Calculate normalized hand strength.
        
        Returns
        -------
        float
            Normalized hand strength between 0 and 1
        """
        our_rank = self.evaluate_hand(self.our_hand)
        
        # Sample opponent hands to estimate relative strength
        n_samples = 100
        wins = 0
        
        for _ in range(n_samples):
            opp_rank = self.evaluate_hand(self.opp_hand)
            if our_rank > opp_rank:
                wins += 1
                
        return wins / n_samples

    def get_board_texture(self) -> np.ndarray:
        """
        Analyze board texture for advanced features.
        
        Returns
        -------
        np.ndarray
            Array of board texture features
        """
        if self._board_texture_cache is not None:
            return self._board_texture_cache
            
        if len(self.board) < 3:
            return np.zeros(4)
            
        # Convert cards to Card objects for easier analysis
        board_cards = [Card(c) for c in self.board]
        
        # Calculate texture features
        features = np.zeros(4)
        
        # Feature 1: Flush draw potential
        suit_counts = {}
        for card in board_cards:
            suit_counts[card.suit] = suit_counts.get(card.suit, 0) + 1
        features[0] = max(suit_counts.values()) / len(board_cards)
        
        # Feature 2: Straight draw potential
        ranks = sorted([card.rank for card in board_cards])
        gaps = [ranks[i+1] - ranks[i] for i in range(len(ranks)-1)]
        features[1] = 1 - (sum(gaps) / (len(gaps) * 4))  # Normalize by max reasonable gap
        
        # Feature 3: Board pairing
        features[2] = 1 - (len(set(ranks)) / len(ranks))
        
        # Feature 4: High card presence
        features[3] = max(ranks) / 14  # Normalize by ace rank
        
        self._board_texture_cache = features
        return features

    def get_pot_odds(self) -> float:
        """
        Calculate pot odds for decision making.
        
        Returns
        -------
        float
            Pot odds as a ratio
        """
        if not self.pot_size:
            return 0.0
        
        # Calculate based on last bet in action history
        last_bet = 0
        for _, amount in reversed(self.action_history):
            if amount > 0:
                last_bet = amount
                break
                
        if last_bet == 0:
            return 0.0
            
        return self.pot_size / (self.pot_size + last_bet)

    def update_game_state(self, action: Tuple[int, int], new_pot_size: Optional[float] = None):
        """
        Update game state with new action and pot size.
        
        Parameters
        ----------
        action : Tuple[int, int]
            Player action as (player_id, amount)
        new_pot_size : Optional[float]
            New pot size if available
        """
        self.action_history.append(action)
        if new_pot_size is not None:
            self.pot_size = new_pot_size

    @property
    def opp_hand(self) -> List[int]:
        """Get random card.

        Returns
        -------
            Two cards for the opponent (Card)
        """
        try:
            return np.random.choice(self.available_cards, 2, replace=False)
        except ValueError as e:
            # Handle case where not enough cards are available
            print(f"Error sampling opponent hand: {str(e)}")
            # Return first two available cards as fallback
            if len(self.available_cards) >= 2:
                return self.available_cards[:2]
            else:
                # Emergency fallback - should never happen in practice
                return np.array([self._cards[0], self._cards[1]])
