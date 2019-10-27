import random

import numpy as np
from jass.base.player_round import PlayerRound
from jass.base.player_round_cheating import PlayerRoundCheating
from jass.base.round_factory import get_round_from_player_round


class Sampler:

    @staticmethod
    def sample(rnd: PlayerRound) -> PlayerRoundCheating:
        my_hand_indexes = np.flatnonzero(rnd.hand)
        free_cards_to_distribute = np.zeros(shape=36, dtype=np.int)
        free_cards_to_distribute.fill(1)
        for card in my_hand_indexes:
            free_cards_to_distribute[card] = 0

        hands = np.zeros(shape=[4, 36], dtype=np.int)

        for j in range(0, 4):
            if j == rnd.player:
                hands[j] = rnd.hand
            else:
                hand, free_cards_to_distribute = Sampler.__get_hands(free_cards_to_distribute)
                hands[j] = hand

        return get_round_from_player_round(rnd, hands)

    @staticmethod
    def __get_hands(free_cards_to_distribute: np.array):
        other_player_hand = np.zeros(shape=36, dtype=int)
        for i in range(0, 9):
            card = random.choice(np.flatnonzero(free_cards_to_distribute))
            free_cards_to_distribute[card] = 0
            other_player_hand[card] = 1

        return other_player_hand, free_cards_to_distribute
