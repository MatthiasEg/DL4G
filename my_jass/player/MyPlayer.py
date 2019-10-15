# HSLU
#
# Created by Thomas Koller on 20.08.18
#

import numpy as np

from jass.base.const import color_masks
from jass.base.player_round import PlayerRound
from jass.player.player import Player


class MyPlayer(Player):
    """
    Sample implementation of a player to play Jass.
    """
    def select_trump(self, rnd: PlayerRound) -> int:
        """
        Player chooses a trump based on the given round information.

        Args:
            rnd: current round

        Returns:
            selected trump
        """
        # select the trump with the largest number of cards
        # print(rnd.hand)
        # print(rnd.hand.shape)

        trump = 0
        max_number_in_color = 0
        for c in range(4):
            number_in_color = (rnd.hand * color_masks[c]).sum()
            if number_in_color > max_number_in_color:
                max_number_in_color = number_in_color
                trump = c
        return trump

    def play_card(self, rnd: PlayerRound) -> int:
        """
        Player returns a card to play based on the given round information.

        Args:
            rnd: current round

        Returns:
            card to play, int encoded
        """
        # play random

        # get the valid cards to play
        valid_cards = rnd.get_valid_cards()

        # select a random card
        return np.random.choice(np.flatnonzero(valid_cards))
