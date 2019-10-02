# HSLU
#
# Created by Thomas Koller on 20.08.18
#

import numpy as np

from jass.base.const import color_masks, card_values
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
        # if there are two color with the same highest number of cards, it will calculate the values of each color
        # and define trump according to the highest color cards.
        print("Hand", rnd.hand)

        trump = 0
        max_number_in_color = 0
        second_color_with_highest_number = 0
        third_color_with_highest_number = 0
        for c in range(4):
            number_in_color = (rnd.hand * color_masks[c]).sum()
            if number_in_color > max_number_in_color:
                max_number_in_color = number_in_color
                trump = c
                second_color_with_highest_number = 0
                third_color_with_highest_number = 0
            elif number_in_color == max_number_in_color:
                if second_color_with_highest_number != 0:
                    third_color_with_highest_number = c
                else:
                    second_color_with_highest_number = c

        # action if hand has multiple colors with same highest amount of cards
        if second_color_with_highest_number != 0:
            total_value_of_trump_color_cards = self.calculateColorCardValues(trump, rnd)
            total_value_of_second_color_cards = self.calculateColorCardValues(second_color_with_highest_number, rnd)

            print("trump color: ", trump)
            print("value of trump color", total_value_of_trump_color_cards)
            print("second color: ", second_color_with_highest_number)
            print("value of second color", total_value_of_second_color_cards)

            if third_color_with_highest_number != 0:
                total_value_of_third_color_cards = self.calculateColorCardValues(third_color_with_highest_number,
                                                                                 rnd)
                print("third color: ", third_color_with_highest_number)
                print("value of third color", total_value_of_third_color_cards)

                if total_value_of_second_color_cards > total_value_of_trump_color_cards and \
                        total_value_of_second_color_cards > total_value_of_third_color_cards:
                    trump = second_color_with_highest_number
                elif total_value_of_third_color_cards > total_value_of_trump_color_cards:
                    trump = third_color_with_highest_number

            if total_value_of_second_color_cards > total_value_of_trump_color_cards and \
                    third_color_with_highest_number == 0:
                trump = second_color_with_highest_number

        print("defining trump as: ", trump)
        return trump

    def calculateColorCardValues(self, possibletrumpcolor: int, rnd: PlayerRound) -> int:
        only_cards_of_this_color = rnd.hand * color_masks[possibletrumpcolor]
        totol_value_of_given_color = (only_cards_of_this_color * card_values[possibletrumpcolor]).sum()
        return totol_value_of_given_color

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
