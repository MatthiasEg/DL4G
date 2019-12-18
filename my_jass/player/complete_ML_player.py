# HSLU
#
# Created by Thomas Koller on 20.08.18
#
import time

import time

import numpy as np
import pandas as pd
from jass.base.player_round import PlayerRound
from jass.player.player import Player
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.saving import load_model
import tensorflow as tf


class CompleteMLPlayer(Player):
    """
    Sample implementation of a player to play Jass.
    """
    graph = None

    def __init__(self):
        self.sess = tf.Session()
        self.graph = tf.get_default_graph()
        set_session(self.sess)
        self.trump_model = load_model(
            'my_jass/ModelCreation/models/dave/final_model_82_games_025_mean_03_std_06_without_schieben.h5')
        self.card_model = load_model('my_jass/ModelCreation/models/matt/card/card_model_test.h5')

    def select_trump(self, rnd: PlayerRound) -> int:
        """
        Player chooses a trump based on the given round information.

        Args:
            rnd: current round

        Returns:
            selected trump
        """
        if rnd.forehand is None:
            forehand = 0
        else:
            forehand = 1
        arr = np.array([np.append(rnd.hand, forehand)])

        with self.graph.as_default():
            set_session(self.sess)
            trump = self.trump_model.predict(arr)

        choice = int(np.argmax(trump))
        percentage = trump[0][choice]

        if percentage < 0.9 and forehand == 1:
            choice = 6
            return choice
        else:
            return choice

        return choice

    def play_card(self, rnd: PlayerRound) -> int:
        """
        Player returns a card to play based on the given round information.

        Args:
            rnd: current round

        Returns:
            card to play, int encoded
        """

        hand = rnd.hand.astype(np.int)
        # 36 elements

        # unordered cards of the current trick
        cards_of_current_trick = np.zeros(36, np.int)
        if rnd.nr_cards_in_trick > 0:
            cards_of_current_trick[rnd.current_trick[0]] = 1
        if rnd.nr_cards_in_trick > 1:
            cards_of_current_trick[rnd.current_trick[1]] = 1
        if rnd.nr_cards_in_trick > 2:
            cards_of_current_trick[rnd.current_trick[2]] = 1
        # 36 elements

        # player to play
        player = self.one_hot(rnd.player, 4)
        # 4 elements

        # trump
        trump = self.one_hot(rnd.trump, 6)
        # 6 elements

        # total 82 elements
        total = np.array([np.concatenate([hand, cards_of_current_trick, player, trump], axis=0)])

        with self.graph.as_default():
            set_session(self.sess)
            cards = self.card_model.predict(total)

        choice = int(np.argmax(cards))

        return choice

    def one_hot(self, number, size) -> np.array:
        """
        One hot encoding for a single value. Output is float array of size size
        Args:
            number: number to one hot encode
            size: length of the returned array
        Returns:
            array filled with 0.0 where index != number and 1.0 where index == number
        """
        result = np.zeros(size, dtype=np.int)
        result[number] = 1
        return result
