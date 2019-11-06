# HSLU
#
# Created by Thomas Koller on 20.08.18
#
import time

import numpy as np
from jass.base.player_round import PlayerRound
from jass.base.player_round_cheating import PlayerRoundCheating
from jass.base.round_factory import get_round_from_player_round
from jass.player.player import Player

from my_jass.IMCTS.Sampler import Sampler
from my_jass.MCTS.UCB import UCB
from my_jass.MCTS.node import Node
from my_jass.MCTS.tree import Tree
from my_jass.player.MyPlayer import MyPlayer
import tensorflow as tf

class MyIMCTSPlayerMLTrump(Player):
    """
    Sample implementation of a player to play Jass.
    """


    def __init__(self):
        # path is relative to working directory(directory where arena-class-file is situated)
        self.model = tf.keras.models.load_model('my_jass/models/trump_model_V1.h5')
        self.model._make_predict_function()

    def select_trump(self, rnd: PlayerRoundCheating) -> int:
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
        trump = self.model.predict(arr)

        choice = np.argmax(trump)
        # 6 is used in ml for PUSH, but doesn't get translated to 10,
        # so this hack is necessary
        if choice == 6:
            choice = 10
        return choice

    def play_card(self, rnd: PlayerRound) -> int:
        """
        Player returns a card to play based on the given round information.

        Args:
            rnd: current round

        Returns:
            card to play, int encoded
        """

        best_card = self.monteCarloTreeSearch(rnd)
        return best_card

    def monteCarloTreeSearch(self, round: PlayerRound):
        round = Sampler.sample(round)
        tree = Tree()
        root_node = tree.get_root_node()
        root_node.getNodeMCTSInformation().setPlayerNr(round.player)
        root_node.getNodeMCTSInformation().setRound(round)

        time_for_mcts_to_run = time.time() + 0.2
        while time.time() < time_for_mcts_to_run:
            promising_node = self.__select_promising_node(root_node)

            if promising_node.getNodeMCTSInformation().getRound().nr_cards_in_trick < 4:
                self.__expand_node(promising_node, round)

            node_to_explore = promising_node
            if len(promising_node.get_children()) > 0:
                node_to_explore = promising_node.getRandomChild()  # random pick to be optimised
            win_score = self.__simulate_round(node_to_explore)
            self.__backPropagation(node_to_explore, round.player, win_score)
        best_card_to_be_played = root_node.getChildWithMaxVisitCount().getNodeMCTSInformation().getCard()
        return best_card_to_be_played

    def __select_promising_node(self, rootnode: Node) -> Node:
        node = rootnode
        if len(node.get_children()) != 0:
            ucb = UCB()
            node = ucb.findBestUCBNode(node)
        return node

    def __expand_node(self, node: Node, round: PlayerRoundCheating):
        valid_cards = np.flatnonzero(round.get_valid_cards())
        for card in valid_cards:
            newNode = Node()
            newNode.set_parent(node)
            newNode.getNodeMCTSInformation().setRound(round)
            newNode.getNodeMCTSInformation().setPlayerNr(node.getNodeMCTSInformation().getRound().player)
            newNode.getNodeMCTSInformation().setCard(card)
            node.addChild(newNode)

    def __simulate_round(self, node: Node):  # Random walk
        round = get_round_from_player_round(node.getNodeMCTSInformation().getRound(),
                                            node.getNodeMCTSInformation().getRound().hands)
        round.action_play_card(node.getNodeMCTSInformation().getCard())
        cards = round.nr_played_cards
        random_player = MyPlayer()
        while cards < 36:
            player_round = PlayerRoundCheating()
            player_round.set_from_round(round)
            card_action = random_player.play_card(player_round)
            round.action_play_card(card_action)
            cards += 1

        myPoints = round.points_team_0
        pointsEnemy = round.points_team_1
        maxPoints = myPoints + pointsEnemy

        if myPoints > pointsEnemy:
            return (myPoints - 0) / (maxPoints - 0)
        else:
            return 0

    def __backPropagation(self, node: Node, playerNr: int, winScore: int):
        tempNode = node
        while tempNode != None:
            tempNode.getNodeMCTSInformation().incrementVisitCount()
            if tempNode.getNodeMCTSInformation().getPlayerNr() == playerNr:
                tempNode.getNodeMCTSInformation().setWinScore(winScore)
            tempNode = tempNode.get_parent()
