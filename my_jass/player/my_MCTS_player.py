# HSLU
#
# Created by Thomas Koller on 20.08.18
#
import time

import numpy as np
from jass.base.const import color_masks
from jass.base.player_round_cheating import PlayerRoundCheating
from jass.base.round_factory import get_round_from_player_round
from jass.player.player import Player

from my_jass.MCTS.UCB import UCB
from my_jass.MCTS.node import Node
from my_jass.MCTS.tree import Tree
from my_jass.player.MyPlayer import MyPlayer


class MyMCTSPlayer(Player):
    """
    Sample implementation of a player to play Jass.
    """

    def select_trump(self, rnd: PlayerRoundCheating) -> int:
        """
        Player chooses a trump based on the given round information.

        Args:
            rnd: current round

        Returns:
            selected trump
        """
        # select the trump with the largest number of cards

        trump = 0
        max_number_in_color = 0
        for c in range(4):
            number_in_color = (rnd.hand * color_masks[c]).sum()
            if number_in_color > max_number_in_color:
                max_number_in_color = number_in_color
                trump = c
        return trump

    def play_card(self, rnd: PlayerRoundCheating) -> int:
        """
        Player returns a card to play based on the given round information.

        Args:
            rnd: current round

        Returns:
            card to play, int encoded
        """

        bestCard = self.monteCarloTreeSearch(rnd)
        return bestCard

    def monteCarloTreeSearch(self, round: PlayerRoundCheating):
        tree = Tree()
        root_node = tree.get_root_node()
        root_node.getNodeMCTSInformation().setPlayerNr(round.player)
        root_node.getNodeMCTSInformation().setRound(round)

        timeForMCTSToRun = time.time() + 0.01
        while time.time() < timeForMCTSToRun:
            promisingnode = self.selectPromisingNode(root_node)

            if promisingnode.getNodeMCTSInformation().getRound().nr_cards_in_trick < 4:
                self.expandNode(promisingnode, round)

            nodeToExplore = promisingnode
            if len(promisingnode.get_children()) > 0:
                nodeToExplore = promisingnode.getRandomChild()
            winScore = self.simulateRound(nodeToExplore)
            self.backPropagation(nodeToExplore, round.player, winScore)
        bestCardToBePlayed = root_node.getChildWithMaxVisitCount().getNodeMCTSInformation().getCard()
        return bestCardToBePlayed

    def selectPromisingNode(self, rootnode: Node) -> Node:
        node = rootnode
        if len(node.get_children()) != 0:
            ucb = UCB()
            node = ucb.findBestUCBNode(node)
        return node

    def expandNode(self, node: Node, round: PlayerRoundCheating):
        validCards = np.flatnonzero(round.get_valid_cards())
        for card in validCards:
            newNode = Node()
            newNode.set_parent(node)
            newNode.getNodeMCTSInformation().setRound(round)
            newNode.getNodeMCTSInformation().setPlayerNr(node.getNodeMCTSInformation().getRound().player)
            newNode.getNodeMCTSInformation().setCard(card)
            node.addChild(newNode)

    def simulateRound(self, node: Node):  # Random walk
        round = get_round_from_player_round(node.getNodeMCTSInformation().getRound(),
                                            node.getNodeMCTSInformation().getRound().hands)
        round.action_play_card(node.getNodeMCTSInformation().getCard())
        cards = round.nr_played_cards
        randomPlayer = MyPlayer()
        while cards < 36:
            player_round = PlayerRoundCheating()
            player_round.set_from_round(round)
            card_action = randomPlayer.play_card(player_round)
            round.action_play_card(card_action)
            cards += 1

        myPoints = round.points_team_0
        pointsEnemy = round.points_team_1
        maxPoints = myPoints + pointsEnemy

        if myPoints > pointsEnemy:
            return (myPoints - 0) / (maxPoints - 0)
        else:
            return 0

    def backPropagation(self, node: Node, playerNr: int, winScore: int):
        tempNode = node
        while tempNode != None:
            tempNode.getNodeMCTSInformation().incrementVisitCount()
            if tempNode.getNodeMCTSInformation().getPlayerNr() == playerNr:
                tempNode.getNodeMCTSInformation().setWinScore(winScore)
            tempNode = tempNode.get_parent()
