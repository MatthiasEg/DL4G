# HSLU
#
# Created by Thomas Koller on 20.08.18
#
import time

import numpy as np
from jass.base.const import color_masks, card_values
from jass.base.player_round import PlayerRound
from jass.base.player_round_cheating import PlayerRoundCheating
from jass.base.round_factory import get_round_from_player_round
from jass.player.player import Player

from my_jass.IMCTS.Sampler import Sampler
from my_jass.MCTS.UCB import UCB
from my_jass.MCTS.node import Node
from my_jass.MCTS.tree import Tree
from my_jass.player.MyPlayer import MyPlayer


class MyIMCTSPlayerRulesTrump(Player):
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
        # if there are two color with the same highest number of cards, it will calculate the values of each color
        # and define trump according to the highest color cards.
        # print("Hand", rnd.hand)

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

        # print("defining trump as: ", trump)
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
