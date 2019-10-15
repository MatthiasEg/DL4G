import random
import sys
import logging
import numpy as np

from my_jass.MCTS.node import Node


class UCB:

    def ucbValue(self, totalvisits: int, nodewinscore: float, nodevisit: float) -> float:
        hyperparameter = 1
        if nodevisit == 0:
            return sys.maxsize
        return (nodewinscore / nodevisit) + hyperparameter * np.sqrt(np.log(totalvisits) / nodevisit)

    def findBestUCBNode(self, node: Node):
        parentVisitCount = node.getNodeMCTSInformation().getVisitCount()

        bestscore = 0.0
        bestchildren = []

        child: Node
        for child in node.get_children():
            score = self.ucbValue(parentVisitCount, child.getNodeMCTSInformation().getWinScore(),
                                  node.getNodeMCTSInformation().getVisitCount())
            if score > bestscore:
                bestchildren = [child]
                bestscore = score
            elif score == bestscore:
                bestchildren.append(child)

        if len(bestchildren) == 0:
            logging.warning("No best child found")

        return random.choice(bestchildren)
