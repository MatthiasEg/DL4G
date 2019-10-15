import numpy as np

from my_jass.MCTS.nodemctsinformation import NodeMCTSInformation
from my_jass.MCTS.status import Status


class Node:

    def __init__(self) -> None:
        self._parent = None
        self._children = []
        self._nodeMCTSInformation = NodeMCTSInformation()
        self._status = Status.NEW

    def get_parent(self):
        return self._parent

    def set_parent(self, parent):
        self._parent = parent

    def get_children(self) -> []:
        return self._children

    def addChild(self, child):
        self._children.append(child)

    def getNodeMCTSInformation(self) -> NodeMCTSInformation:
        return self._nodeMCTSInformation

    def setNodeMCTSInformation(self, nodeMCTSInformation: NodeMCTSInformation):
        self._nodeMCTSInformation = nodeMCTSInformation

    def getChildWithMaxVisitCount(self):
        maxVisitCountChild = Node()
        child: Node
        for child in self._children:
            if child.getNodeMCTSInformation().getVisitCount() > maxVisitCountChild.getNodeMCTSInformation().getVisitCount():
                maxVisitCountChild = child
        return maxVisitCountChild

    def getChildWithMaxScore(self):
        maxVisitScoreChild = Node()
        child: Node
        for child in self._children:
            if child.getNodeMCTSInformation().getWinScore() > maxVisitScoreChild.getNodeMCTSInformation().getWinScore():
                maxVisitScoreChild = child
        return maxVisitScoreChild

    def getRandomChild(self):
        return np.random.choice(self._children)


