from tokenize import String

from jass.base.player_round_cheating import PlayerRoundCheating
from jass.base.round import Round


class NodeMCTSInformation(object):
    def __init__(self) -> None:
        self._playerNr = 0
        self._winScore = 0.0
        self._visitCount = 0
        self._round = None
        self._card = None

    def getPlayerNr(self) -> int:
        return self._playerNr

    def setPlayerNr(self, player: int):
        self._playerNr = player

    def getWinScore(self) -> float:
        return self._winScore

    def setWinScore(self, winscore: int):
        self._winScore = winscore

    def getVisitCount(self) -> int:
        return self._visitCount

    def setVisitCount(self, visitcount: int):
        self._visitCount = visitcount

    def getRound(self) -> PlayerRoundCheating:
        return self._round

    def setRound(self, round: PlayerRoundCheating):
        self._round = round

    def getCard(self) -> int:
        return self._card

    def setCard(self, card: int):
        self._card = card

    def incrementVisitCount(self):
        self._visitCount += 1
