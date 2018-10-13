# HSLU
#
# Created by Thomas Koller on 04.10.18
#

import argparse

from jass.base.player_round import PlayerRound
from jass.stat.jass_stat import PlayerStatCollection, AccuracyByMoveStat
from jass.io.log_parser import LogParser

from my_jass.player.my_player import MyPlayer


def main():
    parser = argparse.ArgumentParser(description='Read rounds from log files and evaluate the players moves')
    parser.add_argument('files', type=str, nargs='+', help='The log files')
    arg = parser.parse_args()

    player = MyPlayer()
    stats = PlayerStatCollection()
    stats.add_statistic(AccuracyByMoveStat())

    for f in arg.files:
        parser = LogParser(f)
        rounds = parser.parse_rounds()
        for rnd in rounds:
            for player_rnd in PlayerRound.all_from_complete_round(rnd):
                action = player.play_card(rnd=player_rnd)
                action_from_log = rnd.get_card_played(player_rnd.nr_played_cards)
                stats.add_result_for_action(label=action_from_log, action=action, player_rnd=player_rnd)


    results = stats.get()
    print(results)


if __name__ == '__main__':
    main()

