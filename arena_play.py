# HSLU
#
# Created by Thomas Koller on 20.08.18
#

import logging
from jass.base.const import JASS_SCHIEBER_1000
from jass.arena.arena import Arena
from jass.arena.trump_selection_players_strategy import TrumpPlayerStrategy
from jass.arena.play_game_nr_rounds_strategy import PlayNrRoundsStrategy
from jass.player.random_player_schieber import RandomPlayerSchieber

from my_jass.player.my_IMCTS_player import MyIMCTSPlayer
from my_jass.player.my_MCTS_player import MyMCTSPlayer
from my_jass.player.my_player_rule_based import MyRulePlayer


def main():
    # Set the global logging level (Set to debug or info to see more messages)
    logging.basicConfig(level=logging.WARNING)

    # setup the arena
    arena = Arena(jass_type=JASS_SCHIEBER_1000,
                  trump_strategy=TrumpPlayerStrategy(),
                  play_game_strategy=PlayNrRoundsStrategy(4))
    player = RandomPlayerSchieber()
    my_player = MyIMCTSPlayer()

    arena.set_players(my_player, player, my_player, player)
    arena.nr_games_to_play = 10
    print('Playing {} games'.format(arena.nr_games_to_play))
    arena.play_all_games()
    total_games = arena.nr_wins_team_0 + arena.nr_wins_team_1 + arena.nr_draws
    print('Wins Team 0: {} ({:.2f}%)'.format(arena.nr_wins_team_0, arena.nr_wins_team_0 / total_games))
    print('Wins Team 1: {} ({:.2f}%)'.format(arena.nr_wins_team_1, arena.nr_wins_team_1 / total_games))
    print('Draws: {} ({:.2f}%)'.format(arena.nr_draws, arena.nr_draws / total_games))
    print('Delta Points: {}'.format(arena.delta_points))


if __name__ == '__main__':
    main()
