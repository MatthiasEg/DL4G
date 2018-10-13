# HSLU
#
# Created by Thomas Koller on 12.10.18
#
"""
Example how to use flask to create a service for one or more players
"""
import logging

from jass.player_service.player_service_app import PlayerServiceApp
from my_jass.player.my_player import MyPlayer


def create_app():
    """
    This is the factory method for flask. It is automatically detected when flask is run, but we must tell flask
    what python file to use:

        export FLASK_APP=my_player_service.py
        export FLASK_ENV=development
        flask run --host=0.0.0.0 --port=8888
    """
    logging.basicConfig(level=logging.DEBUG)

    # create and configure the app
    app = PlayerServiceApp('my_player_service')

    # you could use a configuration file to load additional variables
    # app.config.from_pyfile('my_player_service.cfg', silent=False)

    # add some players
    app.add_player('my_player', MyPlayer())

    return app
