from players.torch_player import TorchPlayer
from players.cli_player import CLIPlayer
from games.game import Game

import pickle as pkl

if __name__ == "__main__":
    file = open("player1.net", "rb")
    player1 = pkl.load(file)
    file.close()

    player1.bankroll=100
    player1.show_cards = False

    hero = CLIPlayer("Hero", 100)
    game = Game(verbose=True)
    game.add_player(player1)
    game.add_player(hero)
    i = 0
    while True:
        if player1.bankroll == 0 or player1.bankroll > 300:
            player1.bankroll = 100
        if hero.bankroll == 0 or hero.bankroll > 300:
            hero.bankroll = 100
        print("RUN #{}:".format(i))
        game.run()
        print()
        print()
        i += 1

    print("--RESULTS--")
    print(player1.get_name(), player1.get_bankroll())
    print(hero.get_name(), hero.get_bankroll())
