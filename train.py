from players.fixed_policy_player import FixedPolicyPlayer
from players.torch_player import TorchPlayer
from games.game import Game
from games.stacked_game import StackedGame

import torch
from pathlib import Path
import datetime, os
import argparse

import pickle as pkl

# Parse input args
parser = argparse.ArgumentParser()
parser.add_argument('--stack-deck', nargs='+', default=[])
args = parser.parse_args()
stacked_deck = args.stack_deck

if stacked_deck:
    print('Stacked deck =', stacked_deck)

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

player1 = TorchPlayer("player1", 100, save_dir=save_dir)
player2 = TorchPlayer("player2", 100, save_dir=save_dir)

state = Game() if not stacked_deck else StackedGame(stacked_deck)

state.add_player(player1)
state.add_player(player2)

player1.build_net(state.state_dim())
player2.build_net(state.state_dim())

def reset_bankroll(player, force=False):
    if force or (player.get_bankroll() == 0 or player.get_bankroll() > 300):
        player.bankroll = 100

episodes = 1000000
print_every = 10000
for e in range(episodes):
    if e % print_every == 0:
        print("-->", round(100*(e/episodes), 2), "%")
        state.verbose = True
    else:
        state.verbose = False
    # Reset bankrolls if needed
    reset_bankroll(player1, force=True)
    reset_bankroll(player2, force=True)
    # Run the game
    state.reset()
    state.run()
    if state.verbose:
        print()
        print()

file1 = open("player1.net", "wb")
pkl.dump(player1, file1)
file1.close()

file2 = open("player2.net", "wb")
pkl.dump(player2, file2)
file2.close()
