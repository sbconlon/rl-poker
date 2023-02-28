from .player import Player
from .player_net import PlayerNet

import torch
from torch import nn
from torchvision import transforms as T
import numpy as np
from pathlib import Path
from collections import deque
import random, copy
from scipy.special import softmax

class TorchPlayer(Player):
    def __init__(self, name, bankroll, save_dir):
        super().__init__(name, bankroll)
        # basic parameters
        self.save_dir = save_dir
        self.state_dim = None
        self.action_dim = 12
        self.use_cuda = torch.cuda.is_available()
        self.net = None
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0
        self.save_every = 5e5
        # memory parameters
        self.memory = deque(maxlen=100000)
        self.batch_size = 5000
        # learning parameters
        self.burnin = 1e4 # min. experiences before training
        self.learn_every = 1000 # no. experiences between updates to Q_online
        self.sync_every = 1e4 # no. experiences between syncs
        self.gamma = 0.9
        # Remember last state for caching purposes
        self.last_state = None # Last state seen
        self.last_bet = 0. # Last bet made
        self.last_action_pair = None # Last action pair
        self.init_bankroll = None # Save starting stack size

    def action_space(self, pot, price, bankroll):
        # List of all (idx, action) pairs
        actions = {
                        0:  ("fold",  0),               # Fold
                        1:  ("check", 0),               # Check
                        2:  ("call",  price),           # Call
                        3:  ("bet",   2),               # Bet 2
                        4:  ("bet",   max(pot//3, 1)),  # Bet 1/3 pot
                        5:  ("bet",   max(pot//2, 1)),  # Bet 1/2 pot
                        6:  ("bet",   pot),             # Bet 1x pot
                        7:  ("bet",   2*pot),           # Bet 2x pot
                        8:  ("bet",   3*pot),           # Bet 3x pot
                        9:  ("raise", 2*price),         # Raise 2x
                        10: ("raise", 3*price),         # Raise 3x
                        11: ("all-in", bankroll)        # Raise all in
                 }
        # Remove invalid actions
        if price == 0:
            del actions[0]   # Remove fold
            del actions[2]   # Remove call
            del actions[9]   # Remove raise
            del actions[10]  # Remove raise
        else:
            del actions[1]   # Remove check
            del actions[3]   # Remove bet
            del actions[4]   # Remove bet
            del actions[5]   # Remove bet
            del actions[6]   # Remove bet
            del actions[7]   # Remove bet
            del actions[8]   # Remove bet
        # Remove any actions that bet more than our bankroll
        actions = {id:action for id, action in actions.items() if action[1] <= bankroll}
        return actions

    def get_action(self, game, price):
        # Assert the net has been built
        assert(self.net)
        # Get state and all valid actions
        actions = self.action_space(game.pot, price, self.bankroll)
        state = game.featurize(price, self)

        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_pair = random.choice(list(actions.items()))

        # EXPLOIT
        else:
            state = state.__array__()
            if self.use_cuda:
                state = torch.tensor(state).cuda()
            else:
                state = torch.tensor(state)
            state = state.unsqueeze(0)
            action_freqs = self.net(state.float(), model="online").tolist()[0]
            # Remove frequencies associated with invalid actions
            action_freqs = [action_freqs[id] for id in actions.keys()]
            # Sample from valid idxs
            action_id = np.random.choice(list(actions.keys()),
                                                    p=softmax(action_freqs))
            # Get action pair (action_id, action)
            action_pair = (action_id, actions[action_id])

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min,
                                        self.exploration_rate)
        # increment step
        return action_pair

    # override base class
    def act(self, game, price):
        # Save starting bankroll if this is our first action
        if self.last_state is None:
            self.save_init_bankroll()
        # Get state
        state = game.featurize(price, self)
        # Learn from last action
        if not self.last_state is None:
            assert(self.last_action_pair)
            # Remember
            reward = -1*self.last_action_pair[1][1]
            self.cache(self.last_state,
                       state,
                       self.last_action_pair[0],
                       0,
                       False)
            # Learn
            q, loss = self.learn()
        # Get next action
        action_pair = self.get_action(game, price)
        # Store state, action, and bet
        self.last_state = state
        self.last_action_pair = action_pair
        self.curr_step += 1
        # Return action to the game
        return action_pair[1]

    def cache(self, state, next_state, action, reward, done):
        state = state.__array__()
        next_state = next_state.__array__()

        if self.use_cuda:
            state = torch.tensor(state).cuda()
            next_state = torch.tensor(next_state).cuda()
            action = torch.tensor([action]).cuda()
            reward = torch.tensor([reward]).cuda()
            done = torch.tensor([done]).cuda()
        else:
            state = torch.tensor(state).float()
            next_state = torch.tensor(next_state)
            action = torch.tensor([action])
            reward = torch.tensor([reward])
            done = torch.tensor([done])

        self.memory.append((state, next_state, action, reward, done,))


    def recall(self):
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        current_Q = self.net(state.float(), model="online")[
            np.arange(0, self.batch_size), action
        ] # Q online(s, a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state.float(), model="online")

        if self.use_cuda:
            next_Q = torch.empty((self.batch_size,), dtype=torch.float, device="cuda")
        else:
            next_Q = torch.empty((self.batch_size,), dtype=torch.float)

        for idx in range(self.batch_size):
            valid_action_pairs = self.action_space(next_state[idx, 0].item(),  # pot
                                                   next_state[idx, 1].item(),  # price
                                                   next_state[idx, 13].item()) # bankroll
            # Hash valid action ids to their corresponding freq (action id -> freq)
            Q_hash = {id:next_state_Q[idx, id].item() for id in valid_action_pairs.keys()}
            # Select action pair w highest freq
            best_action_id = max(Q_hash.keys(), key=Q_hash.get)
            # Compute next Q
            next_Q[idx] = self.net(next_state[idx].float(), model="target")[best_action_id]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        save_path = (self.save_dir / f"player_net_{int(self.curr_step // self.save_every)}.chkpt")
        torch.save(dict(model=self.net.state_dict(),
                   exploration_rate=self.exploration_rate),
                   save_path,)
        print(f"PlayerNet saved to {save_path} at step {self.curr_step}")

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
        if self.curr_step % self.save_every == 0:
            self.save()
        if self.curr_step < self.burnin:
            return None, None
        if self.curr_step % self.learn_every != 0:
            return None, None
        # Sample from memory
        state, next_state, action, reward, done = self.recall()
        # Get TD Estimate
        td_est = self.td_estimate(state, action)
        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)
        # Backprop loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)
        return (td_est.mean().item(), loss)

    def reward(self, value, game):
        super().reward(value, game)
        if self.last_state is None:
            self.last_state = torch.tensor(np.zeros(self.state_dim))
            self.init_bankroll = self.bankroll
        if self.last_action_pair is None:
            self.last_action_pair = (-1, (None, 0))
        # Remember
        state = game.featurize(0, self)
        gain = self.bankroll - self.init_bankroll
        self.cache(self.last_state,
                   state,
                   self.last_action_pair[0],
                   gain,
                   True)
        # Learn
        q, loss = self.learn()

    def build_net(self, state_dim):
        self.state_dim = state_dim
        self.net = PlayerNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device="cuda")
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def reset(self):
        self.last_state = None
        self.last_bet = None

    def save_init_bankroll(self):
        self.init_bankroll = self.bankroll
