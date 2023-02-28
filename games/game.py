import numpy as np
from abc import ABC, abstractmethod

deck = [
         '14s', '14c', '14h', '14d',   # Aces
         '13s', '13c', '13h', '13d',   # Kings
         '12s', '12c', '12h', '12d',   # Queens
         '11s', '11c', '11h', '11d',   # Jacks
         '10s', '10c', '10h', '10d',   # Tens
         '09s', '09c', '09h', '09d',   # Nines
         '08s', '08c', '08h', '08d',   # Eights
         '07s', '07c', '07h', '07d',   # Sevens
         '06s', '06c', '06h', '06d',   # Sixes
         '05s', '05c', '05h', '05d',   # Fives
         '04s', '04c', '04h', '04d',   # Fours
         '03s', '03c', '03h', '03d',   # Threes
         '02s', '02c', '02h', '02d'    # Twos
       ]

class Game:
    def __init__(self, verbose=True):
        self.reset_deck()
        self.reset_pot()
        self.verbose = verbose
        self.players = []
        self.active_players = []
        self.shared_cards = []

    def __str__(self):
        res = "---------------------\n"
        res += "Shared Cards: "
        res += str(self.shared_cards)
        res += "\n"
        res += "Pot: "
        res += str(round(self.pot, 2))
        res += " BB \n"
        for player in self.active_players:
            res += "---> "
            res += player.get_name()
            res += ("(" + str(player.get_bankroll()) + ") ")
            if player.show_cards:
                res += str(player.get_hand())
            else:
                res += str(["***", "***"])
            res += "\n"
        return res

    def reset_deck(self):
        self.deck = deck[:]

    def reset_pot(self):
        self.pot = 0.

    def reset_active_players(self):
        self.active_players = self.players[:]

    def reset_shared_cards(self):
        self.shared_cards = []

    def add_player(self, player):
        self.players.append(player)
        self.active_players.append(player)

    def reset(self):
        self.reset_active_players()
        self.reset_deck()
        self.reset_pot()
        self.reset_shared_cards()
        [player.reset() for player in self.players]

    def pull_from_deck(self):
        card = np.random.choice(self.deck, replace=False)
        self.deck.remove(card)
        return card

    def deal_hand(self, player):
        player.set_card(self.pull_from_deck())
        player.set_card(self.pull_from_deck())

    def deal_hands(self):
        for player in self.players:
            player.reset_hand()
            self.deal_hand(player)

    def deal_flop(self):
        assert(len(self.shared_cards) == 0)
        self.shared_cards.append(self.pull_from_deck())
        self.shared_cards.append(self.pull_from_deck())
        self.shared_cards.append(self.pull_from_deck())

    def deal_turn(self):
        assert(len(self.shared_cards) == 3)
        self.shared_cards.append(self.pull_from_deck())

    def deal_river(self):
        assert(len(self.shared_cards) == 4)
        self.shared_cards.append(self.pull_from_deck())

    def blind_helper(self, player, blind):
        if(player.get_bankroll() >= blind):
            self.pot += blind
            player.bet(blind)
        else:
            self.pot += player.get_bankroll()
            player.bet(player.get_bankroll())

    def big_blind(self, player):
        self.blind_helper(player, 1)

    def small_blind(self, player):
        self.blind_helper(player, 0.5)

    def payback(self, players_in_pot, new_price, old_price):
        assert(new_price < old_price)
        for i in range(len(players_in_pot)):
            paid = players_in_pot[i][1]
            if paid == old_price:
                players_in_pot[i][0].bankroll += (old_price - new_price)
                players_in_pot[i][1] = new_price
                self.pot -= (old_price - new_price)
        return players_in_pot

    def get_actions(self, force_bet=False):
        # initialize parameters
        price = 0
        players_in_pot = [[player, 0] for player in self.active_players]
        ncycles = 0
        # If there is an all-in player then return
        if any([player.bankroll == 0 for player in self.active_players]):
            return
        # If every player has had an oportunity to act and all players in pot
        # have paid the same price, then we stop
        while not (all([price == paid
                            for _, paid in players_in_pot]) and ncycles > 0):
            # Iterate through players in pot and query actions from them
            players_to_remove = []
            for pos in range(len(players_in_pot)):
                if players_in_pot[pos][0].get_bankroll() == 0:
                    return
                # Check blinds
                if force_bet and ncycles == 0 and pos == 0:
                    self.small_blind(players_in_pot[0][0])
                    players_in_pot[0][1] = 0.5
                elif force_bet and ncycles == 0 and pos == 1:
                    self.big_blind(players_in_pot[1][0])
                    players_in_pot[1][1] = 1
                    price = 1
                else:
                    if ncycles > 0 and players_in_pot[pos][1] == price:
                        continue
                    action = players_in_pot[pos][0].act(self, price)
                    if action[0] == "fold":
                        if self.verbose:
                            print(players_in_pot[pos][0].get_name(), "folds")
                        self.active_players.remove(players_in_pot[pos][0])
                        players_to_remove.append(players_in_pot[pos])
                    elif action[0] == "check":
                        assert(price == 0)
                        if self.verbose:
                            print(players_in_pot[pos][0].get_name(), "checks")
                    elif action[0] == "bet":
                        wager = action[1]
                        assert(price == 0 and wager > 0)
                        if self.verbose:
                            print(players_in_pot[pos][0].get_name(), "bets", wager)
                        players_in_pot[pos][0].bet(wager)
                        players_in_pot[pos][1] += wager
                        self.pot += wager
                        price = wager
                    elif action[0] == "call":
                        diff = price - players_in_pot[pos][1]
                        assert(price > 0)
                        if self.verbose:
                            print(players_in_pot[pos][0].get_name(), "calls", action[1])
                        players_in_pot[pos][0].bet(diff)
                        players_in_pot[pos][1] += diff
                        self.pot += diff
                    elif action[0] == "raise":
                        assert(action[1] >= 2*price)
                        if self.verbose:
                            print(players_in_pot[pos][0].get_name(), "raises to", action[1])
                        players_in_pot[pos][0].bet(action[1])
                        players_in_pot[pos][1] = action[1]
                        price = action[1]
                        self.pot += action[1]
                    elif action[0] == "all-in":
                        assert(action[1] == players_in_pot[pos][0].get_bankroll())
                        if self.verbose:
                            print(players_in_pot[pos][0].get_name(), "all-in", action[1])
                        players_in_pot[pos][0].bet(action[1])
                        players_in_pot[pos][1] = action[1]
                        if action[1] < price:
                            players_in_pot = self.payback(players_in_pot, action[1], price)
                        price = action[1]
                        self.pot += action[1]
                    else:
                        raise Exception(action[0] + " is a invalid action.")
            # Remove folded players
            [players_in_pot.remove(player) for player in players_to_remove]
            # Update cylce count
            ncycles += 1
        # Update active players
        self.active_players = [player for player, _ in players_in_pot]

    def check_royal_flush(self, cards):
        royal_flush_hands = (['14s','13s','12s','11s','10s'],
                             ['14c','13c','12c','11c','10c'],
                             ['14h','13h','12h','11h','10h'],
                             ['14d','13d','12d','11d','10d'])
        for hand in royal_flush_hands:
            if all(np.isin(hand, cards)):
                return hand
        return False

    def check_straight_flush(self, cards):
        hash = {'s': [], 'c': [], 'h': [], 'd': []}
        [hash[card[2]].append(card) for card in cards]
        for suited_cards in hash.values():
            if len(suited_cards) >= 5:
                hand = self.check_straight(suited_cards)
                if hand:
                    return hand
        return False

    def check_four_pair(self, cards):
        ranks, counts = np.unique([int(card[:2]) for card in cards], return_counts=True)
        if 4 in counts:
            four_rank = list(reversed(ranks[counts == 4]))[0]
            res = [card for card in cards if int(card[:2]) == four_rank]
            [cards.remove(card) for card in res]
            res += sorted(cards, key=lambda x: int(x[:2]), reverse=True)[:1]
            return res
        return False

    def check_full_house(self, cards):
        ranks, counts = np.unique([int(card[:2]) for card in cards], return_counts=True)

        if 3 in counts and 2 in counts:
            set_rank = ranks[counts >= 3][-1]
            idx = np.where(ranks == set_rank)
            ranks = np.delete(ranks, idx)
            counts = np.delete(counts, idx)
            pair_rank = ranks[counts >= 2][-1]
            res = [card for card in cards if int(card[:2]) == set_rank]
            res += [card for card in cards if int(card[:2]) == pair_rank]
            [cards.remove(card) for card in res]
            res += sorted(cards, key=lambda x: int(x[:2]), reverse=True)[:1]
            return res
        return False

    def check_flush(self, cards):
        hash = {'s': [], 'c': [], 'h': [], 'd': []}
        for card in cards:
            hash[card[2]].append(card)
        res = [sorted(card_lst, key=lambda x: int(x[:2]), reverse=True)
                            for card_lst in hash.values() if len(card_lst) >= 5]
        if res:
            return res[0]
        return False

    def check_straight(self, cards):
        ranks = list(np.unique([int(card[:2]) for card in cards]))
        if len(ranks) < 5:
            return False
        if ranks[-1] == 14:
            ranks = [1] + ranks
        for i in reversed(range(4, len(ranks))):
            straight = True
            for j in range(4):
                lcard, rcard = ranks[i-j-1], ranks[i-j]
                if not (lcard+1==rcard):
                    straight = False
                    break
            if straight:
                straight_ranks = ranks[i-4:i+1]
                hash = {}
                for rank in straight_ranks:
                    hash[rank] = None
                res = []
                for card in cards:
                    rank = int(card[:2])
                    if (rank in hash) and not hash[rank]:
                        hash[rank] = card
                        res.append(card)

                # Edge Case: 5 high straight with the ace acting as a 1
                if straight_ranks[0] == 1:
                    res = sorted(res, key=lambda x: int(x[:2]), reverse=True) # 5, 4, 3, 2
                    res.append([card for card in cards if int(card[:2])==14][0]) # 5, 4, 3, 2, 14
                    return res

                return sorted(res, key=lambda x: int(x[:2]), reverse=True)
        return False

    def check_set(self, cards):
        ranks, counts = np.unique([int(card[:2]) for card in cards], return_counts=True)
        if 3 in counts:
            set_rank = list(reversed(ranks[counts == 3]))[0]
            res = [card for card in cards if int(card[:2]) == set_rank]
            [cards.remove(card) for card in res]
            res += sorted(cards, key=lambda x: int(x[:2]), reverse=True)[:3]
            return res
        return False

    def check_two_pair(self, cards):
        ranks, counts = np.unique([int(card[:2]) for card in cards], return_counts=True)
        if sum(counts == 2) >= 2:
            pair_ranks = list(reversed(ranks[counts == 2]))
            res = [card for card in cards if int(card[:2]) == pair_ranks[0]]
            res += [card for card in cards if int(card[:2]) == pair_ranks[1]]
            [cards.remove(card) for card in res]
            res += sorted(cards, key=lambda x: int(x[:2]), reverse=True)[:1]
            return res
        return False

    def check_pair(self, cards):
        ranks, counts = np.unique([int(card[:2]) for card in cards], return_counts=True)
        if 2 in counts:
            pair_rank = ranks[counts == 2][0]
            res = [card for card in cards if int(card[:2]) == pair_rank]
            [cards.remove(card) for card in res]
            res += sorted(cards, key=lambda x: int(x[:2]), reverse=True)[:3]
            return res
        return False

    def check_high_card(self, cards):
        return sorted(cards, key=lambda x: int(x[:2]), reverse=True)[:5]

    def break_ties(self, candidates):
        for card_idx in range(5):
            high_rank, next_round = 0, []
            for i in range(len(candidates)):
                candidate = candidates[i]
                hand = candidates[i][1]
                rank = int(hand[card_idx][:2])
                if rank > high_rank:
                    high_rank = rank
                    next_round = [candidate]
                elif rank == high_rank:
                    next_round.append(candidate)
            candidates = next_round
            if len(next_round) == 1:
                break
        return candidates

    def showdown_helper(self, func):
        candidates = []
        for player in self.active_players:
            hand = func(self.shared_cards + player.get_hand())
            if hand:
                candidates.append((player, hand))
        return candidates

    def showdown(self):
        hands = (
                    ("ROYAL FLUSH",    self.check_royal_flush),
                    ("STRAIGHT FLUSH", self.check_straight_flush),
                    ("FOUR PAIR",      self.check_four_pair),
                    ("FULL HOUSE",     self.check_full_house),
                    ("FLUSH",          self.check_flush),
                    ("STRAIGHT",       self.check_straight),
                    ("SET",            self.check_set),
                    ("TWO PAIR",       self.check_two_pair),
                    ("PAIR",           self.check_pair),
                    ("HIGH CARD",      self.check_high_card)
                )

        if len(self.active_players) == 1:
            return self.active_players

        for hand in hands:
            candidates = self.showdown_helper(hand[1])
            if candidates:
                winners = self.break_ties(candidates)
                if self.verbose:
                    print("Winning hand:", hand[0], winners[0][1])
                return [winner[0] for winner in winners]

    def run(self):
        self.reset()
        self.deal_hands()
        if self.verbose:
            print(self)
        self.get_actions(force_bet=True)
        if len(self.active_players) > 1:
            self.deal_flop()
            if self.verbose:
                print(self)
            self.get_actions()
        if len(self.active_players) > 1:
            self.deal_turn()
            if self.verbose:
                print(self)
            self.get_actions()
        if len(self.active_players) > 1:
            self.deal_river()
            if self.verbose:
                print(self)
            self.get_actions()
        winners = self.showdown()
        if self.verbose:
            print("The winner(s):", [winner.name for winner in winners])
        for player in self.active_players:
            if player in winners:
                player.reward(self.pot/len(winners), self)
            else:
                player.reward(0, self)

    def featurize_card(self, card):
        suit_map = {'s': 0, 'c': 1, 'h': 2, 'd': 3}
        return [suit_map[card[2]], int(card[:2])]

    def featurize(self, price, selected_player):
        # Order of features:
        #   Game Data
        #     - Pot
        #     - Price
        #   Shared Card Data (x5)
        #     - Suit
        #     - Rank
        #   Selected Player's Data
        #     - Position relative to SB
        #     - Bankroll
        #     Cards (x2)
        #       - Suit
        #       - Rank
        #   Other Player's Data
        #     - Bool still in Pot
        #     - Position relative to SB
        #     - Bankroll

        # Add game data
        state = [self.pot, price]
        # Add shared card data
        for i in range(5):
            if i < len(self.shared_cards):
                state.extend(self.featurize_card(self.shared_cards[i]))
            else:
                state.extend([0, 0])
        # Selected player's data
        state.extend([self.players.index(selected_player),
                      selected_player.bankroll] +
                      self.featurize_card(selected_player.get_hand()[0]) +
                      self.featurize_card(selected_player.get_hand()[1]))
        # Other player's data
        for position, player in enumerate(self.players):
            if not player == selected_player:
                if player in self.active_players:
                    state.extend([1,
                                  position,
                                  player.get_bankroll()])
                else:
                    state.extend([0, 0, 0])
        return np.array(state)

    def state_dim(self):
        if self.players:
            self.deal_hands()
            res = len(self.featurize(0, self.players[0]))
            self.reset()
            return res
        else:
            return 0
