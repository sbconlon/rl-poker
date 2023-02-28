from abc import ABC, abstractmethod

class Player(ABC):
    def __init__(self, name, bankroll, show_cards=True):
        self.reset_hand()
        self.name = name
        self.bankroll = bankroll
        self.show_cards = show_cards

    def get_name(self):
        return self.name

    def reset_hand(self):
        self.hand = []

    def set_card(self, card):
        assert(len(self.hand) < 2)
        self.hand.append(card)

    def get_hand(self):
        return self.hand

    def bet(self, wager):
        assert(wager <= self.bankroll)
        self.bankroll -= wager

    def get_bankroll(self):
        return self.bankroll

    def reward(self, value, game):
        self.bankroll += value

    def reset(self):
        pass

    @abstractmethod
    def act(self, game, price):
        pass
