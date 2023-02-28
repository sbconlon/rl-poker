from games.game import Game

class StackedGame(Game):
    def __init__(self, card_runout, verbose=False):
        self.runout = card_runout
        super().__init__(verbose=verbose)

    def remove_cards_from_deck(self, cards):
        for card in cards:
            self.deck.remove(card)

    def deal_flop(self):
        assert(len(self.shared_cards) == 0)
        for i in range(3):
            if i < len(self.runout):
                self.shared_cards.append(self.runout[i])
            else:
                self.shared_cards.append(self.pull_from_deck())

    def deal_turn(self):
        assert(len(self.shared_cards) == 3)
        if len(self.runout) > 3:
            self.shared_cards.append(self.runout[3])
        else:
            self.shared_cards.append(self.pull_from_deck())

    def deal_river(self):
        assert(len(self.shared_cards) == 4)
        if len(self.runout) > 4:
            self.shared_cards.append(self.runout[4])
        else:
            self.shared_cards.append(self.pull_from_deck())

    def reset_deck(self):
        super().reset_deck()
        self.remove_cards_from_deck(self.runout)
