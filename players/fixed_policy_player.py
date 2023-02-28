from .player import Player

class FixedPolicyPlayer(Player):
    # override base class
    def act(self, game, price):
        if price == 0:
            return ('bet', min(10, self.bankroll))
        else:
            if price > self.bankroll:
                return ('fold', None)
            return ('call', None)
