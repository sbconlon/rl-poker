from .player import Player

class CLIPlayer(Player):
    #override base class
    def act(self, game, price):
        print("User Action:")
        inpt = str(input())
        if inpt == 'fold':
            return (inpt, 0)
        if inpt == 'check':
            return (inpt, 0)
        if inpt == 'call':
            return (inpt, price)
        if inpt[:3] == 'bet':
            assert(price == 0)
            assert(inpt[3] == ' ')
            assert(isinstance(float(inpt[4:]), (float, int)))
            return (inpt[:3], float(inpt[4:]))
        if inpt[:5] == 'raise':
            assert(price > 0)
            assert(inpt[5] == ' ')
            assert(isinstance(float(inpt[6:]), (float, int)))
            return (inpt[:5], float(inpt[6:]))
        if inpt[:6] == 'all-in':
            return (inpt[:6], self.bankroll)
        raise Exception(inpt + " is undefined")
