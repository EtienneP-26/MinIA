##
## Etienne POUILLE, 2025
## IA test
## File description:
## couche
##

from neurons import Neurone

class Couche:
    def __init__(self, n_entrees, n_neurones, learning_rate=0.1):
        self.neurones = [Neurone(n_entrees, learning_rate) for _ in range(n_neurones)]

    def forward(self, entrees):
        self.entrees = entrees
        self.sorties = [n.forward(entrees) for n in self.neurones]
        return self.sorties

    def backward(self, deltas):
        new_deltas = [0] * len(self.neurones[0].poids)
        for neurone, delta in zip(self.neurones, deltas):
            contribution = neurone.backward(delta)
            new_deltas = [nd + c for nd, c in zip(new_deltas, contribution)]
        return new_deltas
