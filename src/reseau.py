##
## Etienne POUILLE, 2025
## IA test
## File description:
## reseau
##

from neurons import Neurone
from couche import Couche

class Reseau:
    def __init__(self):
        self.couche_cachee = Couche(2, 2, learning_rate=0.5)
        self.couche_sortie = Couche(2, 1, learning_rate=0.5)

    def forward(self, x):
        out1 = self.couche_cachee.forward(x)
        out2 = self.couche_sortie.forward(out1)
        return out1, out2

    def backward(self, x, y):
        out1, out2 = self.forward(x)
        erreur = [(out2[0] - y[0])]
        delta2 = self.couche_sortie.backward(erreur)
        self.couche_cachee.backward(delta2)