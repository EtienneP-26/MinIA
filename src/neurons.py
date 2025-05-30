##
## Etienne POUILLE, 2025
## IA test
## File description:
## neurons
##

import math
import random

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

class Neurone:
    def __init__(self, n_entrees, learning_rate=0.1):
        self.poids = [random.uniform(-1, 1) for _ in range(n_entrees)]
        self.biais = random.uniform(-1, 1)
        self.lr = learning_rate

    def forward(self, entrees):
        self.entrees = entrees
        self.z = sum(w * x for w, x in zip(self.poids, entrees)) + self.biais
        self.a = sigmoid(self.z)
        return self.a

    def backward(self, delta):
        grad = sigmoid_derivative(self.a)
        d = delta * grad
        for i in range(len(self.poids)):
            self.poids[i] -= self.lr * d * self.entrees[i]
        self.biais -= self.lr * d
        return [d * w for w in self.poids]
