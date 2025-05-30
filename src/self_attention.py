##
## Etienne POUILLE, 2025
## MinIA
## File description:
## self_attention
##

import math
import random

def matmul(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]

def softmax(v):
    max_v = max(v)
    exps = [math.exp(x - max_v) for x in v]
    sum_exps = sum(exps)

    return [x / sum_exps for x in exps]

def transpose(matrice):
    return list(map(list, zip(*matrice)))

def init_matrix(rows, cols):
    return [[random.uniform(-0.1, 0.1) for _ in range(cols)] for _ in range(rows)]

class SelfAttention:
    def __init__(self, dim):
        self.dim = dim
        self.W_q = init_matrix(dim, dim)
        self.W_k = init_matrix(dim, dim)
        self.W_v = init_matrix(dim, dim)

    def forward(self, inputs):
        Q = matmul(inputs, self.W_q)
        K = matmul(inputs, self.W_k)
        V = matmul(inputs, self.W_v)

        scores = matmul(Q, transpose(K))
        scores = [[s / math.sqrt(self.dim) for s in row] for row in scores]

        weights = [softmax(row) for row in scores]

        output = matmul(weights, V)
        return output
