##
## Etienne POUILLE, 2025
## IA test
## File description:
## MLP
##

import math
import random

class MLP:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.W1 = [[random.uniform(-1, 1) for _ in range(input_dim)] for _ in range(hidden_dim)]
        self.b1 = [0.0 for _ in range(hidden_dim)]

        self.W2 = [[random.uniform(-1, 1) for _ in range(hidden_dim)] for _ in range(output_dim)]
        self.b2 = [0.0 for _ in range(output_dim)]

    def relu(self, x: list[float]) -> list[float]:
        return [max(0, xi) for xi in x]

    def relu_derivative(self, x: list[float]) -> list[float]:
        return [1 if xi > 0 else 0 for xi in x]

    def dot(self, w: list[float], x: list[float]) -> float:
        return sum(wi * xi for wi, xi in zip(w, x))

    def linear(self, W, b, x):
        return [self.dot(Wi, x) + bi for Wi, bi in zip(W, b)]

    def softmax(self, x):
        max_val = max(x)
        exps = [math.exp(xi - max_val) for xi in x]
        sum_exps = sum(exps)
        return [e / sum_exps for e in exps]

    def forward(self, x: list[float]) -> list[float]:
        self.x = x
        self.h = self.linear(self.W1, self.b1, x)
        self.h_relu = self.relu(self.h)
        self.out = self.linear(self.W2, self.b2, self.h_relu)
        self.y_pred = self.softmax(self.out)
        return self.y_pred

    def backward(self, grad_output: list[float], lr: float):
        for i in range(self.output_dim):
            for j in range(self.hidden_dim):
                self.W2[i][j] -= lr * grad_output[i] * self.h_relu[j]
            self.b2[i] -= lr * grad_output[i]
        grad_h_relu = [0.0 for _ in range(self.hidden_dim)]
        for j in range(self.hidden_dim):
            for i in range(self.output_dim):
                grad_h_relu[j] += grad_output[i] * self.W2[i][j]
        grad_h = [grad_h_relu[j] * self.relu_derivative([self.h[j]])[0] for j in range(self.hidden_dim)]
        for j in range(self.hidden_dim):
            for k in range(self.input_dim):
                self.W1[j][k] -= lr * grad_h[j] * self.x[k]
            self.b1[j] -= lr * grad_h[j]
        grad_input = [0.0 for _ in range(self.input_dim)]
        for k in range(self.input_dim):
            for j in range(self.hidden_dim):
                grad_input[k] += grad_h[j] * self.W1[j][k]
        return grad_input
