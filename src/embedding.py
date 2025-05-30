##
## Etienne POUILLE, 2025
## IA test
## File description:
## embedding
##

import random

class Embedding:
    def __init__(self, vocab_size: int, dim: int):
        self.vocab_size: int = vocab_size
        self.dim: int = dim
        self.vectors = [[random.uniform(-1, 1) for _ in range(dim)] for _ in range(vocab_size)]

    def forward(self, indices: list[int]) -> list[list[float]]:
        return [self.vectors[i] for i in indices]

    def __getitem__(self, index: int) -> list[float]:
        return self.vectors[index]

    def update(self, index: int, grad: list[float], lr: float):
        for i in range(self.dim):
            self.vectors[index][i] -= lr * grad[i]

def update_embeddings(embedding: Embedding, indices: list[int], grad_input: list[float], lr: float):
    dim = embedding.dim
    for idx_pos, word_idx in enumerate(indices):
        grad = grad_input[idx_pos * dim : (idx_pos + 1) * dim]
        embedding.update(word_idx, grad, lr)

