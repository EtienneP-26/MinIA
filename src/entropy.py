##
## Etienne POUILLE, 2025
## IA test
## File description:
## entropy
##

import math

def cross_entropy_loss(y_pred: list[float], target_id: int) -> float:
    epsilon = 1e-12 
    return -math.log(y_pred[target_id] + epsilon)

def softmax_cross_entropy_grad(y_pred: list[float], target_id: int) -> list[float]:
    grad = y_pred[:]
    grad[target_id] -= 1
    return grad
