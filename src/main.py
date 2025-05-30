##
## Etienne POUILLE, 2025
## IA test
## File description:
## main
##

from neurons import *
from couche import Couche
from reseau import Reseau
from embedding import *
from MLP import MLP
from tokenizer import *
from entropy import cross_entropy_loss, softmax_cross_entropy_grad
from generate import generate
from charger_data import data, vocabulaire, inv_vocab

taille_vocab: int = len(vocabulaire)
taille_embedding: int = 4
taille_contexte: int = 2

embedding = Embedding(vocab_size=taille_vocab, dim=taille_embedding)
mlp = MLP(
    input_dim=taille_contexte * taille_embedding,
    hidden_dim=16,
    output_dim=taille_vocab
)

for epoch in range(1000):
    total_loss = 0
    for context, target in data:
        x_embed = embedding.forward(context)
        x_flat = [val for vec in x_embed for val in vec]
        y_pred = mlp.forward(x_flat)
        loss = cross_entropy_loss(y_pred, target)
        total_loss += loss
        grad_out = softmax_cross_entropy_grad(y_pred, target)
        grad_input = mlp.backward(grad_out, lr=0.05)
        update_embeddings(embedding, context, grad_input, lr=0.05)
    # if epoch % 100 == 0:
    #     print(f"Époque {epoch}, perte: {total_loss / len(data):.4f}")

contexte = ["tout", "va"]
print(" ".join(generate(embedding, mlp, vocabulaire, inv_vocab, contexte, max_len=2)))
