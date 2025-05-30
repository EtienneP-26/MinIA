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

mlp = MLP(input_dim=8, hidden_dim=16, output_dim=6)
embedding = Embedding(vocab_size=6, dim=4)

texte = "bonjour comment ça va bonjour bien merci"
vocabulaire = creer_vocabulaire(texte)

data = creer_sequences_d_entrainement(encoder_texte(texte, vocabulaire), taille_contexte = 2)

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

    if epoch % 100 == 0:
        print(f"Époque {epoch}, perte: {total_loss / len(data):.4f}")
