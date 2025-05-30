##
## Etienne POUILLE, 2025
## MinIA
## File description:
## generate
##

from embedding import Embedding
from MLP import MLP

def generate(
        embedding: Embedding,
        mlp: MLP,
        vocabulaire: dict[str, int],
        inv_vocab: dict[int, str],
        contexte: list[str],
        max_len: int = 10
    ) -> list[str]:
    """
    Fonction pour générer du texte avec un modèle
    """

    contexte_ids: list[str] = [vocabulaire[mot] for mot in contexte[-2:]]
    resultat: list[str] = contexte[:]

    for _ in range(max_len):
        while len(contexte_ids) < 2:
            contexte_ids = [0] + contexte_ids
        x_embed = embedding.forward(contexte_ids[-2:])
        x_flat = [val for vec in x_embed for val in vec]
        y_pred = mlp.forward(x_flat)
        next_id = y_pred.index(max(y_pred))
        next_word = inv_vocab[next_id]
        resultat.append(next_word)
        contexte_ids.append(next_id)
    return resultat
