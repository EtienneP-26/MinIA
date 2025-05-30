##
## Etienne POUILLE, 2025
## MinIA
## File description:
## charger_data
##

from sys import stderr, exit
from tokenizer import creer_vocabulaire, encoder_texte, creer_sequences_d_entrainement

def load_data_file(filename: str) -> str:
    try:
        with open(filename, "r", encoding="utf-8") as f:
            texte = f.read().lower()
    except FileNotFoundError:
        print(f"Le fichier '{filename}' est introuvable.", file=stderr)
        exit(127) # file not found
    return texte

texte: str = load_data_file("data/simple_conversation.txt")
vocabulaire: dict[str, int] = creer_vocabulaire(texte)
sequence_ids: list[int] = encoder_texte(texte, vocabulaire)
data: list[tuple[list[int], int]] = creer_sequences_d_entrainement(sequence_ids, taille_contexte=2)
inv_vocab: dict[int, str] = {idx: mot for mot, idx in vocabulaire.items()}


