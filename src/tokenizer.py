##
## Etienne POUILLE, 2025
## min-IA
## File description:
## tokenizer
##

def text_separator(texte: str, delimitors: str) -> list[str]:
    liste_de_mots: list[str] = []
    mot_act: str = ""

    for letter in texte:
        if letter in delimitors:
            if mot_act == "":
                continue
            liste_de_mots.append(mot_act)
            mot_act = ""
        else:
            mot_act += letter
    if mot_act != "":
        liste_de_mots.append(mot_act)
    return liste_de_mots

def creer_vocabulaire(texte: str) -> dict[str, int]:
    vocabulaire: dict[str, int] = {}
    index: int = 0
    sub_str: list[str] = text_separator(texte, ", \n\r\t\f!?:;./")

    for mot in sub_str:
        if mot not in vocabulaire:
            vocabulaire[mot] = index
            index += 1
    return vocabulaire

def encoder_texte(texte: str, vocabulaire: dict[str, int]) -> list[int]:
    encoded_list: list[int] = []

    for mot in text_separator(texte, ", \n\r\t\f!?:;./"):
        if mot in vocabulaire:
            encoded_list.append(vocabulaire[mot])
    return encoded_list

def creer_sequences_d_entrainement(sequence_ids: list[int], taille_contexte: int) -> list[tuple[list[int], int]]:
    sequence_train: list[tuple[list[int], int]] = []
    liste_contexte: list[int] = []
    next_id: int = -1

    for i in range(len(sequence_ids) - taille_contexte):
        for j in range(taille_contexte):
            liste_contexte.append(sequence_ids[i + j])
        next_id = sequence_ids[i + taille_contexte]
        sequence_train.append((liste_contexte, next_id))
        liste_contexte = []
        next_id = -1
    return sequence_train
