from sklearn.preprocessing import normalize
import numpy as np
from statistics import mean


def create_meaning_matrix(all_clues, just_combos, model):
    """creates meaning matrix with: rows = clues & columns = guess combos"""
    meaning_matrix = list()
    for (c, i) in all_clues:
        arr = list()
        for jc in just_combos:
            m = mean([model.distance(w,c) for w in jc])
            arr.append(m)
        arr = np.array(arr)
        print(c, arr)
        meaning_matrix.append(arr)
    meaning_matrix = np.array(meaning_matrix)
    print("meaning matrix:\n", meaning_matrix)
    return meaning_matrix


def RSA(meaning_matrix, just_combos, all_clues):
    """takes meaning matrix and computes pragmatic listener matrix"""
    literal_listener = normalize(meaning_matrix, norm='l1', axis=1)
    print("literal listener:\n", literal_listener)

    pragmatic_speaker = normalize(literal_listener, norm='l1', axis=0)
    print("pragmatic speaker:\n", pragmatic_speaker)

    pragmatic_listener = normalize(pragmatic_speaker, norm='l1', axis=1)
    print("pragmatic listener:\n", pragmatic_listener)

    # finding the best guess
    best_prag_guess = just_combos[list(pragmatic_listener[0]).index(min(pragmatic_listener[0]))]
    print("original clue:", all_clues[0])
    print("best literal guess:", just_combos[0])
    print("best pragmatic guess:", best_prag_guess)
    return best_prag_guess