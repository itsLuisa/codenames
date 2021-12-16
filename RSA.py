from sklearn.preprocessing import normalize
import numpy as np
from statistics import mean

# change everything to similarity
def create_meaning_matrix(all_clues, just_combos, model, mean_or_prod="mean"):
    """creates meaning matrix with: rows = clues & columns = guess combos"""
    meaning_matrix = list()
    for (c, i) in all_clues:
        arr = list()
        for jc in just_combos:
            if mean_or_prod == "mean":
                m = mean([model.distance(w, c) for w in jc])
            else:
                m = np.prod([(model.distance(w, c) + 1) / 2 for w in jc]) # + 1 / 2 is for normalizing the cosine [-1;1] to [0;1], should not make any difference
            arr.append(m)
        arr = np.array(arr)
        #print(c, arr)
        meaning_matrix.append(arr)
    meaning_matrix = np.array(meaning_matrix)
    print("meaning matrix:\n", meaning_matrix)
    return meaning_matrix


def RSA(meaning_matrix, just_combos):
    """takes meaning matrix and computes pragmatic listener matrix"""
    literal_listener = normalize(meaning_matrix, norm='l1', axis=1)
    print("literal listener:\n", literal_listener)

    pragmatic_speaker = normalize(literal_listener, norm='l1', axis=0)
    print("pragmatic speaker:\n", pragmatic_speaker)

    pragmatic_listener = normalize(pragmatic_speaker, norm='l1', axis=1)
    print("pragmatic listener:\n", pragmatic_listener)

    # finding the best guess
    best_prag_guess = just_combos[list(pragmatic_listener[0]).index(min(pragmatic_listener[0]))]
    return best_prag_guess
