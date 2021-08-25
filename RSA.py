from sklearn.preprocessing import normalize
import numpy as np
from semantic_models import GloVe_Model
from statistics import mean
from prepareRSA import clues, short_combos, model

#model = GloVe_Model()
#clues = [('war', 2), ('play', 2), ('thought', 2), ('warrior', 2)]
#short_combos = [(('well', 'game'), 0.52), (('well', 'saw'), 0.55), (('battleship', 'crusader'), 0.66)]

# create meaning matrix with
# rows = clues
# columns = combos
sc_nonum = [a for a, b in short_combos]
print(sc_nonum)

meaning_matrix = list()
for (c, i) in clues:
    arr = list()
    for (sc, n) in short_combos:
        m = mean([model.distance(w,c) for w in sc])
        arr.append(m)
    arr = np.array(arr)
    print(c, arr)
    meaning_matrix.append(arr)

meaning_matrix = np.array(meaning_matrix)
print("meaning matrix:\n", meaning_matrix)

"""axis = 0 indicates, normalize by column and if you are 
interested in row normalization just give axis = 1"""

# RSA
literal_listener = normalize(meaning_matrix, norm='l1', axis=1)
print("literal listener:\n", literal_listener)

pragmatic_speaker = normalize(literal_listener, norm='l1', axis=0)
print("pragmatic speaker:\n", pragmatic_speaker)

pragmatic_listener = normalize(pragmatic_speaker, norm='l1', axis=1)
print("pragmatic listener:\n", pragmatic_listener)

# finding the best guess
best_prag_guess = sc_nonum[list(pragmatic_listener[0]).index(min(pragmatic_listener[0]))]
print("original clue:", clues[0])
print("best literal guess:", sc_nonum[0])
print("best pragmatic guess:", best_prag_guess)