import json
from semantic_models import GloVe_Model
from baseline_model import make_guess, produce_clue, google_words
from statistics import mean
from itertools import combinations

# extract example
with open("codenamesexp.json", encoding="utf-8") as f:
    data = json.load(f)
    #clue = data["game 3"]["round 1"]["clue"]
    clue = data["game 3"]["round 5"]["clue"]
    gold_guesses = data["game 3"]["round 5"]["guesses"]
    remaining_words = data["game 3"]["round 5"]["remaining words"]

# make guesses
model = GloVe_Model()
rank = model.get_hierarchy(clue[0], remaining_words)

# extract word combinations and mean distance to clue
combos = dict()
for c in combinations(rank, clue[1]):
    words = tuple(i[0] for i in c)
    m = mean([i[1] for i in c])
    combos[words] = m

# sort combos
sorted_combos = sorted(combos.items(), key=lambda x: x[1])

# shorten the combo list
short_combos = list()
i = 0
for (w,d) in sorted_combos:
    if i == 50:
        break
    short_combos.append((w,d))
    #if "battleship" in w and "crusader" in w: # here I have to find a more general threshold that makes sense
        #break
    i += 1

print(len(short_combos), short_combos)

# find alternative clues in the traditional Kim et al way
alternative_clues = list()
google_words_l = google_words(model)
for c in short_combos:
    target_words = list(c[0])
    print("target words:", target_words)
    bad_words = [i for i in remaining_words if i not in target_words]
    #print(bad_words)
    new_clue = produce_clue(target_words, bad_words, google_words_l, model)
    print("alternative clue:", new_clue)
    alternative_clues.append(new_clue)

# find alternative clues in a (hopefully) faster, vectorbased way
'''for c in shorted_combos:
    target_words = list(c[0])
    print(target_words)
    new_clue = model.closest_words()
'''

# assemble clue combos
clues = [tuple(clue)] + alternative_clues
print(clues)
print(short_combos)
