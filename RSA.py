import json
from semantic_models import GloVe_Model
from baseline_model import make_guess, produce_clue, google_words
from statistics import mean
from itertools import combinations

# extract example
with open("codenamesexp.json", encoding="utf-8") as f:
    data = json.load(f)
    clue = data["game 3"]["round 1"]["clue"]
    gold_guesses = data["game 3"]["round 1"]["guesses"]
    remaining_words = data["game 3"]["round 1"]["remaining words"]

# make guesses
model = GloVe_Model()
rank = model.get_hierarchy(clue[0], remaining_words)

# extract word combinations and mean distance to clue
combos = dict()
for c in combinations(rank, clue[1]):
    words = tuple(i[0] for i in c)
    #print(words)
    m = mean([i[1] for i in c])
    #print(m)
    combos[words] = m

# sort combos
sorted_combos = sorted(combos.items(), key=lambda x: x[1])
#print(sorted_combos, len(sorted_combos))

# shorten the combo list
shorted_combos = list()
for (w,d) in sorted_combos:
    shorted_combos.append((w,d))
    if "battleship" in w and "crusader" in w:
        break
print(shorted_combos, len(shorted_combos))

# find alternative clues
alternative_clues = list()
for c in shorted_combos:
    target_words = list(c[0])
    print(target_words)
    bad_words = [i for i in remaining_words if i not in target_words]
    #print(bad_words)
    new_clue = produce_clue(target_words, bad_words, google_words(model), model)
    print(new_clue)
    alternative_clues.append(new_clue)

# assemble clue combos
clues = [tuple(clue)] + alternative_clues
print(clues)

# create RSA board / meaning matrix


# RSA