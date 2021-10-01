from semantic_models import GloVe_Model
from baseline_model import make_guess, produce_clue, google_words
import json
from prepareRSA import *

alt_clues = dict()
model = GloVe_Model()

with open("data/codenamesexp.json", encoding="utf-8") as f:
    data = json.load(f)

with open("results/rsa_eval_somers_250.tsv", "w", encoding="utf-8") as g:
    first_line = "original clue\tbest literal guess\tbest pragmatic guess\thuman guess\n"
    g.writelines(first_line)

for game in data:
    for round in data[game]:
        unknown_word = False
        if round != "color distribution":
            remaining_words = data[game][round]["remaining words"]
            clue = data[game][round]["clue"]
            gold_guesses = data[game][round]["guesses"]
            if clue[0] not in model.embeddings:
                continue
            if clue[1] == "inf":
                continue
            for word in remaining_words:
                if word not in model.embeddings:
                    unknown_word = True
            if unknown_word:
                continue
            original_clue, best_lit_guess, best_prag_guess, human_guess = rsa_based_guess(clue, gold_guesses, remaining_words, alt_clues, model, 2)
            with open("results/rsa_eval_somers_250.tsv", "a", encoding="utf-8") as h:
                line = str(original_clue) + "\t" + str(best_lit_guess) + "\t" + str(best_prag_guess) + "\t" + str(human_guess) + "\n"
                h.writelines(line)
