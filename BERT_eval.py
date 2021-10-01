from BERT import *
import json
from semantic_models import GloVe_Model

model = GloVe_Model()

with open("data/codenamesexp.json", encoding="utf-8") as f:
    data = json.load(f)

with open("results/bert_eval.tsv", "w", encoding="utf-8") as g:
    first_line = "original clue\tbest baseline guess\tbest bert guess\thuman guess\n"
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

            model_guess = bert_based_guess(remaining_words, clue)
            #original_clue, best_lit_guess, best_prag_guess, human_guess = rsa_based_guess(clue, gold_guesses, remaining_words, alt_clues, model)

            with open("results/bert_eval.tsv", "a", encoding="utf-8") as h:
                line = str(clue) + "\t" + "baseline insert here" + "\t" + str(model_guess) + "\t" + str(gold_guesses) + "\n"
                h.writelines(line)