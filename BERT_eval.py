from BERT import *
import json
from semantic_models import GloVe_Model

model = GloVe_Model()

with open("data/codenamesexp.json", encoding="utf-8") as f:
    data = json.load(f)

with open("results/bert_eval_exsent.tsv", "w", encoding="utf-8") as g:
    first_line = "original clue\tbest baseline guess\tbest bert guess\thuman guess\tscore\tpoints\n"
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

            flag = True
            i = -1
            while flag:
                i += 1
                try:
                    flag = model_guess[i] in gold_guesses
                except IndexError:
                    flag = False
            score = i

            points = 0
            for g in model_guess:
                if g in gold_guesses:
                    points += 1

            with open("results/bert_eval_exsent.tsv", "a", encoding="utf-8") as h:
                line = str(clue) + "\t" + "baseline insert here" + "\t" + str(model_guess) + "\t" + str(gold_guesses) + "\t" + str(score) + "\t" + str(points) + "\n"
                h.writelines(line)
