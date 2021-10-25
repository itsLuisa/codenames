from semantic_models import GloVe_Model
from baseline_model import make_guess, produce_clue
import json
from prepareRSA import *


def make_guesses(method, dataset, input_data, output_data, mean_or_prod):
    alt_clues = dict()
    model = GloVe_Model()

    with open(input_data, encoding="utf-8") as f:
        data = json.load(f)

    with open(output_data, "w", encoding="utf-8") as g:
        first_line = "original clue\tbest literal guess\tbest pragmatic guess\thuman guess\tscore\tpoints\n"
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

                if method == "clue_finder":
                    original_clue, best_lit_guess, best_prag_guess, human_guess = rsa_based_guess_alt_clues(clue, gold_guesses, remaining_words, alt_clues, model, dataset, mean_or_prod)
                else:
                    original_clue, best_lit_guess, best_prag_guess, human_guess = rsa_based_guess_full_dataset(clue, gold_guesses, remaining_words, model, dataset, mean_or_prod)

                flag = True
                i = -1
                while flag:
                    i += 1
                    try:
                        flag = best_prag_guess[i] in human_guess
                    except IndexError:
                        flag = False
                score = i

                points = 0
                for g in best_prag_guess:
                    if g in human_guess:
                        points += 1

                with open(output_data, "a", encoding="utf-8") as h:
                    line = str(original_clue) + "\t" + str(best_lit_guess) + "\t" + str(best_prag_guess) + "\t" + str(human_guess) + "\t" + str(score) + "\t" + str(points) + "\n"
                    h.writelines(line)


def main():
    make_guesses("whole_dataset", "google", "data/codenamesexp.json", "results/rsa_eval_ZiS_full_google_mean.tsv", "mean")


if __name__ == "__main__":
    main()
