import json
from strategic_component import *


def main():
    with open("data/eval_data.json", encoding="utf-8") as f:
        data = json.load(f)

    with open("results/strategic_rsa_eval_data.tsv", "w", encoding="utf-8") as g:
        first_line = "clues\tguesses\tscore\n"
        g.writelines(first_line)

    model = GloVe_Model()
    for game in data:
        colors = data[game]["color distribution"]
        for round in data[game]:
            unknown_word = False
            if round != "color distribution":
                clue = data[game][round]["clue"]
                gold_guesses = data[game][round]["guesses"]
                remaining_words = data[game][round]["remaining words"]
                team = data[game][round]["team"]
                previous_clues = data[game][round]["open clues"]
                if clue[0] not in model.embeddings:
                    continue
                if clue[1] == "inf":
                    continue
                for word in remaining_words:
                    if word not in model.embeddings:
                        unknown_word = True
                if unknown_word:
                    continue

                guesses = make_guesses(clue, previous_clues, gold_guesses, remaining_words, model, team)
                if team in previous_clues:
                    all_clues = [clue] + previous_clues[team]
                else:
                    all_clues = clue
                print(all_clues)
                print(guesses)
                fewer_guesses = decide_for_guesses(guesses, clue[1])
                print(fewer_guesses)
                right = 0
                wrong = 0
                right, wrong, score = check_guesses(fewer_guesses, colors, team, right, wrong)
                print("right:", right, "wrong:", wrong, "score:", score)

                with open("results/strategic_rsa_eval_data.tsv", "a", encoding="utf-8") as g:
                    line = str(all_clues) + "\t" + str(fewer_guesses) + "\t" + str(score) + "\n"
                    g.writelines(line)


if __name__ == "__main__":
    main()
