from semantic_models import GloVe_Model
from collections import defaultdict
from itertools import combinations
import json
from statistics import mean


def make_guess(clue, board, model):
    """
    find the best fitting board word given the clue
    :param clue: string of current clue word
    :param board: list of current board words
    :param model: instance of the GloVe class
    :return: string of best match on the board given the clue
    """
    best = ""
    d = float("inf")
    for word in board:
        if model.distance(word,clue) < d:
            d = model.distance(word,clue)
            best = word
    return best


def produce_clue(red_words, bad_words, words, model):
    """
    finds the best possible clue for the current game state from a set of possible clues
    :param red_words: list of words that the clue can refer to
    :param bad_words: list of words that the clue shouldn't refer to
    :param words: list of possible clues
    :param model: instance of GloVe class
    :return: tuple of clue word and number
    """
    R = defaultdict(dict)
    for r_w in red_words:
        for w in words:
            R[r_w][w] = model.distance(r_w, w)
    B = defaultdict(dict)
    for b_w in bad_words:
        for w in words:
            B[b_w][w] = model.distance(b_w, w)
    Ci = 0
    best = ""
    for i in range(1, len(red_words)+1):
        d = float("inf")
        for rc in combinations(red_words, i):
            for w in words:
                if w not in red_words and w not in bad_words:
                    wd = float("inf")
                    for b_w in bad_words:
                        if B[b_w][w] < wd:
                            wd = B[b_w][w]
                    dr = 0
                    for r_w in rc:
                        if R[r_w][w] > dr:
                            dr = R[r_w][w]
                    if dr < d and dr < wd and dr < 1: # threshold [0;1] to determine the models aggressiveness (the higher the more aggressive)
                        d = dr
                        best = w
                        Ci = i
    return best, Ci


def eval_baseline(input_data, output_data):
    model = GloVe_Model()

    with open(input_data, encoding="utf-8") as f:
        data = json.load(f)

    with open(output_data, "w", encoding="utf-8") as g:
        first_line = "clue\thuman guess\tmodel guess\taverage rank of correct guesses\tmodel score\tpoints\n"
        g.writelines(first_line)

    for game in data:
        for round in data[game]:
            rank = list()
            unknown_word = False
            if round != "color distribution":
                remaining_words = data[game][round]["remaining words"]
                for word in remaining_words:
                    if word not in model.embeddings:
                        unknown_word = True
                if unknown_word:
                    continue
                clue = data[game][round]["clue"]
                gold_guesses = data[game][round]["guesses"]
                if clue[0] not in model.embeddings:
                    continue
                if clue[1] == "inf":
                    continue

                while remaining_words:
                    model_guess = make_guess(clue[0], remaining_words, model)
                    rank.append(model_guess)
                    remaining_words.remove(model_guess)

                # compute average rank
                avg_rank = mean([rank.index(guess) for guess in gold_guesses])

                # compute score
                flag = True
                i = -1
                while flag:
                    i += 1
                    flag = rank[i] in gold_guesses
                score = i

                # compute points
                points = 0
                for g in rank[:clue[1]]:
                    if g in gold_guesses:
                        points += 1

                # write down
                with open(output_data, "a", encoding="utf-8") as g:
                    line = str(clue) + "\t" + str(gold_guesses) + "\t" + str(rank) + "\t" + str(avg_rank) + "\t" + str(score) + "\t" + str(points) +"\n"
                    g.writelines(line)


def main():
    #red_words = ["tap", "leaf", "bermuda", "spider", "ram", "rope", "stadium", "bill"]
    #bad_words = ["alaska", "cast", "dentist", "manicure", "stick", "wool", "racket", "iron", "comic", "bugle", "ranch", "block", "dress", "plate", "vampire", "poison", "van"]

    #board = red_words + bad_words

    eval_baseline("data/eval_data_clean.json", "results/baseline_ranking_eval.tsv")


if __name__ == "__main__":
    main()
