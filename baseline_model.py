from semantic_models import GloVe_Model #, Word2Vec_Model
from collections import defaultdict
from itertools import combinations
import json
from statistics import mean

def make_guess(clue, board, model):
    best = ""
    d = float("inf")
    for word in board:
        if model.distance(word,clue) < d:
            d = model.distance(word,clue)
            best = word
    return best

def produce_clue(red_words, bad_words, words, model):
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
    #d = float("inf")
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
                        #print(rc, w, i, dr)
                        d = dr
                        best = w
                        Ci = i
    return best, Ci

def google_words(model):
    # words that the model can choose the clue from (10,000 most common english words)
    w = list()
    with open("google-10000-english.txt", encoding="utf-8") as f:
        for line in f:
            line = line.split()
            word = line[0]
            if word in model.embeddings:
                w.append(word)
    return w

def main():
    model = GloVe_Model()
    red_words = ["tap", "leaf", "bermuda", "spider", "ram", "rope", "stadium", "bill"]
    bad_words = ["alaska", "cast", "dentist", "manicure", "stick", "wool", "racket", "iron", "comic", "bugle", "ranch", "block", "dress", "plate", "vampire", "poison", "van"]
    #red_words = ["yard", "pew", "casino", "chip", "court", "australia", "minute", "blues", "brush"]
    #bad_words = ["earth", "marathon", "row", "wave", "mexico", "toast", "heart", "whale", "table", "road", "bar", "suit", "lead", "paint", "computer", "pass"]

    board = red_words + bad_words

    # simulating one round of the game

    # producing cue
    clue = produce_clue(red_words, bad_words, google_words(model), model)
    print("--- GloVe gives the following clue:", clue, "---")

    # guessing based on clue
    print("--- GloVe makes the following guess(es): ---")
    for i in range(clue[1]):
        guess = make_guess(clue[0], board, model)
        print(guess)
        board.remove(guess)

    # ranking guesses in annotated data
    with open("data_MCDM.json", encoding="utf-8") as f:
        data = json.load(f)

    with open("baseline_ranking_data3.tsv", "w", encoding="utf-8") as g:
        first_line = "clue\thuman guess\tmodel guess\taverage rank of correct guesses\tmodel score\n"
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

                # write down
                with open("baseline_ranking_data3.tsv", "a", encoding="utf-8") as g:
                    line = str(clue) + "\t" + str(gold_guesses) + "\t" + str(rank) + "\t" + str(avg_rank) + "\t" + str(score) + "\n"
                    g.writelines(line)


if __name__ == "__main__":
    main()