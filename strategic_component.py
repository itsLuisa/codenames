import json
from prepareRSA import *
from semantic_models import GloVe_Model


# look at board/clue and previous clues (see detailed annotations of polygon and mcdm)
def extract_example():
    """extracts example turn chosen by hand for tryouts"""
    with open("data/data_Polygon.json", encoding="utf-8") as f:
        data = json.load(f)
        clue = data["game 1"]["round 3"]["clue"]
        gold_guesses = data["game 1"]["round 3"]["guesses"]
        remaining_words = data["game 1"]["round 3"]["remaining words"]
        team = data["game 1"]["round 3"]["team"]
        previous_clues = data["game 1"]["round 3"]["open clues"]
        colors = data["game 1"]["color distribution"]
    return clue, gold_guesses, remaining_words, team, previous_clues, colors


# make guess & compute certainty score for all guesses
def make_guesses(clue, previous_clues, gold_guesses, remaining_words, model, team): #previous clues only for the right team!
    guesses = list()
    #original_clue, best_lit_guess, best_prag_guess, human_guess = rsa_based_guess(clue, gold_guesses, remaining_words, alt_clues, model, 1)
    original_clue, best_lit_guess, best_prag_guess, human_guess = rsa_based_guess_full_dataset(clue, gold_guesses, remaining_words, model)
    for g in best_prag_guess:
        d = model.distance(g, original_clue[0])
        guesses.append((g, d))
    if team in previous_clues:
        for clue in previous_clues[team]:
            if clue[0] not in model.embeddings:
                continue
            #original_clue, best_lit_guess, best_prag_guess, human_guess = rsa_based_guess(clue, gold_guesses, remaining_words, alt_clues, model, 2)
            original_clue, best_lit_guess, best_prag_guess, human_guess = rsa_based_guess_full_dataset(clue, gold_guesses, remaining_words, model)
            for g in best_prag_guess:
                d = model.distance(g, original_clue[0])
                for (gg, dd) in guesses:
                    if g == gg:
                        guesses.remove((gg,dd))
                        guesses.append((g, d * dd))
                else:
                    guesses.append((g, d))
    return sorted(guesses, key=lambda x: x[1])


# make guesses based on score and threshold
def decide_for_guesses(sorted_guesses, n): # n = number that came with current clue
    #threshold = 0.5
    #fewer_guesses = [g for (g, d) in sorted_guesses if d < threshold]
    sorted_guesses = [g for (g, d) in sorted_guesses]
    if len(sorted_guesses) > n+1: # vllt ist diese Zeile auch unnötig
        sorted_guesses = sorted_guesses[:n]
    return sorted_guesses


# record whether the guesses were good or not
def check_guesses(guesses, colors, team, right, wrong):
    flag = True
    score = 0
    for g in guesses:
        if colors[g] == team:
            right += 1
            if flag:
                score += 1
        else:
            wrong += 1
            flag = False
    return right, wrong, score


def main():
    clue, gold_guesses, remaining_words, team, previous_clues, colors = extract_example()
    model = GloVe_Model()
    guesses = make_guesses(clue, previous_clues, gold_guesses, remaining_words, model, team)
    print(guesses)
    fewer_guesses = decide_for_guesses(guesses, clue[1])
    print(fewer_guesses)
    right = 0
    wrong = 0
    right, wrong, score = check_guesses(fewer_guesses, colors, team, right, wrong)
    print("right:", right, "wrong:", wrong)


if __name__ == "__main__":
    main()
