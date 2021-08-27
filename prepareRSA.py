import json
from semantic_models import GloVe_Model
from baseline_model import make_guess, produce_clue, google_words
from statistics import mean
from itertools import combinations
from RSA import create_meaning_matrix, RSA


def extract_example():
    """extracts example turn chosen by hand for tryouts"""
    with open("codenamesexp.json", encoding="utf-8") as f:
        data = json.load(f)
        #clue = data["game 3"]["round 1"]["clue"]
        clue = data["game 1"]["round 1"]["clue"]
        gold_guesses = data["game 1"]["round 1"]["guesses"]
        remaining_words = data["game 1"]["round 1"]["remaining words"]
    return clue, gold_guesses, remaining_words


def get_word_combos(model, clue, remaining_words):
    """extracts word combinations and mean distance to clue and sort these combinations in descending order"""
    rank = model.get_hierarchy(clue[0], remaining_words)
    combos = dict()
    for c in combinations(rank, clue[1]):
        words = tuple(i[0] for i in c)
        m = mean([i[1] for i in c])
        combos[words] = m
    sorted_combos = sorted(combos.items(), key=lambda x: x[1])
    return sorted_combos


def find_alternative_clues(model, short_combos, remaining_words):
    """assembles alternative clues in the traditional Kim et al way"""
    alternative_clues = list()
    google_words_list = google_words(model)
    for c in short_combos:
        target_words = list(c[0])
        print("target words:", target_words)
        bad_words = [i for i in remaining_words if i not in target_words]
        new_clue = produce_clue(target_words, bad_words, google_words_list, model)
        print("alternative clue:", new_clue)
        alternative_clues.append(new_clue)
    return alternative_clues


def find_alt_clues_vector():
    """find alternative clues in a (hopefully) faster, vectorbased way"""
    for c in shorted_combos:
        target_words = list(c[0])
        print(target_words)
        new_clue = model.closest_words()


def main():
    clue, gold_guesses, remaining_words = extract_example()

    model = GloVe_Model()
    sorted_combos = get_word_combos(model, clue, remaining_words)
    print(len(sorted_combos), sorted_combos)

    # shorten the combo list
    short_combos = sorted_combos[:10]
    print(len(short_combos), short_combos)

    alternative_clues = find_alternative_clues(model, short_combos, remaining_words)
    print(alternative_clues)

    # assemble clue combos
    all_clues = [tuple(clue)] + alternative_clues
    print(all_clues)

    just_combos = [a for a, b in short_combos]
    print(just_combos)

    meaning_matrix = create_meaning_matrix(all_clues, just_combos, model)
    RSA(meaning_matrix, just_combos, all_clues)

if __name__=="__main__":
    main()
