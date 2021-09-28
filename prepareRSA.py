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
        clue = data["game 1"]["round 2"]["clue"]
        #clue = ["water", 3]
        gold_guesses = data["game 1"]["round 2"]["guesses"]
        remaining_words = data["game 1"]["round 2"]["remaining words"]
    return clue, gold_guesses, remaining_words


def get_word_combos(model, clue, remaining_words):
    """extracts word combinations and mean distance to clue and sort these combinations in descending order"""
    rank = model.get_hierarchy(clue[0], remaining_words)
    rank = rank[:11]
    combos = dict()
    for c in combinations(rank, clue[1]):
        words = tuple(i[0] for i in c)
        m = mean([i[1] for i in c])
        combos[words] = m
    sorted_combos = sorted(combos.items(), key=lambda x: x[1])
    return sorted_combos


def find_alternative_clues(google_words_list, model, short_combos, remaining_words, alt_clues):
    """assembles alternative clues in the traditional Kim et al way"""
    alternative_clues = list()
    for c in short_combos:
        target_words = list(c[0])
        print("target words:", target_words)
        if frozenset(target_words) in alt_clues:
            new_clue = alt_clues[frozenset(target_words)]
        else:
            bad_words = [i for i in remaining_words if i not in target_words]
            new_clue = produce_clue(target_words, bad_words, google_words_list, model)
            alt_clues[frozenset(target_words)] = new_clue
        print("alternative clue:", new_clue)
        alternative_clues.append(new_clue)
    return alternative_clues


def find_alt_clues_vector(model, short_combos, remaining_words, alt_clues):
    """find alternative clues in a, vector-based way insp. by Somers"""
    alternative_clues = list()
    for c in short_combos:
        target_words = list(c[0])
        print("target words:", target_words)
        if frozenset(target_words) in alt_clues:
            new_clue = alt_clues[frozenset(target_words)]
        else:
            bad_words = [i for i in remaining_words if i not in target_words]
            new_clue = model.candidates(target_words, bad_words)[0][0]
            new_clue = tuple((new_clue, len(target_words)))
            alt_clues[frozenset(target_words)] = new_clue
        print("alternative clue:", new_clue)
        alternative_clues.append(new_clue)
    return alternative_clues


def rsa_based_guess(clue, gold_guesses, remaining_words, alt_clues, model, mode=1):
    sorted_combos = get_word_combos(model, clue, remaining_words)
    print(len(sorted_combos), sorted_combos)

    # shorten the combo list
    short_combos = sorted_combos[:200]
    print(len(short_combos), short_combos)

    if mode == 1:
        print("kim et al")
        google_words_list = google_words(model)
        alternative_clues = find_alternative_clues(google_words_list, model, short_combos, remaining_words, alt_clues)

    else:
        print("somers")
        alternative_clues = find_alt_clues_vector(model, short_combos, remaining_words, alt_clues)
    print(alternative_clues)

    # assemble clue combos
    all_clues = [tuple(clue)] + alternative_clues
    print(all_clues)

    just_combos = [a for a, b in short_combos]
    print(just_combos)

    meaning_matrix = create_meaning_matrix(all_clues, just_combos, model)
    best_prag_guess = RSA(meaning_matrix, just_combos)
    print("original clue:", all_clues[0])
    print("best literal guess:", just_combos[0])
    print("best pragmatic guess:", best_prag_guess)
    print("human guess:", gold_guesses)
    original_clue = all_clues[0]
    best_lit_guess = just_combos[0]
    human_guess = gold_guesses
    return original_clue, best_lit_guess, best_prag_guess, human_guess


def main():
    clue, gold_guesses, remaining_words = extract_example()
    alt_clues = dict()
    model = GloVe_Model()
    rsa_based_guess(clue, gold_guesses, remaining_words, alt_clues, model)


if __name__ == "__main__":
    main()
