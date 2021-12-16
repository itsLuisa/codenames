from glove_model import GloVe_Model
from glossbert_model import GlossBERT_Model
from RSA_models import RSA_Pairs, RSA_Pairs_Gloss


def baseline_glove(remaining_words, clue, model1, gold_guesses):
    ranked_board = model1.get_hierarchy(clue[0], remaining_words)
    ranked_board = [i for i, j in ranked_board]
    score = sum([1 if g in ranked_board[:clue[1]] else 0 for g in gold_guesses])
    rank = ((sum([ranked_board.index(g) for g in gold_guesses])) - sum(range(len(gold_guesses)))) / len(gold_guesses)
    return ranked_board, score, rank


def gloss_bert(remaining_words, clue, model2, gold_guesses):
    ranked_board = model2.get_hierarchy(clue[0], remaining_words)
    ranked_board = [j.split("%")[0] for i, j, k in ranked_board]
    ranked_board = list(dict.fromkeys(ranked_board))
    #print(ranked_board)
    score = sum([1 if g in ranked_board[:clue[1]] else 0 for g in gold_guesses])
    rank = ((sum([ranked_board.index(g) for g in gold_guesses])) - sum(range(len(gold_guesses)))) / len(gold_guesses)
    return ranked_board, score, rank


def rsa_glove(remaining_words, clue, model3, gold_guesses):
    ranked_board = model3.perform_RSA(remaining_words, clue)
    ranking = list()
    for combo in ranked_board:
        for c in combo:
            ranking.append(c)
    ranking = list(dict.fromkeys(ranking))
    score = sum([1 if g in ranking[:clue[1]] else 0 for g in gold_guesses])
    rank = ((sum([ranking.index(g) for g in gold_guesses])) - sum(range(len(gold_guesses)))) / len(gold_guesses)
    return ranking, score, rank


def rsa_gloss_bert(remaining_words, clue, model4, gold_guesses):
    ranked_board = model4.perform_RSA(remaining_words, clue)
    ranking = list()
    for combo in ranked_board:
        for c in combo:
            ranking.append(c)
    ranking = list(dict.fromkeys(ranking))
    score = sum([1 if g in ranking[:clue[1]] else 0 for g in gold_guesses])
    rank = ((sum([ranking.index(g) for g in gold_guesses])) - sum(range(len(gold_guesses)))) / len(gold_guesses)
    return ranking, score, rank


def main():
    #remaining_words = ["ambulance", "page", "mark", "crow", "well", "ray", "square", "rabbit", "flute", "waitress", "kangaroo", "saw", "buffalo", "banana", "polo", "coach", "ice", "ear"]
    #clue = ["australia", 1]
    #gold_guesses = ["kangaroo"]
    remaining_words = ["snow", "center", "jet", "bomb", "theater", "duck", "block", "log", "dog", "olive", "pyramid", "ship", "paste", "moscow", "forest", "buck"]
    clue = ["christmas", 3]
    gold_guesses = ["snow", "forest", "duck"]
    '''
    model1 = GloVe_Model()
    print(baseline_glove(remaining_words, clue, model1, gold_guesses))
    model2 = GlossBERT_Model()
    print(gloss_bert(remaining_words, clue, model2, gold_guesses))
    remaining_words = ["ambulance", "page", "mark", "crow", "well", "ray", "square", "rabbit", "flute", "waitress", "saw", "buffalo", "banana", "polo", "coach", "ice", "ear"]
    clue = ["playground", 2]
    gold_guesses = ["square", "polo"]
    print(baseline_glove(remaining_words, clue, model1, gold_guesses))
    print(gloss_bert(remaining_words, clue, model2, gold_guesses))
    '''
    #model3 = RSA_Pairs("similarities/glove_similarities.json", "glove.42B.300d/glove_medium.txt")
    model4 = RSA_Pairs_Gloss("similarities/bert_similarities.json")
    #print(rsa_glove(remaining_words, clue, model3, gold_guesses))
    print(rsa_gloss_bert(remaining_words, clue, model4, gold_guesses))


if __name__ == "__main__":
    main()
