from glove_model import GloVe_Model
from glossbert_model import GlossBERT_Model
import json
from models import baseline_glove, gloss_bert, rsa_glove, rsa_gloss_bert
from BERT import bert_based_guess
from RSA_models import RSA_Pairs, RSA_Gloss, RSA, RSA_Pairs_Gloss


def score_rank(ranked_board, clue, gold_guesses):
    ranking = list()
    for combo in ranked_board:
        for c in combo:
            ranking.append(c)
    ranking = list(dict.fromkeys(ranking))
    score = sum([1 if g in ranking[:clue[1]] else 0 for g in gold_guesses])
    rank = ((sum([ranking.index(g) for g in gold_guesses])) - sum(range(len(gold_guesses)))) / len(gold_guesses)
    return ranking, score, rank


def eval(data, outputfile):
    #glove = GloVe_Model()
    #gloss = GlossBERT_Model()
    possible_points = 0

    rsa = RSA_Pairs_Gloss("similarities/bert_similarities.json")

    #rsa2 = RSA_Pairs("similarities/glove_similarities.json", "glove.42B.300d/glove_medium.txt")
    #rsa3 = RSA("similarities/glove_similarities.json")
    #rsa = RSA_Pairs_Gloss("similarities/bert_similarities.json")
    with open(outputfile, "w", encoding="utf-8") as f:
        first_line = "clue;clue_no;human_guesses;guesses;score;rank\n"
        f.writelines(first_line)
    for turn in data:
        remaining_words = data[turn]["remaining words"]
        clue = data[turn]["clue"]
        guesses = data[turn]["guesses"]
        possible_points += clue[1]
        #ranked_board1, score1, rank1 = baseline_glove(remaining_words, clue, glove, guesses)

        #ranked_board1 = bert_based_guess(remaining_words, clue)
        ranked_board1, score1, rank1 = rsa_gloss_bert(remaining_words, clue, rsa, guesses)
        #ranked_board = rsa_glove(remaining_words, clue, rsa)

        #ranked_board1 = rsa.perform_RSA(remaining_words, clue)
        #ranking = [i for i, j in ranked_board1]
        #score1 = sum([1 if g in ranking[:clue[1]] else 0 for g in guesses])
        #rank1 = ((sum([ranking.index(g) for g in guesses])) - sum(range(len(guesses)))) / len(guesses)

        #ranked_board2, score2, rank2 = score_rank(rsa.perform_RSA(remaining_words, clue, 2), clue, guesses)
        #ranked_board3, score3, rank3 = score_rank(rsa2.perform_RSA(remaining_words, clue, 2), clue, guesses)
        #ranked_board4, score4, rank4 = score_rank(rsa3.perform_RSA(remaining_words, clue), clue, guesses)
        #points1 = sum([1 if g in ranked_board1[:clue[1]] else 0 for g in guesses]) # for single rankings
        #points2 = sum([1 if g in ranked_board2[:clue[1]] else 0 for g in guesses])  # for single rankings
        #points = sum([1 if g in ranked_board[0] else 0 for g in guesses]) # for paired rsa
        #try:
            #rank1 = ((sum([ranked_board1.index(g) for g in guesses])) - sum(range(clue[1]))) / clue[1]
            #non_fails += 1
            #avg_rank += rank
        #except ValueError:
            #rank1 = "n/a"

        #rank2 = ((sum([ranked_board2.index(g) for g in guesses])) - sum(range(clue[1]))) / clue[1]
        #print(clue, guesses, ranked_board1, points1, rank1)
        #print(clue, guesses, ranked_board2, points2, rank2)
        print(clue, guesses, ranked_board1[:5], score1, rank1)
        print("")
        #model_points += points
        with open(outputfile, "a", encoding="utf-8") as f:
            #line = str(clue[0]) + ";" + str(clue[1]) + ";" + str(guesses) + ";" + str(ranked_board[clue[1]]) + ";" + str(score) + ";" + str(rank) + ";" + str(ranked_board1[:clue[1]]) + ";" + str(score1) + ";" + str(rank1) + ";" + str(ranked_board2[:clue[1]]) + ";" + str(score2) + ";" + str(rank2) + ";" + str(ranked_board3[:clue[1]]) + ";" + str(score3) + ";" + str(rank3) + ";" + str(ranked_board4[:clue[1]]) + ";" + str(score4) + ";" + str(rank4) + "\n"
            line = str(clue[0]) + ";" + str(clue[1]) + ";" + str(guesses) + ";" + str(ranked_board1[:clue[1]]) + ";" + str(score1) + ";" + str(rank1) + "\n"
            f.writelines(line)
    #print("score:", model_points, "/", possible_points, (model_points/possible_points)*100, "%")
    #print("avg rank:", avg_rank / (len(data.keys())))
    #print(avg_rank/non_fails)


def main():
    outputfile = "results/gloss_eval_rsa.csv"
    #outputfile = "results/baseline_eval.csv"
    with open("data/eval_clean_gloss.json", encoding="utf-8") as f:
        data = json.load(f)
    eval(data, outputfile)


if __name__ == "__main__":
    main()
