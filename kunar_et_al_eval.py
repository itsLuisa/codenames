import json
from models import baseline_glove, rsa_glove, gloss_bert, rsa_gloss_bert
from glove_model import GloVe_Model
from glossbert_model import GlossBERT_Model
from RSA_models import RSA_Pairs, RSA_Pairs_Gloss
from itertools import combinations


def eval(outputfile):
    with open("data/boards.json", encoding="utf-8") as g:
        boards = json.load(g)

    with open(outputfile, "w", encoding="utf-8") as h:
        firstline = "clue;clue_no;gold_guesses;model_guesses;top-5-acc;rank\n"
        h.writelines(firstline)

    #baseline = GloVe_Model()
    gloss = GlossBERT_Model()
    #glove_rsa = RSA_Pairs("similarities/glove_similarities_kunar.json", "glove.42B.300d/glove_short.txt")
    #gloss_rsa = RSA_Pairs_Gloss("similarities/bert_similarities.json")

    with open("data/final_board_clues_all.csv", encoding="utf-8") as f:
        for line in f:
            if "Experiment" not in line:
                lline = line.strip().split(",")
                exp = lline[0]
                board_id = lline[1]
                word1 = lline[2].lower()
                word2 = lline[3].lower()
                clue = lline[4]
                exp = exp.lower()
                #print(board_id)
                board_id = board_id[9:]
                #print(board_id)
                id_b = exp + "_board" + board_id + "_words"
                print((clue, 2), boards[id_b])
                try:
                    #ranking = glove_rsa.perform_RSA(boards[id_b], (clue, 2))
                    ranking = gloss.get_hierarchy((clue, 2), boards[id_b])
                    #ranking = list()
                    #for w1, w2 in combinations(boards[id_b], 2):
                        #ranking.append((w1, w2, baseline.similarity(clue, w1) * baseline.similarity(clue, w2)))
                    #ranking = sorted(ranking, key=lambda x: x[2], reverse=True)
                    #print(ranking)
                    #ranking = [(i, j) for i, j, k in ranking]
                    print(ranking)
                    # sort the pairs first
                    top5acc = 1 if (word1, word2) in ranking[:5] or (word2, word1) in ranking[:5] else 0
                    print(top5acc)
                    try:
                        rank = ranking.index((word1, word2))
                    except ValueError:
                        rank = ranking.index((word2, word1))
                    print(rank)

                    with open(outputfile, "a", encoding="utf-8") as k:
                        #line = "clue;clue_no;gold_guesses;model_guesses;top-5-acc;rank\n"
                        line = str(clue) + ";" + str(2) + ";" + str(word1) + ", " + str(word2) + ";" + str(ranking[0]) + ";" + str(top5acc) + ";" + str(rank) + "\n"
                        k.writelines(line)

                except ValueError:
                    pass


def main():
    outputfile = "results/kunar_gloss.csv"
    eval(outputfile)


if __name__ == "__main__":
    main()
