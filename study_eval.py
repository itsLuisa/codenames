from collections import defaultdict, Counter
from models import baseline_glove, rsa_glove, gloss_bert, rsa_gloss_bert
from glove_model import GloVe_Model
from glossbert_model import GlossBERT_Model
from RSA_models import RSA_Pairs, RSA_Pairs_Gloss


def eval(outputfileoperative, outputfilesypmaster, model):
    with open(outputfilesypmaster, "w", encoding="utf-8") as h:
        first_line = "clue;clue_no;intended_words;board_size;guesses;score;rank\n"
        #first_line = "item;group;participant;guess;boardwords\n"
        h.writelines(first_line)

    with open("eval_study/experiment_raw_data.csv", encoding="utf-8") as f:
        participant_guesses = defaultdict(dict)
        game_state_operative = dict()
        for line in f:
            if "listnumber" not in line:
                lline = line.split("{")
                first_part = lline[0].split(",")
                second_part = lline[1].split("}")
                answer = second_part[0]
                second_part = second_part[1].split(",")
                group = int(first_part[1].strip('"'))
                workerid = first_part[4].strip('"')
                scenario = second_part[2].strip('"')
                item = second_part[3].strip('"')
                board = second_part[4].strip('"').split(" ")
                colors = second_part[5].strip('"').split(" ")
                clue = second_part[7].strip('"')
                clue_no = second_part[8].strip('"')
                if scenario == "operative":
                    answer = answer.replace('\\""', "")
                    answer = answer.split(":")[1].strip("]").strip("[")
                    guesses = answer.split(",")
                    guesses = sorted(guesses)
                    participant_guesses[item][workerid] = guesses
                    game_state_operative[item] = (clue, int(clue_no), board)
                    #with open(outputfilesypmaster, "a", encoding="utf-8") as h:
                        #line = str(item) + ";" + str(group) + ";" + str(workerid) + ";" + str(guesses) + ";" + str(board) + "\n"
                        #h.writelines(line)

                else:
                    if "clueName" in answer and "clueNo" in answer and "spymaster" in answer:
                        print("")
                        print(board)
                        answer = answer.replace('\\""', '')
                        front, middle = answer.split('[')
                        middle, back = middle.split("]")
                        intended_words = middle.split(",")
                        print(intended_words)
                        rest = front + back
                        rest = rest.replace(",spymaster:", "").split(",")
                        if "clueName" in rest[0]:
                            cluee = rest[0].split(":")[1].lower()
                            print(cluee)
                            clue_n = int(rest[1].split(":")[1])
                            print(clue_n)
                        else:
                            cluee = rest[1].split(":")[1].lower()
                            print(cluee)
                            clue_n = int(rest[0].split(":")[1])
                            print(clue_n)
                        try:
                            ranked_board, score, rank = rsa_glove(board, (cluee, clue_n), model, intended_words)
                            print(ranked_board)
                            with open(outputfilesypmaster, "a", encoding="utf-8") as h:
                                line = str(cluee) + ";" + str(clue_n) + ";" + str(intended_words) + ";" + str(len(board)) + ";" + str(ranked_board[:clue_n]) + ";" + str(score) + ";" + str(rank) + "\n"
                                h.writelines(line)
                        except KeyError:
                            pass
                        except ValueError:
                            pass
                
    '''
    with open(outputfileoperative, "w", encoding="utf-8") as f:
        first_line = "clue;clue_no;agreed_guesses;board_size;guesses;score;rank\n"
        f.writelines(first_line)

    for turn in participant_guesses:
        clue, n, board = game_state_operative[turn]
        all_guesses = list(participant_guesses[turn].values())
        guesses = list()
        for i in all_guesses:
            guesses += i
        guesses = Counter(guesses).most_common()
        guesses = [i for i, j in guesses]
        agreed_guesses = guesses[:n]
        try:
            ranking, score, rank = rsa_glove(board, (clue, n), model, agreed_guesses)
            print(ranking, score, rank, board)
            with open(outputfileoperative, "a", encoding="utf-8") as g:
                line = str(clue) + ";" + str(n) + ";" + str(agreed_guesses) + ";" + str(len(board)) + ";" + str(ranking[:n]) + ";" + str(score) + ";" + str(rank) + "\n"
                g.writelines(line)
        except IndexError:
            pass
        '''


def main():
    #model = GloVe_Model()
    #model = GlossBERT_Model()
    model = RSA_Pairs("similarities/glove_similarities.json", "glove.42B.300d/glove_medium.txt")
    #model = RSA_Pairs_Gloss("similarities/bert_similarities.json")
    outputfileoperative = "eval_study/smth.csv"
    outputfilespymaster = "eval_study/study_spymasters_baseline_rsa.csv"
    eval(outputfileoperative, outputfilespymaster, model)


if __name__ == "__main__":
    main()
