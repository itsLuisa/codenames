import json
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import normalize
from itertools import combinations
from scipy.special import softmax
from glove_model import GloVe_Model


class RSA_Pairs:
    def __init__(self, similarities_file, clue_file):
        with open(similarities_file, encoding="utf-8") as f:
            self.similarities = json.load(f)
        self.alt_clues = list()
        with open(clue_file, encoding="utf-8") as g:
            for line in g:
                lline = line.split()
                word = lline[0]
                if word in self.similarities:
                    self.alt_clues.append(word)
        print("init complete")

    def perform_RSA(self, board, clue, method=2):
        rank = list()
        for word in board:
            sim = self.similarities[clue[0]][word]
            rank.append((word, sim))
        rank = sorted(rank, key=lambda x: x[1], reverse=True)
        #rank = rank[:20]
        combos = dict()
        for c in combinations(rank, clue[1]):
            words = tuple(i[0] for i in c)
            p = np.prod([i[1] for i in c])
            combos[words] = p
        sorted_combos = sorted(combos.items(), key=lambda x: x[1], reverse=True)
        short_combos = sorted_combos[:500]

        # find alternative clues
        if method == 1:
            alternative_clues = list()
            for c in short_combos:
                target_words = list(c[0])
                bad_words = [i for i in board if i not in target_words]
                new_clue = produce_clue(target_words, bad_words, self.alt_clues, self.similarities)
                alternative_clues.append(new_clue)
        else:
            alternative_clues = [(i, clue[1]) for i in self.alt_clues if i != clue[0]]

        all_clues = [tuple(clue)] + alternative_clues
        just_combos = [a for a, b in short_combos]

        # create meaning matrix
        meaning_matrix = list()
        for (c, i) in all_clues:
            arr = list()
            for jc in just_combos:
                p = np.prod([self.similarities[c][w] for w in jc])
                arr.append(p)
            arr = np.array(arr)
            try:
                for n in arr - meaning_matrix[0]:
                    if n > 0:
                        meaning_matrix.append(arr)
                        break
            except IndexError:
                meaning_matrix.append(arr)
            #meaning_matrix.append(arr)
        meaning_matrix = np.array(meaning_matrix)
        print("meaning matrix:\n", meaning_matrix)
        print(meaning_matrix.shape)
        literal_listener = normalize(meaning_matrix, norm='l1', axis=1)
        print("literal listener:\n", literal_listener)
        pragmatic_speaker = normalize(literal_listener, norm='l1', axis=0)
        print("pragmatic speaker:\n", pragmatic_speaker)
        pragmatic_listener = normalize(pragmatic_speaker, norm='l1', axis=1)
        print("pragmatic listener:\n", pragmatic_listener)
        relevant_row = pragmatic_listener[0]
        #print(relevant_row)
        ranking = sorted(list(zip(just_combos, relevant_row)), key=lambda x: x[1], reverse=True)
        return [i for i, n in ranking]

'''
class RSA_Pairs_kunar:
    def __init__(self, similarities_file, clue_file):
        with open(similarities_file, encoding="utf-8") as f:
            self.similarities = json.load(f)
        self.alt_clues = list()
        with open(clue_file, encoding="utf-8") as g:
            for line in g:
                lline = line.split()
                word = lline[0]
                if word in self.similarities:
                    self.alt_clues.append(word)
        with open("data/boards.json", encoding="utf-8") as h:
            data = json.load(h)
        self.partial_meaning_matrices = dict()
        for board in data:
            mm = list()
            for (c, i) in all_clues:
                arr = list()
                for comb in combinations(data[board], 2):
                    p = np.prod([self.similarities[c][w] for w in jc])
                    arr.append(p)
                arr = np.array(arr)
        print("init complete")

    def perform_RSA(self, board, clue):
        rank = list()
        for word in board:
            sim = self.similarities[clue[0]][word]
            rank.append((word, sim))
        #rank = sorted(rank, key=lambda x: x[1], reverse=True)
        #rank = rank[:20]
        combos = dict()
        for c in combinations(rank, clue[1]):
            words = tuple(i[0] for i in c)
            p = np.prod([i[1] for i in c])
            combos[words] = p
        #sorted_combos = sorted(combos.items(), key=lambda x: x[1], reverse=True)
        #short_combos = sorted_combos[:500]
        print(combos)

        # find alternative clues
        alternative_clues = [(i, clue[1]) for i in self.alt_clues if i != clue[0]]

        all_clues = [tuple(clue)] + alternative_clues
        just_combos = [a for a, b in short_combos]

        # create meaning matrix
        meaning_matrix = list()
        for (c, i) in all_clues:
            arr = list()
            for jc in just_combos:
                p = np.prod([self.similarities[c][w] for w in jc])
                arr.append(p)
            arr = np.array(arr)
            try:
                for n in arr - meaning_matrix[0]:
                    if n > 0:
                        meaning_matrix.append(arr)
                        break
            except IndexError:
                meaning_matrix.append(arr)
            #meaning_matrix.append(arr)
        meaning_matrix = np.array(meaning_matrix)
        print("meaning matrix:\n", meaning_matrix)
        print(meaning_matrix.shape)
        literal_listener = normalize(meaning_matrix, norm='l1', axis=1)
        print("literal listener:\n", literal_listener)
        pragmatic_speaker = normalize(literal_listener, norm='l1', axis=0)
        print("pragmatic speaker:\n", pragmatic_speaker)
        pragmatic_listener = normalize(pragmatic_speaker, norm='l1', axis=1)
        print("pragmatic listener:\n", pragmatic_listener)
        relevant_row = pragmatic_listener[0]
        #print(relevant_row)
        ranking = sorted(list(zip(just_combos, relevant_row)), key=lambda x: x[1], reverse=True)
        return [i for i, n in ranking]
'''


class RSA:
    def __init__(self, similarities_file):
        with open(similarities_file, encoding="utf-8") as f:
            self.similarities = json.load(f)

    def perform_RSA(self, board, clue):
        board_similarities = defaultdict(dict)
        meaning_matrix = list()
        for alt_clue in self.similarities:
            for card in board:
                sim = self.similarities[alt_clue][card]
                board_similarities[alt_clue][card] = sim

        first_row = [j for (i, j) in board_similarities[clue[0]].items()]
        meaning_matrix.append(np.array(first_row))
        for alt_clue in board_similarities:
            if alt_clue != clue:
                row = np.array([board_similarities[alt_clue][card] for card in board])
                for n in row - meaning_matrix[0]:
                    if n > 0:
                        meaning_matrix.append(row)
                        break
        meaning_matrix = np.array(meaning_matrix)
        print(meaning_matrix)
        print(meaning_matrix.shape)

        literal_listener = normalize(meaning_matrix, norm='l1', axis=1)
        pragmatic_speaker = normalize(literal_listener, norm='l1', axis=0)
        pragmatic_listener = normalize(pragmatic_speaker, norm='l1', axis=1)
        original_clue_scores = pragmatic_listener[0]

        pragmatic_board = list(zip(board, original_clue_scores))
        ranked_pragmatic_board = sorted(pragmatic_board, key=lambda x: x[1], reverse=True)
        return ranked_pragmatic_board


class RSA_Pairs_Gloss:
    def __init__(self, similarities_file):
        with open(similarities_file, encoding="utf-8") as f:
            self.similarities = json.load(f)
        print("init complete")

    def perform_RSA(self, board, clue):
        clue_meanings = list()
        board_meanings = list()
        for meaning in self.similarities:
            last_meaning = meaning
            if meaning.split("%")[0] == clue[0]:
                clue_meanings.append(meaning)
        #print(clue_meanings)
        for meaning2 in self.similarities[last_meaning]:
            if meaning2.split("%")[0] in board:
                board_meanings.append(meaning2)
        #print(board_meanings)

        # sort and shorten the board_meanings
        board_sims = sorted([(word, self.similarities[clue_meanings[0]][word]) for word in board_meanings], key=lambda x: x[1], reverse=True)
        short_board_meanings = [i for i, j in board_sims]
        #print(short_board_meanings)

        combos = list()
        for comb in combinations(short_board_meanings, clue[1]):
            # length of comb = clue[1]
            comb_set = set([c.split("%")[0] for c in comb])
            if len(list(comb_set)) == clue[1]:
                combos.append(comb)
        #print(len(combos))
        combos = combos[:500]
        print(len(combos))

        meaning_matrix = list()
        for clue_m in clue_meanings:
            row = np.array([np.prod([self.similarities[clue_m][n] for n in c]) for c in combos])
            #print(clue_m, row)
            meaning_matrix.append(row)

        clue_matrix = meaning_matrix

        for alt_clue in self.similarities:
            if alt_clue not in clue_meanings:
                row = np.array([np.prod([self.similarities[alt_clue][n] for n in c]) for c in combos])
                flag = False
                for i in range(len(clue_matrix)):
                    if flag:
                        break
                    for n in row - clue_matrix[i]:
                        if n > 0:
                            flag = True
                            break
                if flag:
                    #print(alt_clue)
                    meaning_matrix.append(row)

        meaning_matrix = np.array(meaning_matrix)

        #print("meaning matrix:\n", meaning_matrix)
        print(meaning_matrix.shape)
        literal_listener = normalize(meaning_matrix, norm='l1', axis=1)
        #print("literal listener:\n", literal_listener)
        pragmatic_speaker = normalize(literal_listener, norm='l1', axis=0)
        #print("pragmatic speaker:\n", pragmatic_speaker)
        pragmatic_listener = normalize(pragmatic_speaker, norm='l1', axis=1)
        #print("pragmatic listener:\n", pragmatic_listener)
        relevant_rows = pragmatic_listener[:len(clue_meanings)]
        #print(relevant_rows)
        ranking = list()
        for row in relevant_rows:
            zipped = list(zip(combos, row))
            ranking += zipped
        ranking = sorted(ranking, key=lambda x: x[1], reverse=True)
        print(ranking)
        ranking = [i for i, n in ranking]
        final_ranking = list()
        for j in ranking:
            jj = tuple(i.split("%")[0] for i in j)
            final_ranking.append(jj)
        return list(dict.fromkeys(final_ranking))


class RSA_Gloss:
    def __init__(self, similarities_file):
        with open(similarities_file, encoding="utf-8") as f:
            self.similarities = json.load(f)

    def perform_RSA(self, board, clue):
        board_similarities = defaultdict(dict)
        meaning_matrix = list()
        for alt_clue_sense in self.similarities:
            for card in board:
                #print(alt_clue_sense, card)
                for card_sense in self.similarities[alt_clue_sense]:
                    if card in card_sense:
                        #print(alt_clue_sense, card_sense, self.similarities[alt_clue_sense][card_sense])
                        sim = self.similarities[alt_clue_sense][card_sense]
                        board_similarities[alt_clue_sense][card_sense] = sim
        i = 0
        for clue_sense in board_similarities:
            if clue in clue_sense:
                random_clue_sense = clue_sense
                i += 1
                #print(clue_sense)
                row = [j for (i, j) in board_similarities[clue_sense].items()]
                meaning_matrix.append(np.array(row))
        for alt_clue_sense in board_similarities:
            if clue not in alt_clue_sense:
                row = [board_similarities[alt_clue_sense][card_sense] for card_sense in board_similarities[alt_clue_sense]]
                meaning_matrix.append(np.array(row))
        meaning_matrix = np.array(meaning_matrix)

        literal_listener = normalize(meaning_matrix, norm='l1', axis=1)
        #print("literal listener:\n", literal_listener)

        pragmatic_speaker = normalize(literal_listener, norm='l1', axis=0)
        #print("pragmatic speaker:\n", pragmatic_speaker)  # values get really small (use log?)

        pragmatic_listener = normalize(pragmatic_speaker, norm='l1', axis=1)
        #print("pragmatic listener:\n", pragmatic_listener)

        original_clue_scores = pragmatic_listener[:i]
        #print(self.original_clue_scores)
        #print(i)
        #print(len(self.original_clue_scores))

        all_boards = defaultdict(float)
        #print(self.board_similarities[self.random_clue_sense].keys())
        #print(self.board)
        for i in range(len(original_clue_scores)):
            pragmatic_board = list(zip(list(board_similarities[random_clue_sense].keys()), original_clue_scores[i]))
            #ranked_pragmatic_board = sorted(pragmatic_board, key=lambda x: x[1], reverse=True)
            #print(ranked_pragmatic_board)
            for guess, score in pragmatic_board:
                all_boards[guess] = all_boards[guess] + score
        #print(all_boards)
        return sorted(all_boards.items(), key=lambda x: x[1], reverse=True)


def produce_clue(red_words, bad_words, words, sims):
    R = defaultdict(dict)
    for r_w in red_words:
        for w in words:
            R[r_w][w] = sims[w][r_w]
    B = defaultdict(dict)
    for b_w in bad_words:
        for w in words:
            B[b_w][w] = sims[w][b_w]
    Ci = 0
    best = ""
    for i in range(1, len(red_words)+1):
        #d = float("inf")
        d = 0
        for rc in combinations(red_words, i):
            for w in words:
                if w not in red_words and w not in bad_words:
                    #wd = float("inf")
                    wd = 0
                    for b_w in bad_words:
                        if B[b_w][w] > wd:
                            wd = B[b_w][w]
                    #dr = 0
                    dr = float("inf")
                    for r_w in rc:
                        if R[r_w][w] < dr:
                            dr = R[r_w][w]
                    if dr > d and dr > wd: #and dr < 1: # threshold [0;1] to determine the models aggressiveness (the higher the more aggressive)
                        d = dr
                        best = w
                        Ci = i
    return best, Ci


def main():
    #rsa = RSA_Pairs("similarities/glove_similarities.json", "glove.42B.300d/glove_medium.txt")
    #rsa = RSA("similarities/glove_similarities.json")
    rsa = RSA_Pairs_Gloss("similarities/bert_similarities.json")
    #print(rsa.perform_RSA(["kangaroo", "key", "well"], ["australia", 1]))
    print(rsa.perform_RSA(["bridge", "button", "church"], ["architecture", 2]))


if __name__ == "__main__":
    main()
