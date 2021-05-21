# imports
import numpy as np
import pandas as pd
from scipy import spatial
import json
import gensim.downloader as api
from collections import defaultdict

class GloVe_Model:
    def __init__(self):
        self.embeddings = dict()
        with open("./glove.42B.300d/glove_short.txt", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                self.embeddings[word] = vector

    def distance(self, word, reference):
        return spatial.distance.cosine(self.embeddings[word], self.embeddings[reference])

    def closest_words(self, reference):
        return sorted(self.embeddings.keys(), key=lambda w: distance(w, reference))

    def get_hierarchy(self, clue, wordlist):
        d = dict()
        for word in wordlist:
            d[word] = self.distance(clue, word)
        return sorted(d.items(), key=lambda x: x[1])

class Word2Vec_Model:
    def __init__(self):
        self.model = api.load('word2vec-google-news-300')

    def similarity(self, word, reference):
        return self.model.similarity(word, reference)

    def get_hierarchy(self, clue, wordlist):
        d = dict()
        for word in wordlist:
            d[word] = self.similarity(clue, word)
        return sorted(d.items(), key=lambda x: x[1], reverse=True)

def get_all_cards(data, game):
    return data[game]["color distribution"]

def get_current_cards(data, game, round):
    return data[game][round]["remaining words"]

def get_all_color_cards(data, game, color):
    l = list()
    for card in data[game]["color distribution"]:
        if data[game]["color distribution"][card] == color:
            l.append(card)
    return l

def get_clue(data, game, round):
    return data[game][round]["clue"]

def get_guess(data, game, round):
    return data[game][round]["guesses"]


def main():
    glove = GloVe_Model()
    word2vec = Word2Vec_Model()

    with open("codenamesexp.json", encoding="utf-8") as f:
        data = json.load(f)

    # guesser model
    for game in data:
        all_cards = get_all_cards(data, game)
        red_cards = get_all_color_cards(data, game, "red")
        blue_cards = get_all_color_cards(data, game, "blue")
        for round in data[game]:
            d_avg_scores = defaultdict(list)
            if round != "color distribution":
                current_cards = get_current_cards(data, game, round)
                clue = get_clue(data, game, round)
                print(clue)
                if clue[0] in glove.embeddings:
                    glove_guesses = glove.get_hierarchy(clue[0], current_cards)
                    word2vec_guesses = word2vec.get_hierarchy(clue[0], current_cards)
                    human_guesses = get_guess(data, game, round)
                    print(human_guesses)
                    glove_score = list()
                    word2vec_score = list()
                    for pos, pair in enumerate(glove_guesses):
                        if pair[0] in human_guesses:
                            glove_score.append(pos)
                    for pos, pair in enumerate(word2vec_guesses):
                        if pair[0] in human_guesses:
                            word2vec_score.append(pos)
                    avg_glove = np.mean(glove_score)
                    avg_word2vec = np.mean(word2vec_score)
                    print("glove:", glove_score, avg_glove)
                    print("word2vec:", word2vec_score, avg_word2vec)
                    d_avg_scores["glove"].append(avg_glove)
                    d_avg_scores["word2vec"].append(avg_word2vec)
            print("glove:", np.mean(d_avg_scores["glove"]))
            print("word2vec:", np.mean(d_avg_scores["word2vec"]))
        break


if __name__ == "__main__":
    main()