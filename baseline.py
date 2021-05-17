# imports
import numpy as np
import pandas as pd
from scipy import spatial
import json
import gensim.downloader as api

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

def Word2Vec_Model():
    return api.load('word2vec-google-news-300')


def main():
    glove = GloVe_Model()
    word2vec = Word2Vec_Model()

    with open("codenamesexp.json", encoding="utf-8") as f:
        data = json.load(f)

    for game in data:
        print(data[game])
        for round in data[game]:
            if round != "color distribution":
                print(round)
                clue = data[game][round]["clue"]
                referenced_words = data[game][round]["referenced words"]
                if clue[0] in glove.embeddings:
                    for word in referenced_words:
                        if word in glove.embeddings:
                            print("GloVe:", word, clue[0], glove.distance(word, clue[0]))
                            print("Word2Vec:", word, clue[0], word2vec.similarity(word, clue[0]))

if __name__ == "__main__":
    main()