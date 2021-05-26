import numpy as np
from scipy import spatial
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

def main():
    glove = GloVe_Model()
    print(glove.distance("house", "stone"))
    word2vec = Word2Vec_Model()
    print(word2vec.similarity("house", "stone"))

if __name__ == "__main__":
    main()