import numpy as np
from scipy import spatial
#import gensim.downloader as api


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
        return sorted(self.embeddings.keys(), key=lambda w: self.distance(w, reference))

    def get_hierarchy(self, clue, wordlist):
        d = dict()
        for word in wordlist:
            d[word] = self.distance(clue, word)
        return sorted(d.items(), key=lambda x: x[1])

    def goodness(self, word, answers, bad):
        if word in answers + bad: return -999
        return sum([self.distance(word, b) for b in bad]) - 4.0 * sum([self.distance(word, a) for a in answers])

    def minimax(self, word, answers, bad):
        if word in answers + bad: return -999
        return min([self.distance(word, b) for b in bad]) - max([self.distance(word, a) for a in answers])

    def candidates(self, answers, bad, size=10):
        best = sorted(self.embeddings.keys(), key=lambda w: -1 * self.goodness(w, answers, bad))
        #res = [(str(i + 1), "{0:.2f}".format(self.minimax(w, answers, bad)), w) for i, w in enumerate(sorted(best[:250], key=lambda w: -1 * self.minimax(w, answers, bad))[:size])]
        res = [(w, self.minimax(w, answers, bad)) for w in sorted(best[:250], key=lambda w: -1 * self.minimax(w, answers, bad))[:size]]
        #return [(". ".join([c[0], c[2]]) + " (" + c[1] + ")") for c in res]
        return res


def main():
    glove = GloVe_Model()
    print(glove.distance("himalayas", "aztec"))
    print(glove.closest_words("well")[:10])
    #answers = ["iron", "ham", "beijing"]
    answers = ["manicure", "stick"]
    bad = ["alaska", "cast", "tap", "leaf", "bermuda", "dentist", "spider", "wool", "ram", "racket", "iron", "rope", "comic", "bugle", "ranch", "block", "dress", "stadium", "plate", "vampire", "poison", "van", "bill"]
    print(glove.candidates(answers, bad))


if __name__ == "__main__":
    main()
