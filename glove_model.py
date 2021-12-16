import numpy as np
from scipy import spatial


class GloVe_Model:
    def __init__(self):
        self.embeddings = dict()
        with open("./glove.42B.300d/glove_medium.txt", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                self.embeddings[word] = vector
        self.google_words = list()

    def distance(self, word, reference):
        """
        computes cosine distance between two words
        :param word: string of first word
        :param reference: string of second word
        :return: float of cosine distance within [0;2]
        """
        return spatial.distance.cosine(self.embeddings[word], self.embeddings[reference]) # change it to similarity

    def similarity(self, word, reference):
        return 1 - (spatial.distance.cosine(self.embeddings[word], self.embeddings[reference]) / 2)

    def closest_words(self, reference):
        """
        gives closest words to reference word
        :param reference: string of reference word
        :return: list of close words in descending order
        """
        return sorted(self.embeddings.keys(), key=lambda w: self.distance(w, reference))

    def get_hierarchy(self, clue, wordlist):
        """
        computes hierarchical list of words according to their distance to the clue word
        :param clue: string of clue word
        :param wordlist: list of board words
        :return: sorted list of board words according to distance in ascending order
        """
        d = dict()
        for word in wordlist:
            d[word] = self.similarity(clue, word)
        return sorted(d.items(), key=lambda x: x[1], reverse=True)


def main():
    glove = GloVe_Model()
    print(glove.distance("himalayas", "aztec"))
    print(glove.closest_words("well")[:10])
    answers = ["manicure", "stick"]
    bad = ["alaska", "cast", "tap", "leaf", "bermuda", "dentist", "spider", "wool", "ram", "racket", "iron", "rope", "comic", "bugle", "ranch", "block", "dress", "stadium", "plate", "vampire", "poison", "van", "bill"]
    print(len(glove.embeddings))
    print("australia", "well", glove.similarity("australia", "well"))
    print("australia", "kangaroo", glove.similarity("australia", "kangaroo"))
    print("good", "well", glove.similarity("good", "well"))
    print("good", "kangaroo", glove.similarity("good", "kangaroo"))


if __name__ == "__main__":
    main()
