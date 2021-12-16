from collections import defaultdict
from scipy import spatial


class GlossBERT_Model:
    def __init__(self):
        self.file = "./GlossBERTembeddings/GlossBertEmbeddings.txt"
        self.word_sense_emb = defaultdict(dict)
        with open(self.file, encoding="utf-8") as f:
            for line in f:
                lline = line.strip().split("\t")
                if len(lline) == 1:
                    word = lline[0]
                else:
                    sense = lline[0]
                    emb = lline[1]
                    lemb = emb.strip("[]").split(", ")
                    lemb = [float(i) for i in lemb]
                    self.word_sense_emb[word][sense] = lemb

    def similarity(self, word1, word2, sense1, sense2):
        return 1 - (spatial.distance.cosine(self.word_sense_emb[word1][sense1], self.word_sense_emb[word2][sense2]) / 2)

    def get_hierarchy(self, clue, wordlist):
        model_guesses = list()
        for sense1 in self.word_sense_emb[clue]:
            clue_emb = self.word_sense_emb[clue][sense1]
            for word in wordlist:
                for sense in self.word_sense_emb[word]:
                    sim = 1 - (spatial.distance.cosine(clue_emb, self.word_sense_emb[word][sense]) / 2)
                    model_guesses.append((sense1, sense, sim))
        model_guesses = sorted(model_guesses, key=lambda x: x[2], reverse=True)
        return model_guesses


def main():
    file = "GlossBERTembeddings/GlossBertEmbeddings.txt"
    model = GlossBERT_Model()
    evalfile = "data/codenamesexp.json"


if __name__ == "__main__":
    main()
