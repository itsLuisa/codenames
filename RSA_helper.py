from glove_model import GloVe_Model
from glossbert_model import GlossBERT_Model
import json
from collections import defaultdict

glove = GloVe_Model()
bert = GlossBERT_Model()

# maybe add a helper file that includes the google-10000.txt

glove_board_words = list()
with open("similarities/glove_board_words.txt", encoding="utf-8") as f:
    for line in f:
        word = line.strip()
        glove_board_words.append(word)

glove_sims = defaultdict(dict)
with open("similarities/glove_similarities.json", "w", encoding="utf-8") as g:
    for entry in glove.embeddings:
        for word in glove_board_words:
            print(entry, word)
            sim = glove.similarity(entry, word)
            glove_sims[entry][word] = sim
    json.dump(glove_sims, g)
'''
gloss_board_words = list()
with open("similarities/gloss_board_words.txt", encoding="utf-8") as f:
    for line in f:
        word = line.strip()
        gloss_board_words.append(word)

bert_sims = defaultdict(dict)
with open("similarities/bert_similarities.json", "w", encoding="utf-8") as h:
    for entry in bert.word_sense_emb:
        for sense_entry in bert.word_sense_emb[entry]:
            for word in gloss_board_words:
                for sense_word in bert.word_sense_emb[word]:
                    sim = bert.similarity(entry, word, sense_entry, sense_word)
                    bert_sims[sense_entry][sense_word] = sim
    json.dump(bert_sims, h)
'''