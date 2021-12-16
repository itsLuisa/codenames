import json
from glossbert_model import GlossBERT_Model
from glove_model import GloVe_Model

model1 = GloVe_Model()
model2 = GlossBERT_Model()

word_list = list()
with open("possible_board_words.txt", encoding="utf-8") as f:
    for line in f:
        word_list.append(line.strip())

word_set = set(word_list)
print(word_set)
print(len(list(word_set)))

with open("../data/codenamesexp.json", encoding="utf-8") as f:
    data = json.load(f)
for game in data:
    words = data[game]["color distribution"].keys()
    for word in words:
        word_set.add(word)

with open("../data/eval_data_clean.json", encoding="utf-8") as f:
    data = json.load(f)
for game in data:
    words = data[game]["color distribution"].keys()
    for word in words:
        word_set.add(word)

general_words = set()
glove_words = set()
gloss_words = set()

for word in list(word_set):
    if word in model1.embeddings and word in model2.word_sense_emb:
        general_words.add(word)
    if word in model1.embeddings:
        glove_words.add(word)
    if word in model2.word_sense_emb:
        gloss_words.add(word)

print(len(list(general_words)))

print(len(list(glove_words)))

print(len(list(gloss_words)))

general_word_list = sorted(list(general_words))
with open("general_board_words.txt", "w", encoding="utf-8") as f:
    for word in general_word_list:
        line = word + "\n"
        f.writelines(line)

glove_word_list = sorted(list(glove_words))
with open("glove_board_words.txt", "w", encoding="utf-8") as f:
    for word in glove_word_list:
        line = word + "\n"
        f.writelines(line)

gloss_word_list = sorted(list(gloss_words))
with open("gloss_board_words.txt", "w", encoding="utf-8") as f:
    for word in gloss_word_list:
        line = word + "\n"
        f.writelines(line)
