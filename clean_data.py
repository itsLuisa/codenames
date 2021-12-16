import json
from semantic_models import GloVe_Model
from extract_meaning_embeddings import GlossBERT_Model

glove = GloVe_Model()
gloss = GlossBERT_Model()

with open("data/eval_data_clean.json", encoding="utf-8") as f:
    data = json.load(f)

clean_data = list()
for game in data:
    for turn in data[game]:
        if turn != "color distribution":
            remaining_words = data[game][turn]["remaining words"]
            clue = data[game][turn]["clue"]
            flag = clue[0] in glove.embeddings
            if flag:
                for r in remaining_words:
                    if flag:
                        if r not in glove.embeddings:
                            flag = False
            if flag:
                clean_data.append(data[game][turn])

print(clean_data)
print(len(clean_data))
clean_data_dict = dict()
for n, scenario in enumerate(clean_data):
    clean_data_dict[n] = scenario

print(clean_data_dict)
with open("data/eval_clean_glove.json", "w", encoding="utf-8") as g:
    json.dump(clean_data_dict, g)


clean_data = list()
for game in data:
    for turn in data[game]:
        if turn != "color distribution":
            print("new turn")
            remaining_words = data[game][turn]["remaining words"]
            clue = data[game][turn]["clue"]
            flag = clue[0] in gloss.word_sense_emb
            print(clue[0], flag)
            if flag:
                for r in remaining_words:
                    #if flag:
                    if r not in gloss.word_sense_emb:
                        print(r)
                        flag = False
            if flag:
                clean_data.append(data[game][turn])

print(clean_data)
print(len(clean_data))
clean_data_dict = dict()
for n, scenario in enumerate(clean_data):
    clean_data_dict[n] = scenario

print(clean_data_dict)
with open("data/eval_clean_gloss.json", "w", encoding="utf-8") as g:
    json.dump(clean_data_dict, g)
