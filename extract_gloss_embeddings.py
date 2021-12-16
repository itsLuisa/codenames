from collections import defaultdict
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertModel


def get_id_sense(keys):
    id_sense = dict()
    with open(keys, encoding="utf-8") as f:
        for line in f:
            lline = line.split()
            id_sense[lline[0]] = lline[1:]
    return id_sense


def get_sense_emb(data, tz, model, id_sense):
    with open(data, encoding="utf-8") as f:
        file = f.read()
    sense_emb = defaultdict(list)
    bs_data = BeautifulSoup(file, "lxml")
    sentences = bs_data.find_all("sentence")
    for sen in sentences:
        lsen = sen.contents
        lsen = [i for i in lsen if i != "\n"]
        id_pos = list()
        for word in lsen:
            lword = str(word).split()
            if "<instance" in lword:
                id_pos.append((lword[1], lsen.index(word)))
        csen = [i.contents[0] for i in lsen]
        # tokenize and find embeddings
        ids = tz.convert_tokens_to_ids(csen)
        enc = tz.encode(ids, return_tensors="pt")
        emb = model(enc)
        last_hidden_state = emb.last_hidden_state
        for id, pos in id_pos:
            id = id.split('"')[1]
            for n in range(len(id_sense[id])):
                sense_emb[id_sense[id][n]].append(last_hidden_state[0][pos+1].detach().numpy())
    sense_emb = {sense: sum(emb) / len(emb) for (sense, emb) in sense_emb.items()}
    return sense_emb


def group_word_senses(keys, sense_emb):
    with open(keys, encoding="utf-8") as f:
        word_sense_emb = defaultdict(dict)
        for line in f:
            lline = line.split()
            id = lline[0]
            senses = lline[1:]
            for sense in senses:
                word = sense.split("%")[0]
                word_sense_emb[word][sense] = list(sense_emb[sense])
    return word_sense_emb


def main():
    data = "GlossBERT/Training_Corpora/SemCor/semcor.data.xml"
    keys = "GlossBERT/Training_Corpora/SemCor/semcor.gold.key.txt"

    tz = BertTokenizer.from_pretrained("Sent_CLS_WS")
    model = BertModel.from_pretrained("Sent_CLS_WS")

    id_sense = get_id_sense(keys)
    sense_emb = get_sense_emb(data, tz, model, id_sense)
    word_sense_emb = group_word_senses(keys, sense_emb)
    print(word_sense_emb)

    with open("GlossBERTembeddings/GlossBertEmbeddings.txt", "w", encoding="utf-8") as f:
        for word in word_sense_emb:
            line = str(word) + "\n"
            f.writelines(line)
            for sense in word_sense_emb[word]:
                line = str(sense) + "\t" + str(word_sense_emb[word][sense]) + "\n"
                f.writelines(line)


if __name__ == "__main__":
    main()
