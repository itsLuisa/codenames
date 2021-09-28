from nltk.corpus import wordnet
from collections import defaultdict
from transformers import BertTokenizer, BertModel
import torch
from scipy import spatial


def find_word_definition(word, d):
    synset = wordnet.synsets(word)
    for s in range(len(synset)):
        d[word].append(word + " " + synset[s].definition())
    return d


def get_bert_embeddings(tz, model, sentence):
    inputs = tz(sentence, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    first_word = last_hidden_states[0][0]
    return first_word


def collect_embeddings(tz, model, defs):
    embs = dict()
    for word in defs:
        embs[word] = dict()
        for concept in defs[word]:
            emb_concept = get_bert_embeddings(tz, model, concept)
            embs[word][concept] = emb_concept
    return embs


def cosine_distance(emb1, emb2):
    return spatial.distance.cosine(emb1, emb2)


def get_distances(clue_embeddings, board_embeddings):
    distances = list()
    for word in clue_embeddings:
        for sense in clue_embeddings[word]:
            for word2 in board_embeddings:
                for sense2 in board_embeddings[word2]:
                    d = cosine_distance(clue_embeddings[word][sense].detach().numpy(),
                                        board_embeddings[word2][sense2].detach().numpy())
                    distances.append((sense, sense2, d))
    return sorted(distances, key=lambda x: x[2])


def bert_based_guess(board, clue):
    clue_defs = defaultdict(list)
    clue_defs = find_word_definition(clue[0], clue_defs)
    print(clue_defs)

    board_defs = defaultdict(list)
    for word in board:
        board_defs = find_word_definition(word, board_defs)
    print(board_defs)

    tz = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    clue_embeddings = collect_embeddings(tz, model, clue_defs)
    board_embeddings = collect_embeddings(tz, model, board_defs)

    sorted_distances = get_distances(clue_embeddings, board_embeddings)
    guesses = [j.split()[0] for i,j,k in sorted_distances]
    no_duplicates = list()
    for word in guesses:
        if word not in no_duplicates:
            no_duplicates.append(word)
    final_guesses = no_duplicates[:clue[1]]
    return final_guesses

def main():
    """synset = wordnet.synsets("Travel")
    print('Word and Type : ' + synset[0].name())
    print('Synonym of Travel is: ' + synset[0].lemmas()[0].name())
    print('The meaning of the word : ' + synset[0].definition())
    print('Example of Travel : ' + str(synset[0].examples()))"""

    board = ["saw", "page"]
    clue = ("party", 2)

    guesses = bert_based_guess(board, clue)
    print(guesses)


if __name__ == "__main__":
    main()
