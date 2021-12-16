from nltk.corpus import wordnet
from collections import defaultdict
from transformers import BertTokenizer, BertModel
#import torch
from scipy import spatial


def find_word_definition(word, d):
    synset = wordnet.synsets(word)
    for s in range(len(synset)):
        d[word].append(word + " " + synset[s].definition())
    return d


def find_example_sentences(word, d):
    synset = wordnet.synsets(word)
    for s in range(len(synset)):
        try:
            d[word].append(synset[s].examples()[0])
        except IndexError:
            d[word].append(synset[s].definition() + " " + word)
    return d


def get_synonyms(word):
    synonyms = list()
    synonyms.append(word)
    synset = wordnet.synsets(word)
    for s in range(len(synset)):
        for lemma in synset[s].lemmas():
            synonyms.append(lemma.name())
    synonyms.reverse()
    return synonyms


def get_bert_embeddings(tz, model, sentence, index):
    inputs = tz(sentence, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    first_word = last_hidden_states[0][index]
    return first_word


def collect_embeddings(tz, model, defs):
    # find out which position in the sentence is the relevant one
    embs = dict()
    indices = list()
    for word in defs:
        embs[word] = dict()
        synonyms = get_synonyms(word)
        for concept in defs[word]:
            index = False
            #print(concept)
            sentence_tz = tz.tokenize(concept)
            #print(sentence_tz)
            for syn in synonyms:
                #print(syn)
                if syn in sentence_tz:
                    index = sentence_tz.index(syn)
            if index:
                #print(index)
                #print(sentence_tz[index])
                emb_concept = get_bert_embeddings(tz, model, concept, index)
                #print(emb_concept)
                embs[word][concept] = emb_concept
                indices.append(index)
    return embs


def cosine_similarity(emb1, emb2):
    return 1 - (spatial.distance.cosine(emb1, emb2) / 2)


def get_similarities(clue_embeddings, board_embeddings):
    sims = list()
    for word in clue_embeddings:
        #print(word)
        for sense in clue_embeddings[word]:
            for word2 in board_embeddings:
                for sense2 in board_embeddings[word2]:
                    d = cosine_similarity(clue_embeddings[word][sense].detach().numpy(),
                                        board_embeddings[word2][sense2].detach().numpy())
                    sims.append((word, word2, d))
    return sorted(sims, key=lambda x: x[2], reverse=True)


def bert_based_guess(board, clue):
    clue_defs = defaultdict(list)
    clue_defs = find_word_definition(clue[0], clue_defs)
    #print(clue_defs)
    clue_exs = defaultdict(list)
    clue_exs = find_example_sentences(clue[0], clue_exs)
    #print(clue_exs)

    board_defs = defaultdict(list)
    for word in board:
        board_defs = find_word_definition(word, board_defs)
    #print(board_defs)

    board_exs = defaultdict(list)
    for word in board:
        board_exs = find_example_sentences(word, board_exs)
    #print(board_exs)

    tz = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    #model = BertModel.from_pretrained("Sent_CLS_WS")

    clue_embeddings = collect_embeddings(tz, model, clue_exs)
    board_embeddings = collect_embeddings(tz, model, board_exs)
    #print(clue_embeddings)
    #print(board_embeddings)

    sorted_sims = get_similarities(clue_embeddings, board_embeddings)
    print(sorted_sims)
    guesses = [j for i,j,k in sorted_sims]
    no_duplicates = list()
    for word in guesses:
        if word not in no_duplicates:
            no_duplicates.append(word)
    final_guesses = no_duplicates
    return final_guesses


def main():
    '''synset = wordnet.synsets("Travel")
    print('Word and Type : ' + synset[0].name())
    print('Synonym of Travel is: ' + synset[0].lemmas()[0].name())
    print('The meaning of the word : ' + synset[0].definition())
    print('Example of Travel : ' + str(synset[0].examples()))'''

    board = ["saw", "page"]
    clue = ("party", 2)
    #print(get_synonyms("saw"))

    guesses = bert_based_guess(board, clue)
    print(guesses)


if __name__ == "__main__":
    main()
