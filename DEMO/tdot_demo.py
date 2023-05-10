from collections import defaultdict
from itertools import combinations
import numpy as np
from scipy import spatial
import random
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

class GloVe_Model:
    def __init__(self):
        self.embeddings = dict()
        with open("./glove.42B.300d/glove_short.txt", encoding="utf-8") as f:
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


def produce_clue(red_words, bad_words, words, model, threshold):
    """
    finds the best possible clue for the current game state from a set of possible clues
    :param red_words: list of words that the clue can refer to
    :param bad_words: list of words that the clue shouldn't refer to
    :param words: list of possible clues
    :param model: instance of GloVe class
    :return: tuple of clue word and number
    """
    R = defaultdict(dict)
    for r_w in red_words:
        for w in words:
            R[r_w][w] = model.distance(r_w, w)
    B = defaultdict(dict)
    for b_w in bad_words:
        for w in words:
            B[b_w][w] = model.distance(b_w, w)
    Ci = 0
    best = ""
    for i in range(1, len(red_words)+1):
        d = float("inf")
        for rc in combinations(red_words, i):
            for w in words:
                if w not in red_words and w not in bad_words:
                    wd = float("inf")
                    for b_w in bad_words:
                        if B[b_w][w] < wd:
                            wd = B[b_w][w]
                    dr = 0
                    for r_w in rc:
                        if R[r_w][w] > dr:
                            dr = R[r_w][w]
                    if dr < d and dr < wd and dr < threshold: # threshold [0;1] to determine the models aggressiveness (the higher the more aggressive)
                        d = dr
                        best = w
                        Ci = i
    return best, Ci

def spymaster(model, words):
    while True:
        try:
            clue = input("Pick a clue and a number (clue, n):\n")
            # clue = ("clue", 2)
            clue = clue.strip("(").strip(")").split(", ")
            guesses = model.get_hierarchy(clue[0], words)[:int(clue[1])]
            break
        except TypeError:
            print("\nUse the format: (clue, n)\n")
        except KeyError:
            print("\nIf the format was correct, the model might not know your clue.\n")
        except ValueError:
            print("\nUse the format: (clue, n)\n")
        except IndexError:
            print("\nUse the format: (clue, n)\n")
    # print(clue)
    guesses = [i for i,j in guesses]
    print("The operative thinks you are refering to:", guesses)
    return guesses


def operative(model, board, good_words, possible_clues, ps):
    bad_words = set(board) - set(good_words)

    # remove words that are too similar to board words from list of possible board words
    # stem both board words and possible clues then compare
    valid_clues = list()
    stemmed_board = [ps.stem(w) for w in board]
    for w, ws in zip(possible_clues.keys(), possible_clues.values()):
        if ws not in stemmed_board:
            valid_clues.append(w)

    inp = input("How ambitious do you want the spymaster to be? \n[n]ot so much, [s]omewhat, [v]ery\n")
    ambitiousness = {"n": 0.3, "s": 0.6, "v": 0.9}

    print("The spymaster is generating a clue...")
    clue = produce_clue(list(good_words), list(bad_words), valid_clues, model, ambitiousness[inp])
    print("The clue is:", clue)
    model_guess = model.get_hierarchy(clue[0], board)[:clue[1]]
    model_guess = [i for i,j in model_guess]
    inp = input("Make a guess and press enter if you want to see the results\n")
    print("The spymaster was refering to:", model_guess)
    return model_guess

def visualize_board(board_words, colors, markings=list()):
    # Define the colors
    # BLUE = (0, 0, 1)
    # RED = (1, 0, 0)
    # GRAY = (0.5, 0.5, 0.5)
    # BLACK = (0, 0, 0)
    # WHITE = (1, 1, 1)
    plt.close()
    plt.ion()
    # Define the game board dimensions
    BOARD_WIDTH = 5
    BOARD_HEIGHT = 5

    # Initialize the game board
    board = []
    for i in range(BOARD_HEIGHT):
        row = []
        for j in range(BOARD_WIDTH):
            index = i * BOARD_WIDTH + j
            word = board_words[index]
            color = colors[index]
            card = {"word": word, "color": color}
            row.append(card)
        board.append(row)

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 5))

    # Set the ticks and tick labels
    ax.set_xticks(np.arange(BOARD_WIDTH))
    ax.set_yticks(np.arange(BOARD_HEIGHT))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Create the colored boxes
    for i in range(BOARD_HEIGHT):
        for j in range(BOARD_WIDTH):
            card = board[i][j]
            color = card["color"]
            if card["word"] in markings:
                rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor=(0.3,0.8,0.5), lw=4)
            else:
                rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor="black")
            ax.add_patch(rect)

            # Add the word label
            word = card["word"]
            if color == "black" or color == "red" or color == "blue":
                text = ax.text(j + 0.5, i + 0.5, word, ha="center", va="center", color="white")
            else:
                text = ax.text(j + 0.5, i + 0.5, word, ha="center", va="center", color="black")

    # Set the plot limits and show the plot
    ax.set_xlim(0, BOARD_WIDTH)
    ax.set_ylim(0, BOARD_HEIGHT)

def main():
    print("Welcome to Codenames!\n")
    print("Initializing model...\n")
    model = GloVe_Model()
    ps = PorterStemmer()

    with open("glove_board_words.txt", encoding="utf-8") as f:
        possible_board_words = list()
        for line in f:
            possible_board_words.append(line.strip())

    with open("google-10000-english.txt", encoding="utf-8") as f:
        possible_clues = dict()
        for line in f:
            word = line.strip()
            if word in model.embeddings:
                possible_clues[word] = ps.stem(word)

    # start the game loop here
    while True:
        # create board
        print("Creating a game board...\n")
        board = random.sample(possible_board_words, 25)

        # assign random colors to words, restrictions: 9 blue, 8 red, 7 white, 1 black = 25 cards
        beginner = random.choice(["blue", "red"])
        if beginner == "red":
            colors = ["red"] * 9 + ["blue"] * 8 + ["white"] * 7 + ["black"]
        else:
            colors = ["blue"] * 9 + ["red"] * 8 + ["white"] * 7 + ["black"]
        random.shuffle(colors)
        red = list()
        blue = list()
        for w, c in zip(board, colors):
            if c == "red":
                red.append(w)
            elif c == "blue":
                blue.append(w)
        
        masked_colors = ["white"] * 25
        
        while True:
            print("You are on team", beginner, ".\n")
            inp = input("Do you want to be the [s]pymaster or the [o]perative?\n")

            if inp == "s":
                visualize_board(board, colors)
                guesses = spymaster(model, board)
                visualize_board(board, colors, guesses)
            elif inp == "o":
                visualize_board(board, masked_colors)
                if beginner == "red":
                    guesses = operative(model, board, red, possible_clues, ps)
                    visualize_board(board, colors, guesses)
                else:
                    guesses = operative(model, board, blue, possible_clues, ps)
                    visualize_board(board, colors, guesses)
            else:
                print("Error")
            
            print("Thank you for playing!\n")
            inp = input("Do you want to continue with the [s]ame board or a [d]ifferent one? (or [q]uit entirely)\n")
            if inp == "s":
                continue
            elif inp == "d" or inp == "q":
                break
        
        if inp == "d":
            continue
        elif inp == "q":
            break


if __name__ == "__main__":
    main()
