import random
from glove_model import GloVe_Model
from glossbert_model import GlossBERT_Model


def get_words():
    wordset = set()
    with open("words.txt", encoding="utf-8") as f:
        for line in f:
            word = line.strip().split()[0]
            wordset.add(word)
    return list(wordset)


def clean_words(words):
    cwords = list()
    glove = GloVe_Model()
    bert = GlossBERT_Model()
    for w in words:
        if w in glove.embeddings and w in bert.word_sense_emb:
            cwords.append(w)
    return cwords


def create_board(words, n):
    board = random.sample(words, n)
    return board


def create_colors(n):
    # red and blue: between 2 and 8/9
    # white: between 3 and 7
    # black: 1
    teams = ["red", "blue"]
    colors = teams + ["white", "black"]
    black = 1
    if n < 12:
        white = 4
        red = random.randint(2, n-white-black-2)
        blue = n - white - black - red
    elif n < 15:
        white = 5
        red = random.randint(3, n-white-black-3)
        blue = n - white - black - red
    elif n < 19:
        white = 6
        red = random.randint(4, n-white-black-4)
        blue = n - white - black - red
    elif n < 22:
        white = 7
        red = random.randint(5, n-white-black-5)
        blue = n - white - black - red
    elif n < 25:
        white = 7
        red = random.randint(7, n-white-black-7)
        blue = n - white - black - red
    else:
        white = 7
        red = random.randint(8, 9)
        blue = n - white - black - red
    board_colors = red * [colors[0]] + blue * [colors[1]] + white * [colors[2]] + black * [colors[3]]
    random.shuffle(board_colors)
    return board_colors


def create_random_game_state(cwords):
    board_size = random.randint(9, 25)
    board = create_board(cwords, board_size)
    colors = create_colors(board_size)
    return board, colors


def visualize_board(board, colors):
    visual = list()
    for b, c in zip(board, colors):
        if c == "white":
            visual.append(b)
        if c == "black":
            visual.append('\033[32m' + b + '\033[0m')
        if c == "blue":
            visual.append('\033[34m' + b + '\033[0m')
        if c == "red":
            visual.append('\033[31m' + b + '\033[0m')
    print(" ".join(visual))


def main():
    words = get_words()
    cwords = clean_words(words)
    # write in file
    with open("eval_data.csv", "w", encoding="utf-8") as f:
        for i in range(1, 81):
            board, colors = create_random_game_state(cwords)
            visualize_board(board, colors)
            line = str(i) + "," + " ".join(board) + "," + " ".join(colors) + "\n"
            f.writelines(line)


if __name__ == "__main__":
    main()
