"""
CS23334 Fundamentals of Data Science Lab - Experiment 4
Count the frequency of occurrence of a word in a body of text during text processing.
"""

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import gutenberg


def main():
    nltk.download('gutenberg', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    sample = gutenberg.raw("austen-emma.txt")
    token = word_tokenize(sample)
    wlist = []

    for i in range(50):
        wlist.append(token[i])

    wordfreq = [wlist.count(w) for w in wlist]
    print("Pairs\n" + str(list(zip(wlist, wordfreq))))


if __name__ == "__main__":
    main()
