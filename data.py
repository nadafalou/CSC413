import csv
import numpy as np
import torch

tweets = []
Xtrain, Ttrain = [], []
Xvalid, Tvalid = [], []
Xtest, Ttest = [], []

def readData(path, X, T):
    """
    Reads the csv from path, and outputs the list of words by appending the tweet to the tweets list, as well as X,
    and appending the label (0 for real, 1 for fake) to T.
    path - String: a path to the csv file where the data is located.
    X - List[String]: The X list to store the tweets in.
    T - List[int]: The T list to store the labels in.
    """
    with open(path, newline='', encoding='utf-8') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='\"')
        spamreader.__next__()  # Skip header row
        for row in spamreader:
            words = row[1].split()
            t = 0 if row[2] == 'real' else 1
            sentence = [word.lower() for word in words]
            tweets.append(sentence)
            X.append(sentence)
            T.append(t)

readData("data/Constraint_Train.csv", Xtrain, Ttrain)
readData("data/Constraint_Val.csv", Xvalid, Tvalid)
readData("data/english_test_with_labels.csv", Xtest, Ttest)

# A list of all the words in the data set. We will assign a unique
# identifier for each of these words.
vocab = sorted(list(set([w for s in tweets for w in s])))
# A mapping of index => word (string)
vocab_itos = dict(enumerate(vocab))
# A mapping of word => its index
vocab_stoi = {word:index for index, word in vocab_itos.items()}

from copy import deepcopy

def convert_words_to_indices(sents):
    """
    This function takes a list of sentences (list of list of words)
    and returns a new list with the same structure, but where each word
    is replaced by its index in `vocab_stoi`.
    Example:
    >>> convert_words_to_indices([['one', 'in', 'five', 'are', 'over', 'here'],
    ['other', 'one', 'since', 'yesterday'],
    ['you']])
    [[148, 98, 70, 23, 154, 89], [151, 148, 181, 246], [248]]
    """
    sents = deepcopy(sents)
    for i in range(len(sents)):
        for j in range(len(sents[i])):
            sents[i][j] = vocab_stoi[sents[i][j].lower()]
        #sents[i] = torch.tensor(sents[i])
    return sents

# Convert words to indices in X data.
# Import these into the other modules, as well as vocab_itos and vocab_stoi as you see fit

Xtrain = convert_words_to_indices(Xtrain)
Ttrain = torch.tensor(Ttrain)
Xvalid = convert_words_to_indices(Xvalid)
Tvalid = torch.tensor(Tvalid)
Xtest = convert_words_to_indices(Xtest)
Ttest = torch.tensor(Ttest)

print(Xtrain[*[1,2,3]])
print(Ttrain[:10])
print(Xvalid[:10])
print(Tvalid[:10])
print(Xtest[:10])
print(Ttest[:10])
