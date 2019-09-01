#%% Functions
import os
import pickle
import csv
import spacy
nlp = spacy.load("en_core_web_sm")


# ========================================
def wordCount(wordList, doc):
    """
    wordCount(wordList, doc) counts each word in the wordList in a tweet to create a list of wordCounts.
    Returns a list of wordCounts.
    """
    wCntList = []
    for word in wordList:
        wCnt = 0
        for i in range(len(doc)):
            if doc[i].text.lower() == word and doc[i-1].dep_ != "neg":
                wCnt += 1
        wCntList.append(wCnt)

    return wCntList


# ========================================
def evalRM(wordList, wCntMatrix):
    """
    Create new file evalRM.csv base on eval.scv with added new words and their wordCounts
    """
#    Read from eval.csv and write to evalRM.csv
    pathEval = "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/MachLearning/eval.csv"
    pathEvalRM = "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/MachLearning/evalRM.csv"
    writeRM(wordList, wCntMatrix, pathEval, pathEvalRM)


# ========================================
def trainRM(wordList, wCntMatrix):
    """
    Create new file trainRM.csv base on train.scv with added new words and their wordCounts
    """
#    Read from train.csv and write to trainRM.csv
    pathTrain = "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/MachLearning/train.csv"
    pathTrainRM = "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/MachLearning/trainRM.csv"
    writeRM(wordList, wCntMatrix, pathTrain, pathTrainRM)


# ========================================
def writeRM(wordList, wCntMatrix, path, pathRM):
    """
    writeRM() appends a list of words and their respective wordCounts onto the specified pathRM. The original file's path is use to extract data from the original file such that new data can be appended and written to pathRM
    """
#    Read from path
    fread = open(path, newline = "")
    reader = csv.reader(fread)

#    Extract header from eval.csv
    header = next(reader)
#    Extract data from eval.csv
    data = [row for row in reader]
#    Remove "# " from tweetID
    for row in data:
        row[0] = row[0][2:]

#    Append new words to the header
    headerRM = header + wordList
#    Append new wordCounts to data
    dataRM = []
    for i in range(len(wCntMatrix)):
        dataRM.append(data[i] + wCntMatrix[i])

#    Write to pathRM
    fwrite = open(pathRM, "w")
    writer = csv.writer(fwrite)
    writer.writerow(headerRM)
    for d in dataRM:
        writer.writerow(d)


#%% Load spacy doc files
docListEval = pickle.load(open("docListEval.dat", "rb"))
docListTrain = pickle.load(open("docListTrain.dat", "rb"))


#%% Open tweet files and tokenise each line of tweet in a spacy doc
# THIS CELL TAKES A LONG TIME TO RUN!!!

pathEval = "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/MachLearning/eval-tweets.txt"
file = open(pathEval)
data = [line.strip().split("\t") for line in file]
tweet = [row[1] for row in data]
file.close()

# Tokenise tweets in eval-tweets.txt using spacy
docListEval = []
for t in tweet:
    docListEval.append(nlp(t))


pathTrain = "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/MachLearning/train-tweets.txt"
file = open(pathTrain)
data = [line.strip().split("\t") for line in file]
tweet = [row[1] for row in data]
file.close()

# Tokenise tweets in train-tweets.txt using spacy
docListTrain = []
for t in tweet:
    docListTrain.append(nlp(t))

# Save files
path_docListEval = "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/MachLearning/docListEval.dat"
pickle.dump(docListEval, open(path_docListEval, "wb"))

path_docListTrain = "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/MachLearning/docListTrain.dat"
pickle.dump(docListTrain, open(path_docListTrain, "wb"))

os.system('say "Complete"')


#%% Create CSV files with new words and their word counts for RapidMiner

posWord = [
        "perfect",
        "excellent",
        "best",
        "better",
        "good",
        "excited",
        "happy",
        "thank",
        "screamqueens",
        "thanksgiving",
        "dog",
        "young",
        "voter",
        "ready",
        "cream"
        ]
negWord = [
        "awful",
        "terrible",
        "weak",
        "sad",
        "gun",
        "anti"
        ]
wordList = posWord + negWord
wCntMatEval = [wordCount(wordList, doc) for doc in docListEval]
wCntMatTrain = [wordCount(wordList, doc) for doc in docListTrain]

# ============================== For Debugging
# Count total number of words in all tweets
L = 15
print("------------------------------")
print("eval.csv")
print("------------------------------")
evalTest = list(map(list, zip(*wCntMatEval)))
for i in range(len(evalTest)):
#    print(wordList[i], "\t\t", sum(evalTest[i]))
    print(f"{wordList[i]:{L}} {sum(evalTest[i])}")
print("\n")

print("------------------------------")
print("train.csv")
print("------------------------------")
trainTest = list(map(list, zip(*wCntMatTrain)))
for i in range(len(trainTest)):
    print(f"{wordList[i]:{L}} {sum(trainTest[i])}")
print("\n")


# Create CSV files for RapidMiner
evalRM(wordList, wCntMatEval)
trainRM(wordList, wCntMatTrain)


#%% Negation
w = "I am thankful"

doc1 = nlp(w)
for token in doc1:
    print(token.text, token.dep_)


#%% Tests
import numpy as np

# pos, neg, neu
nb = np.array([
        [1335, 521, 1314],
        [147, 577, 480],
        [906, 732, 2362]
        ])
nb-tfidf = np.array([
        [],
        [],
        []
        ])
randF = np.array([
        [],
        [],
        []
        ])
randF-tfidf = np.array([
        [],
        [],
        []
        ])
