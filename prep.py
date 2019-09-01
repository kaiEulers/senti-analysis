"""
This program performs lemmatisation and removes punctuations and stop words on the tweet text provided by Knowledge Technologies Project 2.
Two CSV files are processed in this program, namely train-tweets.csv and eval-tweets.csv

Created on Mon May 27 15:59:47 2019
@author: kaisoon
"""
#%%
import time
import os
import pandas as pd
from textPrep import textPrep
import string
import spacy
nlp = spacy.load('en_core_web_sm')

# --- Constants
startTime = time.time()
# Load stop words provided by Spacy
sWords = list(spacy.lang.en.stop_words.STOP_WORDS)
print(sWords)
# Load all punctuations
punc = string.punctuation

# --- Load data
data1 = pd.read_csv(
        "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/SentiAnalysis/data/train-tweets.csv"
        )
trainTweets = data1['tweet']

data2 = pd.read_csv(
        "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/SentiAnalysis/data/eval-tweets.csv"
        )
evalTweets = data2['tweet']


# --- Process tweets for training
trainTweets_prep_temp = textPrep(trainTweets)
trainTweets_prep = pd.concat(
        [data1['id'], trainTweets_prep_temp],
        axis=1
        )
trainTweets_prep.columns = ['id', 'tweet']

# --- Process tweets for evaluation
evalTweets_prep_temp = textPrep(evalTweets)
evalTweets_prep = pd.concat(
        [data2['id'], evalTweets_prep_temp],
        axis=1
        )
evalTweets_prep.columns = ['id', 'tweet']


# --- Save processed text for training
fileName_train = "train-tweets-prep-lemma-swNltk.csv"
trainTweets_prep.to_csv(
        "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/SentiAnalysis/data/" + fileName_train,
        index=False
        )
# --- Save processed text for evaluation
fileName_eval = "eval-tweets-prep-lemma-swNltk.csv"
evalTweets_prep.to_csv(
        "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/SentiAnalysis/data/" + fileName_eval,
        index=False
        )


print(f"Program took {(time.time()-startTime)/60}mins to run")
os.system('say "Complete"')


#%%
import time
import os
import pandas as pd
from textPrep import textPrep
import string
import spacy
nlp = spacy.load('en_core_web_sm')

# --- Constants
startTime = time.time()
# Load stop words provided by Spacy
sWords = list(spacy.lang.en.stop_words.STOP_WORDS)
print(sWords)
# Load all punctuations
punc = string.punctuation

# --- Load data
data = pd.read_csv(
        "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/SentiAnalysis/data/test-tweets.csv"
        )
testTweets = data['tweet']

# --- Process tweets for project test
testTweets_prep_temp = textPrep(testTweets)
testTweets_prep = pd.concat(
        [data['id'], testTweets_prep_temp],
        axis=1
        )
testTweets_prep.columns = ['id', 'tweet']

# --- Save processed text for training
fileName_test = "test-tweets-prep-lemma-swSpacy.csv"
testTweets_prep.to_csv(
        "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/SentiAnalysis/data/" + fileName_test,
        index=False
        )

os.system('say "Complete"')


