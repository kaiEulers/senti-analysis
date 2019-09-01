"""
This program computes the precision, recall, and f1-score of Machine Learning algorithms used to predict the sentiment of a collection of tweets.
A comparison is made between not removing stop words from the vector and removing stop words from the vector.

Created on Sun May 26 16:28:11 2019
@author: kaisoon
"""
#%% Import Libraries & Declare Constants
# --- Constants
resultLabel = ['Not preprocessed', 'Preprocessed']

# --- Functions
from func import *

# --- Libraries
# Feature Extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Pipeline
from sklearn.pipeline import Pipeline


#%% Using Naive Bayes
# Unprocessed and processed by transforming to lower case VS removing punctuation
from sklearn.naive_bayes import MultinomialNB

# --- Without Preprocessing
# Select Dataset
featTrain, featTest = getTweets()
labelTrain, labelTest = getLabels()

pipe = Pipeline([("tfid", TfidfVectorizer()), ("nb", MultinomialNB())])
pipe.fit(featTrain, labelTrain)
pred = pipe.predict(featTest)

resultBefore = getResult(labelTest, pred)

# --- With Preprocessing
# Select Dataset
featTrain, featTest = getTweets(suffix='-prep')

pipe = Pipeline([("tfid", TfidfVectorizer()), ("nb", MultinomialNB())])
pipe.fit(featTrain, labelTrain)
pred = pipe.predict(featTest)

resultAfter = getResult(labelTest, pred)

# Plot results
f1Plot(resultBefore, resultAfter, resultLabel)

# Compile Results
compileResult(resultBefore, resultAfter, 'Naive Bayes')


#%% Using Naive Bayes
# Processed by transforming to lower case and removing punctuation VS Lemmatisation
from sklearn.naive_bayes import MultinomialNB

# --- Without Preprocessing
# Select Dataset
featTrain, featTest = getTweets(suffix='-prep')
labelTrain, labelTest = getLabels()

pipe = Pipeline([("tfid", TfidfVectorizer()), ("nb", MultinomialNB())])
pipe.fit(featTrain, labelTrain)
pred = pipe.predict(featTest)

resultBefore = getResult(labelTest, pred)

# --- With Preprocessing
# Select Dataset
featTrain, featTest = getTweets(suffix='-prep-lemma')

pipe = Pipeline([("tfid", TfidfVectorizer()), ("nb", MultinomialNB())])
pipe.fit(featTrain, labelTrain)
pred = pipe.predict(featTest)

resultAfter = getResult(labelTest, pred)

# Plot results
f1Plot(resultBefore, resultAfter, resultLabel)

# Compile Results
compileResult(resultBefore, resultAfter, 'Naive Bayes')


#%% Using Naive Bayes
# Lemmatisation VS Removing stop words
from sklearn.naive_bayes import MultinomialNB

# --- Without Preprocessing
# Select Dataset
featTrain, featTest = getTweets(suffix='-prep-lemma')
labelTrain, labelTest = getLabels()

pipe = Pipeline([("tfid", TfidfVectorizer()), ("nb", MultinomialNB())])
pipe.fit(featTrain, labelTrain)
pred = pipe.predict(featTest)

resultBefore = getResult(labelTest, pred)

# --- With Preprocessing
# Select Dataset
featTrain, featTest = getTweets(suffix='-prep-lemma-swSpacy')

pipe = Pipeline([("tfid", TfidfVectorizer()), ("nb", MultinomialNB())])
pipe.fit(featTrain, labelTrain)
pred = pipe.predict(featTest)

resultAfter = getResult(labelTest, pred)

# Plot results
f1Plot(resultBefore, resultAfter, resultLabel)

# Compile Results
compileResult(resultBefore, resultAfter, 'Naive Bayes')
