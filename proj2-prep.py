"""
This program computes the precision, recall, and f1-score of Machine Learning algorithms used to predict the sentiment of a collection of tweets.
A comparison is made between not removing stop words from the vector and removing stop words from the vector.

Created on Sun May 26 16:28:11 2019
@author: kaisoon
"""
#%% Import Libraries & Declare Constants
# --- Constants
resultLabel = ['Not preprocessed', 'Preprocessed']

# Choose which stopwords library to use
#swSuffix = 'Nltk'
swSuffix = 'Spacy'

# Choose which kind of vectorisation method to sue
#vectoriser = 'count'
vectoriser = 'tfidf'

# --- Functions
import os
import pandas as pd
from func import *

# --- Libraries
# Feature Extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Pipeline
from sklearn.pipeline import Pipeline

results_noPrep_prep = pd.DataFrame()


#%% Using Naive Bayes
# GOOD RESULTS
from sklearn.naive_bayes import MultinomialNB

# --- Without Preprocessing
# Select Dataset
featTrain, featTest = getTweets()
labelTrain, labelTest = getLabels()

if vectoriser == 'count':
    pipe = Pipeline([("tfid", CountVectorizer()), ("nb", MultinomialNB())])
else:
    pipe = Pipeline([("tfid", TfidfVectorizer()), ("nb", MultinomialNB())])
pipe.fit(featTrain, labelTrain)
pred = pipe.predict(featTest)

resultBefore = getResult(labelTest, pred)

# --- With Preprocessing
# Select Dataset
suffix = '-prep-lemma-sw' + swSuffix
featTrain, featTest = getTweets(suffix=suffix)

if vectoriser == 'count':
    pipe = Pipeline([("tfid", CountVectorizer()), ("nb", MultinomialNB())])
else:
    pipe = Pipeline([("tfid", TfidfVectorizer()), ("nb", MultinomialNB())])
pipe.fit(featTrain, labelTrain)
pred = pipe.predict(featTest)

resultAfter = getResult(labelTest, pred)

# Plot results
f1Plot(resultBefore, resultAfter, resultLabel)

# Compile Results
results_noPrep_prep['Naive Bayes'] = compileResult(resultBefore, resultAfter)


#%% Using Random Forest
# GOOD RESULTS
from sklearn.ensemble import RandomForestClassifier

# --- Without Preprocessing
# Select Dataset
featTrain, featTest = getTweets()
labelTrain, labelTest = getLabels()

if vectoriser == 'count':
    pipe = Pipeline([("tfid", CountVectorizer()), ("rndF", RandomForestClassifier())])
else:
    pipe = Pipeline([("tfid", TfidfVectorizer()), ("rndF", RandomForestClassifier())])
pipe.fit(featTrain, labelTrain)
pred = pipe.predict(featTest)

resultBefore = getResult(labelTest, pred)

# --- With Preprocessing
# Select Dataset
suffix = '-prep-lemma-sw' + swSuffix
featTrain, featTest = getTweets(suffix=suffix)

if vectoriser == 'count':
    pipe = Pipeline([("tfid", CountVectorizer()), ("rndF", RandomForestClassifier())])
else:
    pipe = Pipeline([("tfid", TfidfVectorizer()), ("rndF", RandomForestClassifier())])
pipe.fit(featTrain, labelTrain)
pred = pipe.predict(featTest)

resultAfter = getResult(labelTest, pred)

print(resultAfter[2])

# Plot results
f1Plot(resultBefore, resultAfter, resultLabel)

# Compile Results
results_noPrep_prep['Random Forest'] = compileResult(resultBefore, resultAfter)


#%% Using Linear Support Vector Machine
# BAD RESULTS
from sklearn.svm import LinearSVC

# --- Without Preprocessing
# Select Dataset
featTrain, featTest = getTweets()
labelTrain, labelTest = getLabels()

if vectoriser == 'count':
    pipe = Pipeline([("tfid", CountVectorizer()), ("linearSVC", LinearSVC())])
else:
    pipe = Pipeline([("tfid", TfidfVectorizer()), ("linearSVC", LinearSVC())])
pipe.fit(featTrain, labelTrain)
pred = pipe.predict(featTest)

resultBefore = getResult(labelTest, pred)

# --- With Preprocessing
# Select Dataset
suffix = '-prep-lemma-sw' + swSuffix
featTrain, featTest = getTweets(suffix=suffix)

if vectoriser == 'count':
    pipe = Pipeline([("tfid", CountVectorizer()), ("linearSVC", LinearSVC())])
else:
    pipe = Pipeline([("tfid", TfidfVectorizer()), ("linearSVC", LinearSVC())])
pipe.fit(featTrain, labelTrain)
pred = pipe.predict(featTest)

resultAfter = getResult(labelTest, pred)

# Plot results
f1Plot(resultBefore, resultAfter, resultLabel)

# Compile Results
results_noPrep_prep['Linear SVC'] = compileResult(resultBefore, resultAfter)


#%% Using K-Nearest Neighbour
# BAD RESULTS
from sklearn.neighbors import KNeighborsClassifier

# --- Without Preprocessing
# Select Dataset
featTrain, featTest = getTweets()
labelTrain, labelTest = getLabels()

if vectoriser == 'count':
    pipe = Pipeline([("tfid", CountVectorizer()), ("knn", KNeighborsClassifier())])
else:
    pipe = Pipeline([("tfid", TfidfVectorizer()), ("knn", KNeighborsClassifier())])
pipe.fit(featTrain, labelTrain)
pred = pipe.predict(featTest)

resultBefore = getResult(labelTest, pred)

# --- With Preprocessing
# Select Dataset
suffix = '-prep-lemma-sw' + swSuffix
featTrain, featTest = getTweets(suffix=suffix)

if vectoriser == 'count':
    pipe = Pipeline([("tfid", CountVectorizer()), ("knn", KNeighborsClassifier())])
else:
    pipe = Pipeline([("tfid", TfidfVectorizer()), ("knn", KNeighborsClassifier())])
pipe.fit(featTrain, labelTrain)
pred = pipe.predict(featTest)

resultAfter = getResult(labelTest, pred)

# Plot results
f1Plot(resultBefore, resultAfter, resultLabel)

# Compile Results
results_noPrep_prep['K-Nearest Neighbour'] = compileResult(resultBefore, resultAfter)


#%% Compile all results into a dataframe
# Results uses stop words from Spacy


path_results = "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/SentiAnalysis/saved/results-noPrep-prep"+ swSuffix +"_" + vectoriser +".csv"

if vectoriser == 'count' and swSuffix == 'Spacy':
    results_noPrep_prep.to_csv(path_results)

elif vectoriser == 'count' and swSuffix == 'Nltk':
    results_noPrep_prep.to_csv(path_results)

elif vectoriser == 'tfidf' and swSuffix == 'Spacy':
    results_noPrep_prep.to_csv(path_results)

elif vectoriser == 'tfidf' and swSuffix == 'Nltk':
    results_noPrep_prep.to_csv(path_results)

os.system('say "Complete"')

