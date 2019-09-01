"""
This program computes the precision, recall, and f1-score of Machine Learning algorithms used to predict the sentiment of a collection of tweets.
A comparison is made between vectorising the tweets with vocab count VS vocab count weighted with TF-IDF.

Created on Sun May 26 16:28:11 2019
@author: kaisoon
"""
#%% Import Libraries & Declare Constants
# --- Constants
resultLabel = ['Count', 'TF-IDF']

# --- Functions
import os
import pandas as pd
from func import *

# --- Libraries
import pandas as pd
# Feature Extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Pipeline
from sklearn.pipeline import Pipeline


results_vct_tfidf = pd.DataFrame()


#%% Using Naive Bayes
# BAD RESULTS
from sklearn.naive_bayes import MultinomialNB

# Select Dataset
featTrain, featTest = getTweets(suffix='-prep-lemma-swSpacy')
labelTrain, labelTest = getLabels()

# --- Vocab Count
pipe = Pipeline([("cnt", CountVectorizer()), ("nb", MultinomialNB())])
pipe.fit(featTrain, labelTrain)
pred = pipe.predict(featTest)

resultBefore = getResult(labelTest, pred)

# --- Vocab Count weighted with TFIDF
pipe = Pipeline([("cnt", TfidfVectorizer()), ("nb", MultinomialNB())])
pipe.fit(featTrain, labelTrain)
pred = pipe.predict(featTest)

resultAfter = getResult(labelTest, pred)

# Plot results
f1Plot(resultBefore, resultAfter, resultLabel)

# Compile Results
results_vct_tfidf['Naive Bayes'] = compileResult(resultBefore, resultAfter)


#%% Using Random Forest
# BAD RESULTS
from sklearn.ensemble import RandomForestClassifier

# Select Dataset
featTrain, featTest = getTweets(suffix='-prep-lemma-swSpacy')
labelTrain, labelTest = getLabels()

# --- Vocab Count
pipe = Pipeline([("cnt", CountVectorizer()), ("randf", RandomForestClassifier())])
pipe.fit(featTrain, labelTrain)
pred = pipe.predict(featTest)

resultBefore = getResult(labelTest, pred)

# --- Vocab Count weighted with TFIDF
pipe = Pipeline([("cnt", TfidfVectorizer()), ("randf", RandomForestClassifier())])
pipe.fit(featTrain, labelTrain)
pred = pipe.predict(featTest)

resultAfter = getResult(labelTest, pred)

# Plot results
f1Plot(resultBefore, resultAfter, resultLabel)

# Compile Results
results_vct_tfidf['Random Forest'] = compileResult(resultBefore, resultAfter)


#%% Using Support Vector Machine
from sklearn.svm import SVC

# Select Dataset
featTrain, featTest = getTweets(suffix='-prep-lemma-swSpacy')
labelTrain, labelTest = getLabels()

# --- Using Vocab Count
pipe = Pipeline([("cnt", CountVectorizer()), ("svc", SVC())])
pipe.fit(featTrain, labelTrain)
pred = pipe.predict(featTest)

resultBefore = getResult(labelTest, pred)

# --- Using Vocab Count weighted with TFIDF
pipe = Pipeline([("cnt", TfidfVectorizer()), ("svc", LinearSVC())])
pipe.fit(featTrain, labelTrain)
pred = pipe.predict(featTest)

resultAfter = getResult(labelTest, pred)

# Plot results
f1Plot(resultBefore, resultAfter, resultLabel)

# Compile Results
results_vct_tfidf['SVC'] = compileResult(resultBefore, resultAfter)


#%% Using Linear Support Vector Machine
# OK RESULTS
from sklearn.svm import LinearSVC

# Select Dataset
#featTrain, featTest = getTweets(suffix='-prep-lemma-swSpacy')
#labelTrain, labelTest = getLabels()
featTrain, featTest, labelTrain, labelTest = getData_rndSplit()

# --- Using Vocab Count
pipe = Pipeline([("cnt", CountVectorizer()), ("linearSVC", LinearSVC())])
pipe.fit(featTrain, labelTrain)
pred = pipe.predict(featTest)

resultBefore = getResult(labelTest, pred)

# --- Using Vocab Count weighted with TFIDF
pipe = Pipeline([("cnt", TfidfVectorizer()), ("linearSVC", LinearSVC())])
pipe.fit(featTrain, labelTrain)
pred = pipe.predict(featTest)

resultAfter = getResult(labelTest, pred)

# Plot results
f1Plot(resultBefore, resultAfter, resultLabel)

# Compile Results
results_vct_tfidf['Linear SVC'] = compileResult(resultBefore, resultAfter)


#%% Using K-Nearest Neighbour
# GOOD RESULTS
from sklearn.neighbors import KNeighborsClassifier
n = 20

# Select Dataset
featTrain, featTest = getTweets(suffix='-prep-lemma-swSpacy')
labelTrain, labelTest = getLabels()
#featTrain, featTest, labelTrain, labelTest = getData_rndSplit(suffix='-prep-lemma-swSpacy')

# --- Using Vocab Count
pipe = Pipeline([("cnt", CountVectorizer()), ("knn", KNeighborsClassifier(n_neighbors=n))])
pipe.fit(featTrain, labelTrain)
pred = pipe.predict(featTest)

resultBefore = getResult(labelTest, pred)

# --- Using Vocab Count weighted with TFIDF
pipe = Pipeline([("cnt", TfidfVectorizer()), ("lnn", KNeighborsClassifier(n_neighbors=n))])
pipe.fit(featTrain, labelTrain)
pred = pipe.predict(featTest)

resultAfter = getResult(labelTest, pred)

# Plot results
f1Plot(resultBefore, resultAfter, resultLabel)

# Compile Results
results_vct_tfidf['K-Nearest Neighbour'] = compileResult(resultBefore, resultAfter)


#%% Save all results to csv file
results_vct_tfidf.to_csv("/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/SentiAnalysis/saved/results-vct-tfidf.csv")

os.system('say "Complete"')

