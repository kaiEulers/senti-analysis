"""
This program computes the precision, recall, and f1-score of Machine Learning algorithms used to predict the sentiment of a collection of tweets.
A comparison is made between a given bad vectorisation of the tweets VS a properly vectorisation of the tweets.
The bad vector was provide by project 2 of Knowlegde Technologies Semester 1 2019.
The good vector is vectorised with vocab count weighted with TF-IDF, with stop words removed.

Created on Sun May 26 17:14:25 2019
@author: kaisoon
"""
#%% Import Libraries & Declare Constants
# --- Constants
resultLabel = ['Bad Vector', 'Good Vector']

# --- Functions
import os
from func import *
import pandas as pd

# --- Libraries
# Feature Extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Pipeline
from sklearn.pipeline import Pipeline


results_bad_good = pd.DataFrame()


#%% Using Support Vector Machine
# WORST RESULTS
from sklearn.svm import SVC

# --- Badly vectorised
featTrain, featTest = getBadVector()
labelTrain, labelTest = getLabels()

clf = SVC()
clf.fit(featTrain, labelTrain)
pred = clf.predict(featTest)

resultBefore = getResult(labelTest, pred)

# --- Properly vectoriser and preprocessed
featTrain, featTest = getTweets(suffix='-prep-lemma-swSpacy')

pipe = Pipeline([("tfid", CountVectorizer()), ("svc", SVC())])
pipe.fit(featTrain, labelTrain)
pred = pipe.predict(featTest)

resultAfter = getResult(labelTest, pred)

# Plot results
f1Plot(resultBefore, resultAfter, resultLabel)

# Compile Results
results_bad_good['SVC'] = compileResult(resultBefore, resultAfter, 'SVC')


#%% Using Linear Support Vector Machine
# Compare with badly vectorised VS properly preprocessed vector

# GREAT RESULTS!
from sklearn.svm import LinearSVC
results = pd.DataFrame()

# --- Badly vectorised
featTrain, featTest = getBadVector()
labelTrain, labelTest = getLabels()

clf = LinearSVC()
clf.fit(featTrain, labelTrain)
pred = clf.predict(featTest)

resultBefore = getResult(labelTest, pred)

# Get results for plot
results_temp = pd.DataFrame([sum(resultBefore[2])/len(resultBefore[2]), resultBefore[3]])
results_temp.index = ['F-Measure', 'Accuracy']
results_temp.columns = ['Badly Vectorised']
results = pd.concat([results, results_temp], axis=1)

# --- Properly vectoriser and preprocessed
featTrain, featTest = getTweets(suffix='-prep-lemma-swSpacy')

pipe = Pipeline([("tfid", CountVectorizer()), ("linearSvc", LinearSVC())])
pipe.fit(featTrain, labelTrain)
pred = pipe.predict(featTest)

resultAfter = getResult(labelTest, pred)

# Get results for plot
results_temp = pd.DataFrame([sum(resultAfter[2])/len(resultAfter[2]), resultAfter[3]])
results_temp.index = ['F-Measure', 'Accuracy']
results_temp.columns = ['Preprocessed']
results = pd.concat([results, results_temp], axis=1)

# Plot results
f1Plot(resultBefore, resultAfter, resultLabel)

# Compile Results
results_bad_good['Linear SVC'] = compileResult(resultBefore, resultAfter)


#%% Using K-Nearest Neighbour
# GOOD RESULTS
from sklearn.neighbors import KNeighborsClassifier
n = 5;

# --- Badly vectorised
featTrain, featTest = getBadVector()
labelTrain, labelTest = getLabels()

clf = KNeighborsClassifier()
clf.fit(featTrain, labelTrain)
pred = clf.predict(featTest)

resultBefore = getResult(labelTest, pred)

# --- Properly vectoriser and preprocessed
featTrain, featTest = getTweets(suffix='-prep-lemma-swSpacy')

pipe = Pipeline([("tfid", TfidfVectorizer()), ("knn", KNeighborsClassifier())])
pipe.fit(featTrain, labelTrain)
pred = pipe.predict(featTest)

resultAfter = getResult(labelTest, pred)

# Plot results
f1Plot(resultBefore, resultAfter, resultLabel)

# Compile Results
results_bad_good['K-Nearest Neighbour'] = compileResult(resultBefore, resultAfter)


#%% Using Naive Bayes
# OK RESULTS
from sklearn.naive_bayes import MultinomialNB

# --- Badly vectorised
featTrain, featTest = getBadVector()
labelTrain, labelTest = getLabels()

clf = MultinomialNB()
clf.fit(featTrain, labelTrain)
pred = clf.predict(featTest)

resultBefore = getResult(labelTest, pred)

# --- Properly vectoriser and preprocessed
featTrain, featTest = getTweets(suffix='-prep-lemma-swSpacy')

pipe = Pipeline([("tfid", TfidfVectorizer()), ("nbc", MultinomialNB())])
pipe.fit(featTrain, labelTrain)
pred = pipe.predict(featTest)

resultAfter = getResult(labelTest, pred)

# Plot results
f1Plot(resultBefore, resultAfter, resultLabel)

# Compile Results
results_bad_good['Naive Bayes'] = compileResult(resultBefore, resultAfter)


#%% Using Random Forest
# GOOD RESULTS
from sklearn.ensemble import RandomForestClassifier

# --- Badly vectorised
featTrain, featTest = getBadVector()
labelTrain, labelTest = getLabels()

clf = RandomForestClassifier()
clf.fit(featTrain, labelTrain)
pred = clf.predict(featTest)

resultBefore = getResult(labelTest, pred)

# --- Properly vectoriser and preprocessed
featTrain, featTest = getTweets(suffix='-prep-lemma-swSpacy')

pipe = Pipeline([("tfid", TfidfVectorizer()), ("rndF", RandomForestClassifier())])
pipe.fit(featTrain, labelTrain)
pred = pipe.predict(featTest)

resultAfter = getResult(labelTest, pred)

# Plot results
f1Plot(resultBefore, resultAfter, resultLabel)

# Compile Results
results_bad_good['Random Forest'] = compileResult(resultBefore, resultAfter)


#%% Save all results

results_bad_good.to_csv("/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/SentiAnalysis/saved/results-bad-good.csv")

#%% Number of features in preprocessed data

featTrain, featTest = getTweets(suffix='-prep-lemma-swSpacy')
tweets = pd.concat([featTrain, featTest])

vectoriser = TfidfVectorizer()
vect = vectoriser.fit_transform(tweets)
print(vect.shape)
# Preprocessed vector contains 48'225 vocab features(words) from 27'913 documents.


#%% Number of features in data that is not preprocessed

featTrain, featTest = getTweets()
tweets = pd.concat([featTrain, featTest])

vectoriser = TfidfVectorizer()
vect = vectoriser.fit_transform(tweets)
print(vect.shape)
# Vector contains 51'820 vocab features(words) from 27'913 documents.

#%% Number of features in badly vectorised data

featTrain, featTest = getBadVector()

vect = pd.concat([featTrain, featTest])
print(vect.shape)
# Bad Vector contains 45 vocab features(words) from 27913 documents. Stop Words are NOT removed.

