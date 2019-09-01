"""
Find best n value for K-Nearest Neighbour ML

Created on Tue May 28 16:13:44 2019
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


#%% Using K-Nearest Neighbour
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import os

N = 50
results_KNN = pd.DataFrame()
for n in range(5, N+1):
    # Select Dataset
    featTrain, featTest = getTweets(suffix='-prep-lemma-swSpacy')
    labelTrain, labelTest = getLabels()
    #featTrain, featTest, labelTrain, labelTest = getData_rndSplit(suffix='-prep-lemma-swSpacy')

    # --- Using Vocab Count
    pipe = Pipeline([("cnt", CountVectorizer()), ("knn", KNeighborsClassifier(n_neighbors=n))])
    pipe.fit(featTrain, labelTrain)
    pred = pipe.predict(featTest)

    resultBefore = result(labelTest, pred)

    # --- Using Vocab Count weighted with TFIDF
    pipe = Pipeline([("cnt", TfidfVectorizer()), ("lnn", KNeighborsClassifier(n_neighbors=n))])
    pipe.fit(featTrain, labelTrain)
    pred = pipe.predict(featTest)

    resultAfter = result(labelTest, pred)

    # Compile Results
    result_KNN = compileResult(resultBefore, resultAfter, 'K-Nearest Neighbour')
    result_KNN.columns = [n]
    results_KNN = pd.concat([results_KNN, result_KNN], axis=1)

os.system('say "Complete"')


#%% Plot results

results_KNN.iloc[2].plot()
results_KNN.iloc[5].plot()