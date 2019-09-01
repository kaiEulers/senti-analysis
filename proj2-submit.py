# Import Libraries & Declare Constants
# --- Constants
import spacy
nlp = spacy.load('en_core_web_sm')
# Load stop words provided by Spacy
sWords = list(spacy.lang.en.stop_words.STOP_WORDS)
import string
# Load all punctuations
punc = string.punctuation
resultLabel = ['Bad Vector', 'Good Vector']

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
from func import *
from textPrep import textPrep
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline


# --- Preprocess raw data
# Load data
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
fileName_train = "train-tweets-prep-lemma-swSpacy.csv"
trainTweets_prep.to_csv(
        "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/SentiAnalysis/data/" + fileName_train,
        index=False
        )
# --- Save processed text for evaluation
fileName_eval = "eval-tweets-prep-lemma-swSpacy.csv"
evalTweets_prep.to_csv(
        "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/SentiAnalysis/data/" + fileName_eval,
        index=False
        )


# --- Predict sentiment with SVM
# Initialise dataframe to store results
results = pd.DataFrame()

# --- "Badly vectorised" dataset
featTrain, featTest = getBadVector()
labelTrain, labelTest = getLabels()

clf = LinearSVC()
clf.fit(featTrain, labelTrain)
pred = clf.predict(featTest)
results_badVect = getResult(labelTest, pred)

# Store prediction resutls
results['Badly Vectorised'] = pd.Series([sum(results_badVect[2])/len(results_badVect[2]), results_badVect[3]], index=['F-Measure', 'Accuracy'])

# --- "Preprocessed" dataset
featTrain, featTest = getTweets(suffix='-prep-lemma-swSpacy')

pipe = Pipeline([("tfid", CountVectorizer()), ("linearSvc", LinearSVC())])
pipe.fit(featTrain, labelTrain)
pred = pipe.predict(featTest)
results_prep = getResult(labelTest, pred)

# Store prediction resutls
results['Preprocessed'] = pd.Series([sum(results_prep[2])/len(results_prep[2]), results_prep[3]], index=['F-Measure', 'Accuracy'])


# --- "TF-IDF Weighted" dataset
featTrain, featTest = getTweets(suffix='-prep-lemma-swSpacy')

pipe = Pipeline([("tfid", TfidfVectorizer()), ("linearSvc", LinearSVC())])
pipe.fit(featTrain, labelTrain)
pred = pipe.predict(featTest)

result_tfidf = getResult(labelTest, pred)

# Get results for plot
results['TF-IDF Weighted'] = pd.Series([sum(result_tfidf[2])/len(result_tfidf[2]), result_tfidf[3]], index=['F-Measure', 'Accuracy'])

# Plot and print results
fig, ax = plt.subplots(ncols=2, figsize=(6,3), dpi=100)
results.loc['F-Measure'].plot.bar(title="F-Measure", ax=ax[0])
results.loc['Accuracy'].plot.bar(title="Accuracy", ax=ax[1])
print(results)


#%% Predict test-tweets given by project
import pandas as pd
from func import *

# --- Libraries
# Feature Extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Pipeline
from sklearn.pipeline import Pipeline

# Get data
featTrain, featTest = getTweets(suffix='-prep-lemma-swSpacy')
del featTest
labelTrain, labelTest = getLabels()
del labelTest

testId, testTweet = getTestData()


# Train SVM
pipe = Pipeline([("tfid", TfidfVectorizer()), ("linearSvc", LinearSVC())])
pipe.fit(featTrain, labelTrain)
# Make prediction
pred_submit = pipe.predict(testTweet)

# Save file
pred_submit = pd.DataFrame([testId, pred_submit], ).transpose()
pred_submit.columns = ['id', 'tweet']
#print(pred_submit.head())
#pred_submit.to_csv("/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/SentiAnalysis/saved/predictions.csv", index=False)
pred_submit.to_csv("/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/SentiAnalysis/saved/predictions.txt", sep='\t',  index=False, header=None)

