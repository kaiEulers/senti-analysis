"""
Created on Sun May 26 16:21:38 2019

@author: kaisoon
"""
# ------------------------------------------------------------
def getLabels():
    """
    getLabels() returns series (labelTrain, labelTest) as per data given by subject Knowledge Technologies Project 2.
    The Series portions are used for training and evaluating a ML algorithm.
    Note that training data has already been vectorised, though it is badly vectorised.
    """
    import pandas as pd

    # Open train-labels.csv
    labelTrain = pd.read_csv(
            "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/SentiAnalysis/data/train-labels.csv"
            )
    labelTrain = labelTrain['label']

    # Open eval-labels.csv
    labelTest = pd.read_csv(
            "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/SentiAnalysis/data/eval-labels.csv"
            )
    labelTest = labelTest['label']

    return (labelTrain, labelTest)


# ------------------------------------------------------------
def getBadVector():
    """
    getData_badVector() returns series (featTrain, featTest) as per data given by subject Knowledge Technologies Project 2.
    The Series portions are used for training and evaluating a ML algorithm.
    Note that training data has already been vectorised, though it is badly vectorised.
    """
    import pandas as pd
    # Open train.csv
    featTrain = pd.read_csv(
            "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/SentiAnalysis/data/train.csv"
            )
    featTrain.drop("id", axis=1, inplace=True)

    # Open eval.csv
    featTest = pd.read_csv(
            "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/SentiAnalysis/data/eval.csv"
            )
    featTest.drop("id", axis=1, inplace=True)

    return (featTrain, featTest)


# ------------------------------------------------------------
def getTestData():

    import pandas as pd
    # Open train.csv
    testData = pd.read_csv(
            "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/SentiAnalysis/data/test-tweets-prep-lemma-swSpacy.csv"
            )

    testId = testData["id"]
    testTweet = testData["tweet"]

    return (testId, testTweet)


# ------------------------------------------------------------
def getTweets(suffix=''):
    """
    getTweets() returns series (featTrain, featTest) as per data given by subject Knowledge Technologies Project 2.
    The four Series are used for training and evaluating a ML algorithm.
    """
    import pandas as pd

    # Open train-tweets.csv
    path_train = "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/SentiAnalysis/data/train-tweets" + suffix + ".csv"
    featTrain = pd.read_csv(path_train)
    featTrain = featTrain['tweet']

    # Open eval-tweets.csv
    path_eval = "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/SentiAnalysis/data/eval-tweets" + suffix + ".csv"
    featTest = pd.read_csv(path_eval)
    featTest = featTest['tweet']

    return (featTrain, featTest)


# ------------------------------------------------------------
def getData_rndSplit(suffix='', testSize=0.33):
    """
    getData_rndSplit() returns four Series (featTrain, featTest, labelTrain, labelTest). Using data given by subject Knowledge Technologies Project 2, the data is randomly split into the above four Series used for training and evaluating a ML algorithm.
    testSize is the percentage of the data that is set aside for evaluating the ML algorithm
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Open train-tweets.csv
    path_train_tweets = "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/SentiAnalysis/data/train-tweets" + suffix + ".csv"
    tweet1 = pd.read_csv(path_train_tweets)

    # Open eval-tweets.csv
    path_eval_tweets = "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/SentiAnalysis/data/eval-tweets" + suffix + ".csv"
    tweet2 = pd.read_csv(path_eval_tweets)

    # Concatenate tweet data
    tweets = pd.concat([tweet1, tweet2], ignore_index=True)

    # Open train-labels.csv
    path_train_labels = "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/SentiAnalysis/data/train-labels.csv"
    label1 = pd.read_csv(path_train_labels)

    # Open eval-labels.csv
    path_eval_labels = "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/SentiAnalysis/data/eval-labels.csv"
    label2 = pd.read_csv(path_eval_labels)

    # Concatenate label data
    labels = pd.concat([label1, label2], ignore_index=True)
    # Merge data
    data = pd.merge(tweets, labels, how='inner', on='id')

    # Split data
    tweets = data['tweet']
    labels = data['label']
    if tweets.shape == labels.shape:
        # Split dataset into four portions
        return train_test_split(tweets, labels, test_size=testSize)

    else:
        print("ERROR: features and label data do not have the same length!")
        return []


# ------------------------------------------------------------
def printClfReport(labelTest, pred):
    """
    printReport() prints the confustion matrix and the classification report after comparing the predicted labels and the label used for evaluation.
    """
    import pandas as pd
    from sklearn.metrics import confusion_matrix, classification_report

    # --- Print Confusion Matrix and Classification Report
    cm_vcnt = pd.DataFrame(confusion_matrix(labelTest, pred))
    print(cm_vcnt)
    print(classification_report(labelTest, pred))


# ------------------------------------------------------------
def getResult(labelTest, pred):
    """
    result() returns a dataframe containing the precision, recall, and f1-score and accuracy after comparing the predicted labels and the label used for evaluation.
    """
    from sklearn import metrics

    prec = metrics.precision_score(labelTest, pred, average=None)
    recall = metrics.recall_score(labelTest, pred, average=None)
    f1 = metrics.f1_score(labelTest, pred, average=None)
    acc = metrics.accuracy_score(labelTest, pred)

    return [prec, recall, f1, acc]


# ------------------------------------------------------------
def f1Plot(result1, result2, resultLabel):
    """
    f1Plot plots the precision, recall, and f1-score of two evaluation results, namely in the program, the results from  using a count vector and the results from using a tfidf weighted vector in training a ML algorithm fr sentiment analysis.
    vcnt = [prec_vcnt, recall_vcnt, f1_vcnt]
    tfidf = [prec_tfidf, recall_tfidf, f1_tfidf]
    """
    import numpy as np
    import matplotlib.pyplot as plt
    plt.style.use('seaborn')

    # --- Constants
    index = np.arange(3)
    width = 0.35
    labels = ('Neg.', 'Neut.', 'Pos.')

    # --- Data
    prec1 = result1[0]
    rec1 = result1[1]
    f1_1 = result1[2]
    f1Avg1 = sum(f1_1)/len(f1_1)
    acc1 = result1[3]

    prec2 = result2[0]
    rec2 = result2[1]
    f1_2 = result2[2]
    f1Avg2 = sum(f1_2)/len(f1_2)
    acc2 = result2[3]


    fig1, ax = plt.subplots(ncols=3, figsize=(9,3), dpi=100)
    # --- Plot Precisions
    ax[0].bar(
            index - width/2,
            prec1,
            width, label=resultLabel[0], zorder=2
            )
    ax[0].bar(
            index + width/2,
            prec2,
            width, label=resultLabel[1], zorder=2
            )
    ax[0].set_title('Precision')

    # --- Plot Recalls
    ax[1].bar(
            index - width/2,
            rec1,
            width, label=resultLabel[0], zorder=2
            )
    ax[1].bar(
            index + width/2,
            rec2,
            width, label=resultLabel[1], zorder=2
            )
    ax[1].set_title('Recall')
#    ax[1].set_yticklabels('')

    # --- Plot F1-Score
    ax[2].bar(
            index - width/2,
            f1_1,
            width, label=resultLabel[0], zorder=2
            )
    ax[2].bar(
            index + width/2,
            f1_2,
            width, label=resultLabel[1], zorder=2
            )
    ax[2].set_title('F1-Score')
#    ax[2].set_yticklabels('')
    ax[2].legend(loc="best", fontsize='small')

    for a in ax:
        a.set_xticks(index)
        a.set_xticklabels(labels)
        a.grid(linestyle=':')
        a.set_ylim([0, 1])


#    fig2, ax = plt.subplots(ncols=2, figsize=(6,3), dpi=100)
#    # --- Plot Average F1-Score
#    ax[0].bar(
#            [resultLabel[0], resultLabel[1]],
#            [f1Avg1, f1Avg2],
#            width, align = 'center', color=['#1f77b4', '#ff7f0e'], zorder=2
#            )
#    ax[0].set_title('Average F1-Score')
##    ax[0].set_yticklabels('')
#
#    # --- Plot Accuracy
#    ax[1].bar(
#            [resultLabel[0], resultLabel[1]],
#            [acc1, acc2],
#            width, align = 'center', color=['#1f77b4', '#ff7f0e'], zorder=2
#            )
#    ax[1].set_title('Accuracy')
#    ax[1].set_yticklabels('')
#
#    for a in ax:
#        a.grid(linestyle=':')
#        a.set_ylim([0, 1])

    plt.tight_layout()
    plt.show()


def compareResult(resultBefore, resultAfter):
    """
    compareResult() returns the comparision of the average F1-score and the accuracy of the prediciton results.
    """
    import pandas as pd

    f1Avg_before = sum(resultBefore[2])/len(resultBefore[2])
    f1Avg_after = sum(resultAfter[2])/len(resultAfter[2])
    acc_before = resultBefore[3]
    acc_after = resultAfter[3]

    f1Avg_percIncre = (f1Avg_after - f1Avg_before)/f1Avg_before * 100
    acc_percIncre = (resultAfter[3] - resultBefore[3])/resultBefore[3]* 100

    indexNames = [
            'Average F1-Score before',
            'Average F1-Score after',
            'Average F1-Score Percentage Increase',
            'Accuracy before',
            'Accuracy after',
            'Accuracy Percentage Increase'
            ]
    result = pd.Series([f1Avg_before, f1Avg_after, f1Avg_percIncre, acc_before, acc_after, acc_percIncre], index=indexNames)
    print(round(result, 2))

    return result