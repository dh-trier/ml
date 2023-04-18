"""
Script to use different classifiers from sklearn to classify the DOAJ journal data. 
https://scikit-learn.org/stable/supervised_learning.html#supervised-learning

On MultinomialNB, see: 
https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB

On GaussianNB, see: 
https://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes

On SCM/SVC, see: 
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

"""

# === Imports ===

# general
import pandas as pd
import numpy as np
from os.path import join, realpath, dirname
from collections import Counter
from matplotlib import pyplot as plt

# classifiers
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm

# evaluation helpers
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import ConfusionMatrixDisplay as cmd
from sklearn.metrics import classification_report as cr
from sklearn.metrics import precision_recall_fscore_support as prfs

# === Files and folders === 

#np.random.seed(42)
workdir = join(realpath(dirname(__file__)))
doajdata = join(workdir, "data", "doaj-journaldata_prepared.csv")

# === Functions === 

def load_data(): 
    """
    Loads the prepared CSV file. 
    Returns: DataFrame. 
    """
    with open(doajdata, "r", encoding="utf8") as infile: 
        data = pd.read_csv(infile, index_col=0) 
    #print(data.head())
    #print(list(data.columns))
    return data


def select_data(data): 
    """
    Selects, from the entire dataset, the columns (=features) 
    to be used for prediction (X) and the category used as target 
    for the classification (y). 
    Returns: A DataFrame for X, a Series for y.
    Both share the same index.  
    """
    X = data.loc[:,[
        #== numerical (continuous, this is the complete list)
        "publication-time",
        "added-date", 
        "article-records",
        #"population",
        #"gdp",
        #== categorical (True/False)
        #"author-copyright",
        #"created-early",
        #"publication-time-fast",
        # publisher countries (ordered here by descending number of journals)
        #"pc_Indonesia",
        #"pc_United Kingdom",
        #"pc_Brazil",
        #"pc_United States",
        #"pc_Spain",
        #"pc_Poland",
        #"pc_Iran",
        #"pc_Switzerland",
        #"pc_Russia",
        #"pc_Italy",
        #"pc_Turkey",
        #"pc_Colombia",
        #"pc_Ukraine",
        #"pc_Netherlands",
        #"pc_Romania",
        #"pc_Argentina",
        #"pc_Germany",
        #"pc_India",
        #"pc_France",
        #"pc_China",
        #"pc_Serbia",
        #"pc_Canada",
        #"pc_Mexico",
        # domains (all, ordered by descending number of journals)
        #"d_SocialSciences",
        #"d_Medicine",
        #"d_Humanities",
        #"d_Science",
        #"d_Technology",
        #"d_Education",
        #"d_Agriculture",
        #"d_LIS",
        #"d_General",
        # languages (unordered, all languages retained as one-hot-vectors in dataset)
        #"lang_English",
        #"lang_German",
        #"lang_French",
        #"lang_Spanish",
        #"lang_Italian",
        #"lang_Portuguese",
        #"lang_Indonesian",
        #"lang_Chinese",
        #"lang_Arabic",
        #"lang_Malay",
        #"lang_Swedish",
        #"lang_Polish",
        #"lang_Hungarian",
        #"lang_Czech",
        #"lang_Greek",
        #"lang_Danish",
        #"lang_Catalan",
        #"lang_Croatian",
        #"lang_Romanian",
        #"lang_Ukrainian",
        #"lang_Japanese",
        #"lang_Persian",
        #"lang_Russian",
        #"lang_Serbian",
        #"lang_Turkish"
        ]]
    y = data.loc[:,"APC"]
    print("Items available, features selected:", X.shape)
    return X, y


def split_data(X, y): 
    """
    Splits the dataset (both X and y) into a training and test set. 
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    return X_train, X_test, y_train, y_test


def use_classifier(X_train, X_test, y_train, y_test, classifier):
    """
    Uses one of several classifiers to classify the data (which one: parameter). 
    Returns the predicted classes for the test set as a numpy array. 
    """
    clf = classifier
    trained_clf = clf.fit(X_train, y_train)
    y_pred = trained_clf.predict(X_test)
    #print(type(y_pred), type(y_train))
    #print(len(y_pred), len(y_test))
    return y_pred



def evaluate_performance(y_pred, y_test, classifier): 
    """
    Calculates performance scores and create a confusion matrix. 
    """
    # Create an automated performance report
    report = cr(y_test, y_pred, labels=None, target_names=None, sample_weight=None, 
                digits=2, output_dict=False, zero_division='warn')
    print(report)
    # Create and save the confusion matrix
    conf_matrix = cm(y_test, y_pred, labels=None, sample_weight=None, normalize="true")
    plot = cmd(conf_matrix).plot()
    plt.savefig(join(workdir, "figures", "heatmap_confusion-matrix_"+ str(classifier)[:-2] +".png"), dpi=300)
    plt.close()


# === Main === 

def main(): 
    data = load_data()
    X, y = select_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    for classifier in [GaussianNB(), MultinomialNB(), svm.SVC()]: 
        print("\nUsing", classifier)
        y_pred = use_classifier(X_train, X_test, y_train, y_test, classifier)
        evaluate_performance(y_pred, y_test, classifier)

main()