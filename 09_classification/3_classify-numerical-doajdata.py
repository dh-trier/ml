"""
Script to classify the DOAJ data, 
using a step-by-step implemented Naive Bayes Classifier.

This particular script works with numerical / continuous features.
It implements a Gaussian Naive Bayes Classifier. 

"""


# === Imports ===

import pandas as pd
import numpy as np
from os.path import join, realpath, dirname
from math import sqrt
from math import pi
from math import exp
from collections import Counter
from matplotlib import pyplot as plt



# === Files and folders === 

np.random.seed(42)
workdir = join(realpath(dirname(__file__)))
doajdata = join(workdir, "data", "doaj-journaldata_prepared.csv")



# === Functions === 

def load_data(): 
    """
    Read the prepared CSV metadata file from DOAJ.
    Returns: DataFrame. 
    """
    with open(doajdata, "r", encoding="utf8") as infile: 
        data = pd.read_csv(infile, index_col=0) 
    #print(data.head())
    #print(list(data.columns))
    return data



def select_features(data): 
    """
    From all available features, we only select the ones that are numerical.
    In addition, we select the target category, which is binary. 
    """
    data = data.loc[:,[
        #"gdp",
        "population",
        "publication-time",
        "added-date", 
        #"article-records",
        "APC", # target category
        ]]
    #print(list(data.columns))
    #print(data.dtypes)
    #print("rows=items, columns=features+target:", data.shape)
    return data


def split_data(data):
    """
    Split the dataset into two parts, one for training, one for testing.
    Returns two dataframes. 
    """
    # Create a random order of the dataset rows
    data = data.sample(frac=1).reset_index(drop=True)
    # Define the proportion of the data to be used for training
    numtrain = int(len(data)*0.90)
    train = data.iloc[:numtrain,:]
    test = data.iloc[numtrain:,:]
    #print(train.shape, test.shape)
    return train, test



def calculate_class_probs(train): 
    """
    Calculates the base probabilities of the two classes
    or target categories that are contained in the dataset. 
    The base probabilities are derived from the proportion of each case in the dataset. 
    These are P(H) and P(-H), the a priori probabilities of the classes.
    Returns: two floats
    """
    counts = Counter(train["APC"])
    P_true = counts[True] / (counts[True] + counts[False])
    P_false = counts[False] / (counts[True] + counts[False])
    #print("overall probs true, false:", P_true, P_false)
    return P_true, P_false



def calculate_gaussians(train):
    """
    Groups the data by target class. 
    Then, calculates the mean and standard deviation for each feature for each class. 
    These two values describe the Gaussian distribution of each feature, given each class. 
    These represent P(X|H) from the literature. 
    Gs = Gaussians, containing one Gaussian distribution descriptors per class and feature.
    Returns the grouped data and a table per target class with mean and stdev. 
    """
    # Group and get means and stdevs
    grouped = train.groupby(by="APC")
    grouped_means = grouped.mean()
    grouped_stdevs = grouped.std()
    #print("\nmeans\n", grouped_means)
    #print("\nstdevs\n", grouped_stdevs)
    # Reshape so that each variable contains the data for one class. 
    Gs_true = pd.concat(
        [grouped_means.loc[True,:].rename("means"),
        grouped_stdevs.loc[True,:].rename("stdevs")],
        axis=1)
    Gs_false = pd.concat(
        [grouped_means.loc[False,:].rename("means"),
        grouped_stdevs.loc[False,:].rename("stdevs")],
        axis=1)
    #print("\ntrue\n", Gs_true)
    #print("\nfalse\n", Gs_false)
    return Gs_true, Gs_false



def build_classifier(P_true, P_false, Gs_true, Gs_false): 
    """
    The model consists simply of the list of probabilities and the Gaussian distributions. 
    Returns: dict. 
    """
    clf = {
        "P_true" : P_true, 
        "P_false" : P_false,
        "Gs_true" : Gs_true,
        "Gs_false" : Gs_false,
        }
    return clf



def get_probs(item, mean, stdev):
    """
    For numerical / continuous values. 
    For a given value of x, the group mean and the group stdev, 
    calculates the probability of this value. 
    This function is called by the function that
    creates a table of probabilities for each item in the dataset. 
    Returns: a float (probability)
    """
    #print(item)
    one = 1 / (sqrt(2 * pi) * stdev)
    two = - (1 / 2)
    three = (item - mean)**2
    four = stdev**2  
    prob = one * exp(two * (three / four)) * 1000
    return prob



def apply_classifier(test, clf): 
    """
    The classifier (clf) is applied to the data / instances in the test set. 
    Uses the "get_probs" function to calculate the probability of the feature to be either class. 
    The class with the higher probability is selected as the most likely class. 
    """
    test_X = test.drop(["APC"], axis=1).astype(float)
    #print(test_X.shape)
    #print(test_X.dtypes)
    #print(test_X.head())
    #print(clf["Gs_true"]["mean"])
    features = list(test_X.columns)
    results = test
    results["probs_true"] = 1
    results["probs_false"] = 1
    for feature in features: 
        #print(feature)
        #print(test_X[feature].head())
        #print(clf["Gs_true"]["means"][feature])
        probs_true = test_X[feature].apply(lambda item: get_probs(
            item, 
            clf["Gs_true"]["means"][feature],
            clf["Gs_true"]["stdevs"][feature]))
        probs_false = test_X[feature].apply(lambda item: get_probs(
            item, 
            clf["Gs_false"]["means"][feature],
            clf["Gs_false"]["stdevs"][feature]))
        results["probs_true"] = results["probs_true"] * probs_true
        results["probs_false"] = results["probs_false"] * probs_false
    results["pred"] = results["probs_true"] >= results["probs_false"]
    #print(results.head())
    return results



def evaluate_performance(results): 
    """
    Calculates the base values for performance assessment: 
    true positives, true negatives, false positives, false negatives.
    Based on this, calculates various performance scores: accuracy, precision, recall, F-score. 
    """
    # Split the results dataframe into true and predicted classes
    true = results["APC"]
    pred = results["pred"]

    # Transform Series to numpy array for easier matching. 
    true = true.to_numpy()
    pred = pred.to_numpy()
    #print(type(true), type(pred))
    
    # Check the balance of the predicted values
    #N_abs = len(true)
    P_pred = Counter(pred)[True]
    N_pred = Counter(pred)[False]
    print("predicted classes: true, false:", P_pred, N_pred)

    # Calculate the base values for calculation of performance. 
    #N_abs = len(true)
    P = Counter(true)[True]
    N = Counter(true)[False]
    print("actual classes: true (P), false (N):", P, N)
    
    TP, TN, FP, FN = [0,0,0,0]
    for i in range(0, len(true)): 
        if pred[i] == True == true[i]: 
            TP +=1
        elif pred[i] == False == true[i]: 
            TN +=1
        elif pred[i] == True != true[i]: 
            FP +=1
        elif pred[i] == False != true[i]: 
            FN +=1
    print("TP, TN, FP, FN:", TP, TN, FP, FN)

    # Calculate the performance scores
    accuracy = (TP + TN) / (P + N)
    error_rate =  (FP + FN) / (P + N)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    fscore = (2 * precision * recall) / (precision + recall)
    print("accuracy:", round(accuracy, 3))
    print("error rate:", round(error_rate, 3))
    print("precision:", round(precision, 3))
    print("recall:", round(recall, 3))
    print("fscore:", round(fscore, 3))


# === Main ===

def main(): 
    # Prepare the data
    data = load_data()
    data = select_features(data)
    train, test = split_data(data)
    # Calculate values needed for the classifier using the training data
    P_true, P_false = calculate_class_probs(train)
    Gs_true, Gs_false = calculate_gaussians(train)
    clf = build_classifier(P_true, P_false, Gs_true, Gs_false)
    # Classify the items for the test set
    results = apply_classifier(test, clf)
    # Evaluate the performance of the classifier
    evaluate_performance(results)

main()


