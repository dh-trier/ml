"""
Script to illustrate several measures of clustering quality: 
Cluster Purity (CP), Rand Index (RI) and F1-Score (F1). 
Based on Manning et al., "Chapter 16: Clustering", Information Retrieval.
"""


import numpy as np
from collections import Counter
import itertools
import math
import pandas as pd

# Define the clusters as in the book chapter
# (Depending on how you modify the clusters, what change in scores do you expect?)
clusters = {
    "c1" : ["x", "x", "x", "x", "x", "o"],
    "c2" : ["x", "o", "o", "o", "o", "d"],
    "c3" : ["x", "x", "d", "d", "d"],          
    }


def get_CP(): 
    """
    Calculate cluster purity CP. 
    CP = average proportion of items belonging to the majority group
    in each cluster among all items.
    """
    all_items = []
    all_majoritycounts = []
    for cluster,items, in clusters.items():
        #print(cluster, items)
        # Get majority item with m count and its count. 
        items_counts = dict(Counter(items))
        item_majority = max(items_counts, key=items_counts.get)
        item_majority_count = items_counts.get(item_majority)
        # Collect them all together, as well as the number of items
        all_majoritycounts.append(item_majority_count)
        all_items.append(len(items))
    #print(all_majoritycounts, np.sum(all_majoritycounts))
    #print(all_items, np.sum(all_items))
    # Calculate CP by dividing sum of majority counts by total items 
    CP = np.sum(all_majoritycounts) / np.sum(all_items)
    print("CP =", np.round(CP,3))


def get_scores(): 
    """
    Calculate values needed for Rand Index and F1-Score. 
    Based on true positives, true negatives, false positives, false negatives. 
    The following is a step-by-step implementation for clarity, withouth shortcuts.
    """
    # For each item in the dataset, get each item with its cluster assignment
    allitems = []
    for cluster,items in clusters.items(): 
        allitems.extend([(item, cluster) for item in items])
    #print(allitems)

    # For each item with its cluster assignment, get all its combinations
    combs_with_cluster = list([item for item in itertools.combinations(allitems, 2)])
    #print(combs_with_cluster)

    # For each combination, check agreement of item and cluster
    # Increment TP, TN, FP, FN depending on the result of the check
    TP, TN, FP, FN = [0,0,0,0]
    # Always compare items first [0][0] to [1][0], then cluster assignment [0][1] to [1][1]
    for item in combs_with_cluster: 
        # identical items, identical clusters => TP
        if item[0][0] == item[1][0] and item[0][1] == item[1][1]: 
            TP +=1
        # identical items, different clusters => TN
        if item[0][0] != item[1][0] and item[0][1] != item[1][1]: 
            TN +=1
        # different items, same clusters => FP
        if item[0][0] != item[1][0] and item[0][1] == item[1][1]: 
            FP +=1
        # different item, different clusters => FN
        if item[0][0] == item[1][0] and item[0][1] != item[1][1]: 
            FN +=1
    #print(TP, TN, FP, FN)
    return TP,TN,FP,FN


def get_RI(TP,TN,FP,FN):     
    """
    Based on TP, TN, FP and FN, calculate the Rand Index RI
    """
    RI = (TP+TN) / (TP+TN+FP+FN)
    print("RI =", np.round(RI,3))
    

def get_F1(TP, TN, FP, FN): 
    """
    Based on the same, calculate F1-Score (P=precision, R=recall)
    """
    P = TP / (TP+FP)
    R = TP / (TP+FN)
    FScore = (2*P*R) / (P+R)
    print("F1 =", np.round(FScore,3))
    

def main(): 
    get_CP()
    TP,TN,FP,FN = get_scores()
    get_RI(TP,TN,FP,FN)
    get_F1(TP,TN,FP,FN)
main()