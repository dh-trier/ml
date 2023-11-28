"""
Script implementing k-Means in a step-by-step manner. 
It starts from a small sample dataset about bikes (number of gears and price). 
Then, iteratively, it performs a loop over four steps: 
- Create initial random centroids or recalulate centroids from earlier cluster assignments. 
- Create cluster assignments based on the (random or recalulated) centroids 
- Perform an evaluation of the clustering using purity. 
- Visualize the centroids, resulting cluster assignments, and purity. 

Lessons
- Effect of normalization
- Importance of data container format
- Complexity of simple procedures

Procedure 
- Before running the code, have a good look at the dataset. 
- Check out the plot "data-only+true-classes.png", which shows the dataset. 
- Where would you expect the cluster boundaries to be, and where the centroids?
- Run the clustering and observe the resulting plots one by one. 
- Look at the "kmeans" function first to understand the process. 
- Check out individual functions as well, especially those related to the clustering itself.



"""

# === Imports ===

import os
import seaborn as sns
from matplotlib import pyplot as plt
from os.path import join
import pandas as pd
import random
import numpy as np
import math


# === Parameters / global variables ===

workdir = join(os.path.realpath(os.path.dirname(__file__)))
datafile = join(workdir, "data-initial.csv")



# === Suppress warnings === 

import warnings
warnings.filterwarnings("ignore")



# === Functions: define data === 

def load_data():
    """
    Load the sample data from a file. 
    Performs a simple normalization against the maximum value in each column.
    Returns: DataFrame. 
    """
    #= Load the CSV file
    with open(datafile, "r", encoding="utf8") as infile: 
        data = pd.read_csv(infile, sep=",")  
    #print(data)
    #= Perform the normalization (maximum value => 1)
    #= (If we don't do this, results will be worse.)
    #= (What is the best performance you obtain with and without
    #= this step when running multiple test runs?)
    data["gears"] = round(data["gears"] / max(data["gears"]) *100, 0)
    data["price"] = round(data["price"] / max(data["price"]) *100, 0)
    return data



# === Functions: visualisations === 

def create_scatterplot_data(data): 
    """
    Create a scatterplot with the initial data. 
    Colors show assignment to the true classes.
    Returns: nothing, but writes PNG file to disk.
    """
    sns.set_style("whitegrid")
    plot = sns.scatterplot(
        data=data, 
        x="gears", 
        y="price", 
        hue="type", 
        s=70)
    #plot.ylim=(0,110)
    #plot.xlim=(0,110)
    plt.suptitle("Bicycle data", size=14)
    plt.title("(color=true classes)", size=10)
    plt.savefig(
        join(workdir, "a_data-only+true-classes.png"), 
        dpi=300)



def create_scatterplot_assignments(data, centroids, iter, purity, cluster_sizes): 
    """
    Create a scatterplot that shows three things: the original datapoints,
    the current centroids, and the cluster assignments based on these centroids. 
    Colors show the hypothetical cluster assignments.
    Returns: nothing, but writes PNG file to disk.
    """
    sns.set_style("whitegrid")
    fig,ax = plt.subplots()
    #= Plot the original datapoints as a scatterplot
    plot = sns.scatterplot(data=data, x="gears", y="price", hue="i"+str(iter), s=70)
    #plot.set_ylim(0, 110)
    #plot.set_xlim(0, 110)
    #= Add the two current centroids, including labels
    sns.scatterplot(
        x = [centroids.iloc[0+(iter*2),2], centroids.iloc[1+(iter*2),2]],
        y = [centroids.iloc[0+(iter*2),3], centroids.iloc[1+(iter*2),3]],
        marker="x", color="darkred", s=70, linewidth=2)
    plt.text(
        centroids.iloc[0+(iter*2),2]+1, 
        centroids.iloc[0+(iter*2),3]+1, 
        "0")
    plt.text(
        centroids.iloc[1+(iter*2),2]+1,
        centroids.iloc[1+(iter*2),3]+1,
        "1")
    #= Make sure sizes are compatible
    #= Add plot title and description
    plt.suptitle("Bicycle data", size=14)
    color_str = "color = cluster assignment; "
    purity_str = "purity = " + str(round(purity, 3)) + "; "
    clustersize_str = "cluster sizes: " + str(cluster_sizes[0]) + " and " + str(cluster_sizes[1]) + "."
    plt.title(color_str + purity_str + clustersize_str, size=10)
    #= Save the plot to file
    plt.savefig(
        join(workdir, "b" + str(iter) +"_data+centroids+assignments.png"), 
        dpi=300)



# === Functions: kMeans algorithm === 

def initialise_centroids(data):
    """
    Random initialisation of the two hypothetical centroids. 
    The values are random but within the range of values existing in the data. 
    Returns: DataFrame (permits to add further centroid data). 
    """
    #= Define function to create random floats for x (= gears)
    def getx():
        return round(random.uniform(np.min(data["gears"]), np.max(data["gears"])), 2)
    #= Define function to create random floats for y (=  price)
    def gety():
        return round(random.uniform(np.min(data["price"]), np.max(data["price"])), 2)
    # Define the random centroids (x and y for centroids 0 and 1)
    centroids = [
        [0, 0, getx(), gety()],
        [1, 0, getx(), gety()]]
    # Turn this into a DataFrame
    columns = ["centroid", "iteration", "gears", "price"]
    centroids = pd.DataFrame(centroids, columns=columns)
    # Save the DataFrame to disk
    with open(join(workdir, "centroids.csv"), "w", encoding="utf8") as outfile: 
        centroids.to_csv(outfile)
    return centroids



def determine_closest_centroid(x, y, centroids, iter): 
    """
    This function is called by the function "get_assignments". 
    Determines which centroid is closest to a given data point. 
    This is based (here) on the Euclidean distance; other distance measures may be used. 
    Returns: For the current data point, numerical assignment to one of two centroids (0 or 1). 
    """
    #= Get the euclidean distance between the item and the two centroids
    ed0 = math.dist([x, y], [centroids.iloc[0+(iter*2),2], centroids.iloc[0+(iter*2),3]])
    ed1 = math.dist([x, y], [centroids.iloc[1+(iter*2),2], centroids.iloc[1+(iter*2),3]])
    #= Identify which centroid is closer, add this as the point's assignment
    if min(ed0, ed1) == ed0: 
        assignment = 0
    else:
        assignment = 1
    return assignment



def get_assignments(data, centroids, iter):
    """
    Identify the cluster assignment for each data point based on the current centroids.
    Calls the function "determine_closest_centroid" multiple times, once for each data point. 
    Returns: An augmented DataFrame (additional column with the current cluster assignments.)
    """
    data["i"+str(iter)] = np.nan
    assignments = []
    for i in range(0, data.shape[0]):
        x,y = data.iloc[i,1], data.iloc[i,2]
        assignments.append(determine_closest_centroid(x, y, centroids, iter))
    data["i"+str(iter)] = assignments
    with open(join(workdir, "data-plus.csv"), "w", encoding="utf8") as outfile: 
        data.to_csv(outfile)
    return data



def recalculate_centroids(data, centroids, iter): 
    """
    Based on the cluster assignments, recalculate updated centroids. 
    From the data points assigned to a cluster, get the mean values in each dimension.
    Returns: The centroids dataframe with two new rows for the new centroids.
    """
    for i in [0,1]:
        cluster = data[data["i"+str(iter-1)] == i]
        x_mean = round(np.mean(cluster["gears"]), 2)
        y_mean = round(np.mean(cluster["price"]), 2)
        centroids = centroids.append({
            "centroid": i,
            "iteration" : iter,
            "gears" : x_mean,
            "price" : y_mean}, ignore_index=True)
        centroids = centroids.astype({"centroid" : "int", "iteration" : "int"})
    with open(join(workdir, "centroids.csv"), "w", encoding="utf8") as outfile: 
        centroids.to_csv(outfile)
    return centroids



# === Evaluation functions === 

def calculate_purity(data, iter): 
    """
    Per designated cluster, calculate the proportion of correct assignments. 
    Returns: purity as a float, and list of cluster sizes. 
    """
    from collections import Counter
    majority_counts = []
    cluster_sizes = []
    for i in [0,1]: 
        #= The loop does all this once for cluster 0, then for cluster1. 
        #= Select the data for one hypothetical cluster
        cluster_i = data.loc[data["i"+str(iter)] == i, :]
        #= Determine the cluster size
        cluster_size = len(cluster_i)
        cluster_sizes.append(cluster_size)
        #= Identify the majority class in the hypothetical cluster
        majority_object = Counter(cluster_i["true"]).most_common
        majority = majority_object(1)[0][0]
        majority_count = majority_object(1)[0][1]
        #= Identify the number of items that belong to the majority class
        #majority_count = len(cluster_i[cluster_i["i"+str(iter)] == majority])
        majority_counts.append(majority_count)
    purity = sum(majority_counts) / sum(cluster_sizes)
    #print(str(purity))
    return purity, cluster_sizes


def calculate_rss(data, centroids, iter): 
    """
    For each cluster, calculate the residual sum of squares. 
    Returns: rss as a float. 
    """
    
    # Get the two centroids for the current iteration from the centroids df
    c01 = centroids[centroids.T.loc["iteration"] == iter][["gears", "price"]]
    xy_c0 = np.array(c01.iloc[0,:])
    xy_c1 = np.array(c01.iloc[1,:])
    #print("xy_c0", xy_c0)
    #print("xy_c1", xy_c1)
    
    # Get the point coordinates of the points assigned to clusters 0 and 1,
    # in the current / most recent iteration
    grouper = "i"+str(iter)
    points_grouped = data.groupby(by=grouper)
    points_dict = {}
    for name,group in points_grouped: 
        #print(name)
        x_vals = group.loc[:,"gears"]
        y_vals = group.loc[:,"price"]
        points_dict[name] = np.array(list(zip(x_vals, y_vals)))
        
    # Calculate the squared sums of the differences between each point and the centroid
    # Because the values are defined as np.arrays, the subtraction works with broadcasting. 
    # Do this for the points assigned to cluster 0 against centroid 0, at the current iteration
    sum_sqd_diffs_0 = np.sum(points_dict[0] - xy_c0) ** 2
    #print(sum_sqd_diffs_0)
    # As well as for the points assigned to cluster 1 against centroid 1, at the current iteration
    sum_sqd_diffs_1 = np.sum(points_dict[1] - xy_c1) ** 2
    #print(sum_sqd_diffs_1)
    
    # Sum up the two values to get the RSS
    rss = sum_sqd_diffs_0 + sum_sqd_diffs_1
    #print(rss)

    return rss



# === Coordination function === 

def kmeans():
    """
    Step-by-step implementation of kMeans with visualisations and evaluation.
    """
    print("\nStarting a new run.")
    #= Load the input data from file: Here, bikes with gears and price.
    data = load_data()
    #= Visualize just the input data to be clustered as a scatterplot. 
    create_scatterplot_data(data)
    #= Starting the iteration, with four steps (and one alternative)
    for iter in range(0,6): 
        #= Determine the two centroids (randomly or from assignments)
        if iter == 0:
            #= Initialise random centroids (new DataFrame)
            centroids = initialise_centroids(data) 
        else: 
            #= Recalculate centroids based on the current assignments
            centroids = recalculate_centroids(data, centroids, iter)
        #= Determine the centroid assignments for each datapoint based on the current centroids
        data = get_assignments(data, centroids, iter)
        #= Evaluate the clustering quality, for the current centroids / assignments
        purity, cluster_sizes = calculate_purity(data, iter)
        rss = calculate_rss(data, centroids, iter)
        #= Visualise the data with the hypothetical cluster assignments
        create_scatterplot_assignments(data, centroids, iter, purity, cluster_sizes)
        #= Print the results for the current iteration
        print(
            "- Done with iteration", str(iter), 
            "| purity:", str(np.round(purity,3)), 
            "| RSS", str(np.round(rss,3))
            )

    print("Done with this run.\n")

kmeans()
