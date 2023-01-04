"""
Skript, das als Beispiel für Clustering als ML-Methode dient. 

Transformation eines Korpus von Textdateien in eine Term-Dokument-Matrix: 
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer



"""

# === Importe === 

# Allgemein
import os
from os.path import join
from glob import glob
import pandas as pd
import numpy as np

# ML
import sklearn
from sklearn.feature_extraction.text import CountVectorizer as cv
from sklearn.cluster import Birch
from sklearn.cluster import KMeans
from sklearn import metrics


# === Globale Variablen === 

workdir = join(os.path.realpath(os.path.dirname(__file__)))
#= Bikes data
datafile = join(workdir, "data-bikes.csv")
#= Textual data
textfolder = join(workdir, "data", "*.txt")


# === Funktionen === 

def load_data():
    """
    Load the bike sample data from a file. 
    Performs a simple normalization against the maximum value in each column.
    Returns: DataFrame and list of ids.  
    """
    #= Load the CSV file
    with open(datafile, "r", encoding="utf8") as infile: 
        bikedata = pd.read_csv(infile, sep=",")  
    bikedata["label+type"] = bikedata["label"] + "_" + bikedata["type"]
    bikedata.set_index("label+type", inplace=True)
    #print(bikedata)
    #= Perform the normalization (maximum value => 1)
    bikedata["gears"] = round(bikedata["gears"] / max(bikedata["gears"]) *100, 0)
    bikedata["price"] = round(bikedata["price"] / max(bikedata["price"]) *100, 0)
    bikeids = list(bikedata.index)
    bikedata.drop(labels=["type", "true", "label"], axis=1, inplace=True)
    print(bikedata.head())
    #print(bikeids)
    return bikedata, bikeids



def create_dtm(textfolder):
    """
    From the corpus of plain text files, create a document-term matrix. 
    Uses sklearns Count Vectorizer. Normalization applied is zscores. 
    Returns: sparse matrix as dataframe.  
    """
    print("\nNow creating DTM.")
    #== Create list of files
    textfiles = glob(textfolder) 
    textids = [os.path.basename(filename).split(".")[0] for filename in glob(textfolder)]
    #print(textids)
    print("number of texts", len(textids))
    print("number of authors", len(set([id.split("_")[1] for id in textids])))
    #== Define the vectorizer
    vectorizer = cv(
        input="filename",
        #strip_accents="unicode",
        #lowercase=True,
        #analyzer="word",
        #token_pattern=r"(?u)\b\w+\b", # at least 2 letters!
        #min_df=0.4,             # df = document frequency
        #max_df=0.6,             
        #max_features=1000,       # number of words
        )
    #== Apply the vectorizer
    dtm = vectorizer.fit_transform(textfiles)
    #== Inspect the resulting dtm
    #print(vectorizer.get_feature_names_out())
    #print("size of vocabulary", len(vectorizer.get_feature_names_out()))
    #print(vectorizer.vocabulary_)
    #== Transform to a pandas dataframe with meaningful labels
    dtm = pd.DataFrame(
        data = dtm.toarray(), 
        columns = vectorizer.get_feature_names_out(),
        index = textids)
    print("vectorized", dtm.head())
    print(dtm.shape)
    #= Perform the basic relative frequency transformation
    counts = np.sum(dtm, axis=1)
    #print(counts)
    dtm = dtm.divide(counts, axis=0)
    #print("relative", dtm.head())
    #= Select the words by highest overall (relative) frequency
    dtm = dtm.T
    dtm["totalfreqs"] = np.sum(dtm, axis=1)
    dtm.sort_values(by="totalfreqs", ascending=False, inplace=True)
    print("\ntotalfreqs\n", dtm.iloc[-5:,-5:])
    dtm = dtm[0:10]
    dtm = dtm.drop("totalfreqs", axis=1)
    dtm = dtm.T
    print("\nMFW\n", dtm.head())
    print(dtm.shape)
    #= Perform a normalization on the frequency data (here: z-scores)
    means = np.mean(dtm, axis=0) # means for each word across texts
    stdevs = np.std(dtm, axis=0) # standard deviation for each word across texts
    dtm = (dtm - means) / stdevs # = zscores (mean = 0, stdev = 1, per words)
    print("\nz-scores\n", dtm.head())
    print(dtm.shape)
    #print(textids)
    #== Return results
    return dtm, textids


def apply_birch(dtm, ids, n_clusters): 
    """
    Apply the BIRCH clustering algorithm to the dtm. 
    Inspect the results by checking which texts are assigned to which cluster. 
    https://scikit-learn.org/stable/modules/clustering.html#birch
    """
    print("\nNow applying BIRCH.")
    print("number of clusters", n_clusters)
    #= Define the clustering parameters and apply
    birch_model = Birch(
        threshold=0.8, 
        branching_factor=50, 
        n_clusters=n_clusters, 
        compute_labels=True, 
        copy=True)
    birch_model.fit_predict(dtm)
    #= Inspect the labels and cluster assignments
    #print(textids)
    #print(birch_model.labels_) # Cluster labels
    #print(len(birch_model.subcluster_centers_))
    assignments = dict(zip(ids, birch_model.labels_))
    #print(assignments)
    for cluster in range(0, n_clusters): 
        items = [k.split("_")[1] for k,v in assignments.items() if int(v) == cluster]
        print("cluster", cluster, ", size", len(items), ":", ", ". join(items)) 

    return birch_model, ids


def apply_kmeans(dtm, ids, n_clusters): 
    """
    Apply the KMeans clustering algorithm to the DTM. 
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
    """
    print("\nNow applying KMeans.")
    print("number of clusters", n_clusters)
    #== Define the clustering parameters and apply clustering
    kmeans_model = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=10, 
        max_iter=300, 
        tol=0.0001, 
        verbose=0, 
        random_state=None, 
        copy_x=True, 
        algorithm='lloyd'
        )
    kmeans_model.fit_predict(dtm)

    #== Inspect the labels and cluster assignments
    #print(kmeans_model.labels_) # Cluster labels
    #print(len(kmeans_model.cluster_centers_)) # number of clusters
    assignments = dict(zip(ids, kmeans_model.labels_))
    #print(assignments)
    for cluster in range(0, n_clusters): 
        #items = [k for k,v in assignments.items() if int(v) == cluster] # For bikedata
        items = [k.split("_")[1] for k,v in assignments.items() if int(v) == cluster] # For textdata
        print("cluster", cluster, ", size", len(items), ":", ", ". join(items))

    return kmeans_model, ids


def evaluate_clustering(model, ids):
    """
    Use some metrics to evaluate the clustering quality, 
    depending on the parameters
    https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
    """
    print("\nSome evaluation metrics")
    authors = [id.split("_")[1] for id in ids]
    #print(authors)
    #mappings = {"city" : 0, "trekking" : 1} # Mappings for bikedata
    mappings = {"Ward" : 0, "Wells" : 1, "Nesbit" : 2, "Trollope" : 3, "Broughton" : 4} # Mappings for text data
    labels_true = np.asarray([mappings[a] for a in authors])
    #print(labels_true)
    labels_pred = model.labels_
    #print(labels_pred)
    ## For purity, compute contingency matrix
    contingency_matrix = metrics.cluster.contingency_matrix(labels_true, labels_pred)
    purity = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 
    print("Purity", round(purity,3))
    rs = metrics.rand_score(labels_true, labels_pred)
    print("Rand score", round(rs,3))
    ars = metrics.adjusted_rand_score(labels_true, labels_pred)
    print("Adjusted Rand score", round(ars,3))
    amis = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
    print("Adjusted mutual information score", round(amis,3))


# === Main === 

def main(): 
    #= One option is to use the bikedata again (toy dataset)
    #bikedata, bikeids = load_data()
    #= Alternatively we can use a small real-world textual dataset
    textdata, textids = create_dtm(textfolder)
    #= Set number of clusters for the tests
    n_clusters = 5
    #= Apply and evaluate kMeans
    kmeans_model, ids = apply_kmeans(textdata, textids, n_clusters)
    evaluate_clustering(kmeans_model, ids)
    #= Alternatively, apply and evaluate BIRCH
    #birch_model, ids = apply_birch(textdata, textids, n_clusters)
    #evaluate_clustering(birch_model, ids)


main()