"""
Skript, das als Beispiel für Clustering als ML-Methode dient. 

Transformation eines Korpus von Textdateien in eine Term-Dokument-Matrix: 
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer

"""

# === Importe === 

# Allgemein
import os
import shutil
import re
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

# Files and folders 
workdir = join(os.path.realpath(os.path.dirname(__file__)))
textfolder = join(workdir, "data", "full", "*.txt")
segfolder = join(workdir, "data", "segs", "*.txt")


# === Funktionen === 


def split_texts(textfolder, seglen): 
    #print("\nNow splitting texts.")
    shutil.rmtree(join(workdir, "data", "segs", ""))
    os.mkdir(join(workdir, "data", "segs", ""))
    textfiles = glob(textfolder) 
    for tf in textfiles: 

        #= Get id and author from filename
        filename,ext = os.path.basename(tf).split(".")
        id,author = filename.split("_")

        #= Read text and determine number of segments
        with open(tf, "r", encoding="utf8") as infile: 
            text = infile.read() 
        text = re.split("\W+", text)
        lentext = len(text)
        numsegments = lentext // seglen
        #print(id, author, lentext, numsegments)

        #= Split the text into n segments, create filenames with counter, and save to folder
        for i in range(0,numsegments): 
            segment = " ".join(text[(i*seglen) : ((i+1)*seglen)])
            segfilename = id + "-" + "{:03d}".format(i) + "_" + author + ".txt"
            with open(join(workdir, "data", "segs", segfilename), "w", encoding="utf8") as outfile: 
                outfile.write(segment)




def create_dtm(textfolder):
    """
    From the corpus of plain text files, creates a document-term matrix. 
    Uses sklearns Count Vectorizer. 
    Transforms to relative frequencies. 
    Selects most frequent words based on relative frequency across corpus. 
    Applies z-score normalization. 
    Returns: sparse matrix as dataframe.  
    """
    #print("\nNow creating DTM.")
    #== Create list of files
    textfiles = glob(textfolder) 
    textids = [os.path.basename(filename).split(".")[0] for filename in glob(textfolder)]
    #print("number of texts", len(textids))
    #print("number of authors", len(set([id.split("_")[1] for id in textids])))
    #== Define and apply the vectorizer
    vectorizer = cv(
        input="filename",
        strip_accents="unicode",
        lowercase=True,
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b", # at least 1 letters!
        # Removing the following options for better rel freq calculation
        #min_df=0.4,             # df = document frequency
        #max_df=0.6,             
        #max_features=1000,       # number of words
        )
    dtm = vectorizer.fit_transform(textfiles)

    #== Transform to a pandas dataframe with meaningful labels
    dtm = pd.DataFrame(
        data = dtm.toarray(), 
        columns = vectorizer.get_feature_names_out(),
        index = textids)
    #print("\nvectorized / absolute\n", dtm.iloc[:5,-5:])
    #print(dtm.shape)

    #= Perform the basic relative frequency transformation
    if freqtype == "relative": 
        counts = np.sum(dtm, axis=1)
        dtm = dtm.divide(counts, axis=0)
        #print("relative", dtm.iloc[:5,:5])
    elif freqtype == "absolute": 
        pass

    #= Select the words by highest overall (relative) frequency
    dtm = dtm.T
    dtm["totalfreqs"] = np.sum(dtm, axis=1)
    dtm.sort_values(by="totalfreqs", ascending=False, inplace=True)
    #print("\ntotalfreqs\n", dtm.iloc[:5,-5:])
    dtm = dtm[0:mfw] # this could be an important parameter
    dtm = dtm.drop("totalfreqs", axis=1)
    dtm = dtm.T
    #print("\nMFW\n", dtm.iloc[:5,:5])
    #print(dtm.shape)

    #= Perform a normalization on the frequency data (here: z-scores)
    if normalization == True: 
        means = np.mean(dtm, axis=0) # means for each word across texts
        stdevs = np.std(dtm, axis=0) # standard deviation for each word across texts
        dtm = (dtm - means) / stdevs # = zscores (mean = 0, stdev = 1, per words)
        #print("\nz-scores\n", dtm.iloc[:5,:5])
        #print(dtm.shape)
    elif normalization == False: 
        pass

    return dtm, textids


def apply_birch(dtm, ids, n_clusters): 
    """
    Apply the BIRCH clustering algorithm to the dtm. 
    Inspect the results by checking which texts are assigned to which cluster. 
    https://scikit-learn.org/stable/modules/clustering.html#birch
    """
    #print("\nNow applying BIRCH.")
    #print("number of clusters", n_clusters)
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
        #print("cluster", cluster, ", size", len(items), ":", ", ". join(items)) 

    return birch_model, ids


def apply_kmeans(dtm, ids, n_clusters): 
    """
    Apply the KMeans clustering algorithm to the DTM. 
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
    """
    #print("\nNow applying KMeans.")
    #print("number of clusters", n_clusters)
    #== Define the clustering parameters and apply clustering
    kmeans_model = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=10, 
        max_iter=10, 
        tol=0.0001, 
        verbose=0, 
        #random_state=42, 
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
        #print("cluster", cluster, ", size", len(items), ":", ", ". join(items))

    return kmeans_model, ids


def evaluate_clustering(model, ids):
    """
    Use some metrics to evaluate the clustering quality, 
    depending on the parameters
    https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
    """
    #print("\nSome evaluation metrics")
    authors = [id.split("_")[1] for id in ids]
    mappings = {"Ward" : 0, "Wells" : 1, "Nesbit" : 2, "Trollope" : 3, "Broughton" : 4} 
    labels_true = np.asarray([mappings[a] for a in authors])
    labels_pred = model.labels_

    #= For purity, compute contingency matrix
    contingency_matrix = metrics.cluster.contingency_matrix(labels_true, labels_pred)
    purity = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 
    #print("Purity", round(purity,3))

    #= Rand score
    rs = metrics.rand_score(labels_true, labels_pred)
    #print("Rand score", round(rs,3))

    #= Adjusted Rand Index
    #= https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
    ars = metrics.adjusted_rand_score(labels_true, labels_pred)
    #print("Adjusted Rand score", round(ars,3))

    #= Adjusted mutual information
    amis = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
    #print("Adjusted mutual information score", round(amis,3))

    #= Display and save summary results
    #print("input:" + str(inputcall) + ", seglen:" + str(seglen) + ", numtexts:" + str(len(ids)) + ", mfw:" + str(mfw) + ", freqtype:" + str(freqtype) + ", normalization:" + str(normalization) +  ", ARI:" + str(round(amis,3)))
    label = str(inputcall) + "-" + str(seglen) + "-" + str(len(ids)) + "-" + str(mfw) + "-" + str(freqtype) + "-" + str(normalization)
    print(label, str(inputcall), str(seglen), str(len(ids)), str(mfw), str(freqtype), str(normalization), str(round(amis,3)))
    return [label, inputcall, seglen, len(ids), mfw, freqtype, normalization, round(amis,4)]



# === Main === 
print("label", "input", "seglen", "numtexts", "mfw", "freqtype", "normalization", "ARI")
allresults = []
def main(seglen): 
    #= Prepare and load the text data
    split_texts(textfolder, seglen)
    textdata, textids = create_dtm(input)
    #= Set number of clusters for the tests
    #= Apply and evaluate kMeans
    kmeans_model, ids = apply_kmeans(textdata, textids, n_clusters)
    allresults.append(evaluate_clustering(kmeans_model, ids))

# Loop for various parameters
for inputcall in ["full", "segs"]: 
    if inputcall == "segs": 
        input = segfolder
        for seglen in [5000,2000]: 
            for mfw in [10,100,1000,2000]:
                for freqtype in ["relative", "absolute"]:
                    for normalization in [True, False]: 
                        n_clusters = 5
                        main(seglen)
                        main(seglen)
                        main(seglen)
                        main(seglen)
                        main(seglen)
                        main(seglen)
                        main(seglen)
                        main(seglen)
                        main(seglen)
                        main(seglen)
    if inputcall == "full":
        input = textfolder
        seglen = 99999
        for mfw in [10,100,1000,2000]:
            for freqtype in ["relative", "absolute"]:
                    for normalization in [True, False]: 
                        n_clusters = 5
                        main(seglen)
                        main(seglen)
                        main(seglen)
                        main(seglen)
                        main(seglen)
                        main(seglen)
                        main(seglen)
                        main(seglen)
                        main(seglen)
                        main(seglen)

def save_results(): 
    columns = ["label", "input", "seglen", "ids", "mfw", "freqtype", "normalization", "ARI"]
    results = pd.DataFrame(allresults, columns=columns)
    print(results.head())
    with open(join(workdir, "allresults.csv"), "w", encoding="utf8") as outfile: 
        results.to_csv(outfile)
    return results
results = save_results()


def visualize_results(): 
    import seaborn as sns
    from matplotlib import pyplot as plt
    with open(join(workdir, "allresults.csv"), "r", encoding="utf8") as infile: 
        results = pd.read_csv(infile)
    results = results[["label", "ARI"]]
    #ordered_index = results.groupby(by="label").median().sort_values(by="ARI", ascending=False).index
    #print(results.shape)
    #print(len(ordered_index))
    results.sort_values(by="ARI", ascending=False, inplace=True)
    fig,ax = plt.subplots(figsize=(16, 12))
    ax = sns.boxplot(data=results, x="label", y="ARI", palette="light:grey", showfliers=False)
    ax.set_title('Grid search for clustering parameters')
    ax.set_ylabel('ARI')
    ax.set_xlabel('Parameter combination')
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.grid()
    sns.set_style("whitegrid")
    filename = join(workdir, "allresults.svg")
    plt.savefig(filename, dpi=300)
visualize_results()