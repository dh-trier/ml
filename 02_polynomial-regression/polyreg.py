"""
Script to analyse and visualize the average sentence length per novel over time,
using tests for statistical difference and regression line plots. 

Two datasets are supplied here, with average sentence length data per novel, for English
novels from ELTeC and from Gutenberg Project. 

Using: https://seaborn.pydata.org/tutorial/regression.html. 

For some background, see: https://github.com/christofs/sentence-length
and: https://dragonfly.hypotheses.org/1152. 

"""

# === Imports

import re
import glob
import os
from os.path import join

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# === Global variables

workdir = join(os.path.realpath(os.path.dirname(__file__)))
gutenberg = {"label" : "gutenberg", "path" : join(workdir, "Gutenberg3_sentence-lengths.csv")}
eltec ={"label" : "eltec", "path" : join(workdir, "ELTeC-eng_sentence-lengths.csv")}


# === Functions

def read_data(dataset): 
    with open(dataset, "r", encoding="utf8") as infile: 
        data = pd.read_csv(infile, sep=";", index_col=0)
    #print(data.head())
    return data


def plot_seaborn(data, order, label): 
    plt.figure()
    regplot = sns.regplot(
        x="year", 
        y="avgsentlen", 
        marker=".", 
        data=data, 
        x_jitter=0.3, 
        order=order, 
        color="#117b99", 
        scatter_kws={"color": "#117b99"}, 
        line_kws={"color": "#00264D"}
        )
    fig = regplot.get_figure()
    plt.ylim(0, 60)
    plt.grid()
    # Filename that includes the dataset and the order of the polynomial
    regplotfile = join(workdir, label + "_avgsentlens+regression-" + str(order) +"order.png")
    fig.savefig(regplotfile, dpi=600)
    

# === Main

def main(): 
    """
    Coordinates the process. 
    """
    for dataset in [gutenberg, eltec]: 
        data = read_data(dataset["path"])
        for order  in range(1,7): 
            plot_seaborn(data, order, dataset["label"])
      
main()
