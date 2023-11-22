"""
Script for logistic regression on topic data regarding French Drama. 

Source of the data: 
- https://github.com/dh-trier/datasets

Background paper
- https://www.digitalhumanities.org/dhq/vol/11/2/000291/000291.html

Using sklearn
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression

"""


# === Importe === 

# general
import pandas as pd 
import numpy as np
from os.path import join
import os

# specific
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression as logreg



# === Parameter === 

wd = join(os.path.realpath(os.path.dirname(__file__)))
datafile = join("/", "media", "christof", "Data", "Github", "dh-trier", "datasets", "tabular", "topics", "French-Drama_Topics-and-Genres.csv")


# === Funktionen === 

def load_dataset(datafile): 

    # Open as DataFrame
    with open(datafile, "r", encoding="utf8") as infile: 
        data = pd.read_csv(infile, sep=",", index_col=0)

    # Inspect data
    print("shape", data.shape)
    #print("columns", data.columns)
    #print(data.head())

    # Return
    return data



def prepare_dataset(data): 
      
    # Remove unnecessary columns (features / metadata)
    data = data.drop([
        'segmentID', 
        'idno', 
        'fievre_idno', 
        'author-short', 
        'author-full',
        'title-full', 
        'year-premiere', 
        'year-print', 
        'year_reference',
        'form', 
        'inspiration', 
        'structure',
        'binID',
        ], axis=1)
   
    # Remove unnecessary rows (= plays): focus on tragedy and comedy
    #print(list(set(data["subgenre"])))
    include = ["Tragédie", "Comédie"]
    data = data.query("subgenre in @include")
    
    # Recode subgenre in a binary way: tragedy 1, comedy 0
    data["tragedy"] = pd.get_dummies(data["subgenre"])["Tragédie"]
    data = data.drop(["subgenre"], axis=1)
    
    # Check result
    print("shape", data.shape)
    #print("columns", data.columns)
    
    # Return
    return data



def perform_logreg(data): 
    
    # Prepare data: array of features, array of target values
    featurelabels = np.array(data.columns[:-1])
    X = np.array(data.iloc[:,:])   
    y = np.array(data["tragedy"])
    
    # Fit the regression (using the sklearn method)
    reg = logreg().fit(X, y)
    
    # Get the score (R²: 1 is perfect, 0 is bad)
    print("\nscore", reg.score(X,y))
    
    # Get the results
    #print("\nfeaturelabels", featurelabels)
    print("coefficients", reg.coef_)
    print("intercept", reg.intercept_)
    
    # Make results more readable
    coeffs = np.round(reg.coef_[0], 3)
    results = dict(zip(featurelabels, coeffs))
    results = pd.DataFrame.from_dict(results, orient="index", columns=["coeff"])
    results = results.sort_values(by="coeff", ascending=False)
    print("\npositive (tragedy):\n", results.head(3), "\n\nnegative (comedy):\n", results.tail(3))
    
    # Save coefficients to disk
    with open(join(wd, "Topics-and-subgenre_coefficients.csv"), "w", encoding="utf8") as outfile: 
        results.to_csv(outfile, sep=";")
    return results


def visualize_coeffs(results):
    
    # Some settings    
    sns.set_theme(style="whitegrid")
    sns.set(font_scale=0.5)

    # Select the most relevant topics (extreme scores)
    top = results.iloc[:10]
    bottom = results.iloc[-10:]
    selection = pd.concat([top, bottom])

    # Define barplot
    sns.barplot(
        data = selection["coeff"],
        orient = "h",
    ).set(title="Topic coefficients (positive = tragedy, negative = comedy)")

    # Save to disk and close
    plt.tight_layout()
    plt.savefig(join(wd, "Topics-by-coefficient.png"), dpi=300)
    plt.close()
    


def visualize(data):  
    
    # Define scatterplot
    sns.scatterplot(
        data=data,
        x="39_sang-mort-main", 
        y="36_bon-monsieur-beau",
        hue = "tragedy",
        style = "tragedy",
        palette = "Set1",
        markers = ["o", "s"],
        alpha = 0.4,
        sizes = [0.5,0.5],
        )
    
    # Optionally, use a log scale
    #plt.xscale('log')
    #plt.yscale('log')    

    # Save to disk and close
    plt.savefig(join(wd, "Topics-and-subgenre_39+36.png"), dpi=300)
    plt.close()



# === Main === 

def main(): 
    data = load_dataset(datafile)
    data = prepare_dataset(data)
    results = perform_logreg(data)
    visualize_coeffs(results)
    visualize(data)
main()
