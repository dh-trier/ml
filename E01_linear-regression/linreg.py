"""
"Linear regression attempts to model the relationship between two variables 
by fitting a linear equation to observed data. One variable is considered to 
be an explanatory variable, and the other is considered to be a dependent variable."
http://www.stat.yale.edu/Courses/1997-98/101/linreg.htm 
"""

# === Imports ===

# General
import os
from os.path import join

# Data
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# ML 
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics


# === Parameters / global variables ===

workdir = join(os.path.realpath(os.path.dirname(__file__)))
#print(workdir)


# === Functions === 


def load_dataset(datafile): 
    """
    Load the "Weather in WW2" dataset from Kaggle. 
    https://www.kaggle.com/datasets/smid80/weatherww2
    """
    with open(datafile, "r", encoding="utf8") as infile: 
        data = pd.read_csv(infile)
        return data


def get_overview(data):
    """
    Get an idea of what the file contains, using .shape, .head(), columns
    """
    # What is the shape of the dataset? 
    print(data.shape)
    # See the start of the dataframe using .head()
    print(data.head())
    # What columns are there in the dataset?
    print(data.columns)


def prepare_data(data): 
    """
    Prepare the data.
    """
    # Select those columns that are of interest to us
    data = data.loc[:,["MinTemp", "MaxTemp", "MeanTemp"]]
    return data


def inspect_data(data, scatterplotfile, heatmapfile):
    """
    Visual inspection of the data using a scatteplot. 
    https://seaborn.pydata.org/generated/seaborn.scatterplot.html
    """
    # See some summary data using .describe()
    print(data.describe())
    # Visual inspection using a scatterplot
    sns.scatterplot(data=data, x="MinTemp", y="MeanTemp") # Min, Max, Mean variieren! 
    plt.savefig(scatterplotfile, dpi=300)
    # Visual inspection of correlations in the dataset
    sns.heatmap(data.corr(), annot=True, cmap="GnBu", vmin=0.75, vmax=1)
    plt.xticks()
    plt.savefig(heatmapfile, dpi=300)


def clean_data(data): 
    # Look for values where there is a problem (MinTemp higher than MaxTemp)
    data["DiffTemp"] = data["MaxTemp"] - data["MinTemp"]
    print(data["DiffTemp"].describe())
    # Remove the faulty values, then remove the DiffTemp column again
    data = data.drop(data[data["DiffTemp"] < 0].index)
    print(data["DiffTemp"].describe())
    data = data.drop("DiffTemp", axis=1)
    return data


def linreg_sklearn(data, modelfile): 
    """
    Perform linear regression using sciki-learn
    https://scikit-learn.org/stable/modules/linear_model.html
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    # Initialise linear regression object
    model = lm.LinearRegression()
    # Split dataset in train and test set
    # Wir können hier entweder die Beziehung zwischen einer und einer anderen Variable lernen
    # Oder aber die Beziehung zwischen zwei Variable und einer anderen Variablen lernen
    X_train, X_test, y_train, y_test = tts(data[["MinTemp"]], data["MaxTemp"], test_size=0.10)
    # Fit (or "train") the regression on the training data
    model.fit(X_train, y_train)
    # Make predictions using test set and model
    y_pred = model.predict(X_test)

    # Inspect the model by extracting some key indicators
    # Model coefficients
    print("Coefficients: \n", model.coef_)
    # The intercept
    print("Intercept: \n", model.intercept_)
    # Mean squared error between true values and predicted values
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

    # Plot the results
    sns.set_style("whitegrid")
    fig,ax = plt.subplots()
    ax = sns.scatterplot(x=X_test["MinTemp"], y=y_test, color="darkgreen")
    ax = plt.plot(X_test["MinTemp"], y_pred, color="darkblue", linewidth=3)
    plt.savefig(modelfile, dpi=300)

    # We can even add a single predicted datapoint; it will always be on the line!
    # A linear regression line has an equation of the form Y = a + bX
    input = 0  # The explanatory variable: MinTemp
    output = model.intercept_ + (model.coef_ * input)
    print("Predicted value for input:", input, output[0])
    ax = sns.scatterplot(x=input, y=output, color="red", marker="X", s=150)
    plt.savefig(modelfile, dpi=300)


# === Main ===

def main(): 
    # Local variables
    datafile = join(workdir, "weather-ww2.csv")
    scatterplotfile = join(workdir, "weather-ww2_scatter.png")
    heatmapfile = join(workdir, "weather-ww2_heatmap.png")
    modelfile = join(workdir, "weather-ww2_model.png")
    # Pipeline
    data = load_dataset(datafile)
    get_overview(data)
    data = prepare_data(data)
    data = clean_data(data)
    inspect_data(data, scatterplotfile, heatmapfile)
    linreg_sklearn(data, modelfile)

main()