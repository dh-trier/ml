"""
Skript zur Illustration von linearer Regression mit historischen Wetterdaten. 

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
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn import metrics


# === Parameters / global variables ===

workdir = join(os.path.realpath(os.path.dirname(__file__)))
#print(workdir)


# === Functions === 


def load_dataset(datafile): 
    """
    Load the Kaggle dataset "Weather in WW2" from disk.
    https://www.kaggle.com/datasets/smid80/weatherww2
    Returns: Dataframe. 
    """
    with open(datafile, "r", encoding="utf8") as infile: 
        data = pd.read_csv(infile)
        return data


def get_overview(data):
    """
    Get a first idea of what the file contains, using .shape, .head(), columns
    Does not return anything, just prints a few things. 
    """
    # What is the shape of the dataset? 
    print("\nshape\n", data.shape)
    # See the start of the dataframe using .head()
    print("\nhead\n", data.head())
    # What columns are there in the dataset?
    print("\ncolumns\n", data.columns)


def inspect_variable(data): 
    """
    Look at the data using histograms
    https://seaborn.pydata.org/generated/seaborn.histplot.html
    We can create a histogram for one variable, 
    or for several variables together 
    (useful if they are related and have a similar shape)
    """
    sns.set_style("whitegrid")
    #== The precipitation column
    # There are non-numerical values in the column, which we delete.
    # We need to set a relatively low number of bins to see something.
    # We also set a log-axis for y, because of quick drop-off in values.
    plt.figure()
    precip = pd.to_numeric(data["Precip"], errors='coerce')   
    sns.histplot(data=data, x=precip, bins=100, color="darkgreen") 
    plt.xlim([0, 300])
    plt.yscale('log')
    plt.savefig(join(workdir, "variable-precip.png"), dpi=300)
    #== One temperature variable
    # Just one variable is easy and seaborn will figure out good settings.
    # We use the KDE - kernel density estimation to fit a curve over the histogram.
    # Explanation: https://scikit-learn.org/stable/modules/density.html
    plt.figure()
    sns.histplot(data=data, x=data["MeanTemp"], kde=True, color="darkred") 
    plt.savefig(join(workdir, "variable-onetemp.png"), dpi=300)
    #== Two/three of the temperature variables
    plt.figure()
    sns.histplot(data=data, x=data["MinTemp"], kde=True, color="darkblue") 
    sns.histplot(data=data, x=data["MaxTemp"], kde=True, color="darkgreen") 
    sns.histplot(data=data, x=data["MeanTemp"], kde=True, color="darkred") 
    plt.savefig(join(workdir, "variable-moretemps.png"), dpi=300)
    

def select_data(data): 
    """
    Prepare the data by selecting relevant columns. 
    """
    # Select those columns that are of interest to us
    # (Alternatively, we could delete irrelevant columns. But this is easier.)
    data = data.loc[:,["MinTemp", "MaxTemp"]]
    return data


def inspect_data(data, scatterplotfile, heatmapfile):
    """
    Visual inspection of the data using .describe and a scatterplot. 
    https://seaborn.pydata.org/generated/seaborn.scatterplot.html
    Returns: File to disk. 
    """
    # See some summary data using .describe()
    print("\ndescribe\n", data.describe())
    # Visual inspection using a scatterplot
    # This time, we are already inspecting the relationship between the two variables.
    plt.figure()
    sns.scatterplot(data=data, x="MinTemp", y="MaxTemp") 
    plt.savefig(scatterplotfile, dpi=300)
    # Visual inspection of correlations in the dataset
    # The heatmap shows how strongly the two variables correlate. 
    # This gets more interesting when we have more variables. 
    sns.heatmap(data.corr(), annot=True, cmap="GnBu", vmin=0.75, vmax=1)
    plt.xticks()
    plt.savefig(heatmapfile, dpi=300)


def clean_data(data): 
    """
    Now we preprocess the data. 
    In this case, mostly removing erroneous values.
    Returns: DataFrame, reduced size. 
    """
    # Look for values where there is a problem 
    # Here: MinTemp higher than MaxTemp (difference should not be negative)
    data["DiffTemp"] = data["MaxTemp"] - data["MinTemp"]
    print("\ndescribe DiffTemp\n", data["DiffTemp"].describe())
    # Remove the faulty values, then remove the DiffTemp column again
    data = data.drop(data[data["DiffTemp"] < 0].index)
    print("\ndescribe DiffTemp\n", data["DiffTemp"].describe())
    data = data.drop("DiffTemp", axis=1)
    print("\nclean data\n", data.head())
    return data


def linreg_sklearn(data): 
    """
    Perform linear regression using sciki-learn
    https://scikit-learn.org/stable/modules/linear_model.html
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    # Initialise linear regression object
    # Wir verwenden hier die "normale" lineare Regression ("ordinary least squares").
    # Wir könnten aber auch "ml.Ridge" einsetzen und also Ridge regression verwenden
    model = lm.LinearRegression()
    # Split dataset in train and test set
    # Wir lernen hier die Beziehung zwischen einer und einer anderen Variable.
    # Möglich ist auch, die Beziehung zwischen mehreren Variablen und einer anderen Variablen zu lernen.
    X_train, X_test, y_train, y_test = tts(data[["MinTemp"]], data["MaxTemp"], test_size=0.1)
    # Fit (or "train") the regression on the training data
    model.fit(X_train, y_train)
    # Make predictions using test set and model
    y_pred = model.predict(X_test)

    # Get the model parameters
    # Model coefficient(s)
    print("\nCoefficient(s) (slope): \n", model.coef_)
    # The intercept
    print("\nIntercept (crossing point): \n", model.intercept_)
    return X_train, X_test, y_train, y_test, y_pred, model


def inspect_model(X_train, X_test, y_train, y_test, y_pred, model, modelfile): 
    """
    Find our more about our model than just the key parameters (slope and intercept.)
    We do this by plotting the model, plotting a prediction and calculating model metrics. 
    """
    #== Plot the results
    plt.figure()
    sns.set_style("whitegrid")
    fig,ax = plt.subplots()
    ax = sns.scatterplot(x=X_test["MinTemp"], y=y_test, color="darkgreen")
    ax = plt.plot(X_test["MinTemp"], y_pred, color="darkblue", linewidth=3)
    plt.savefig(modelfile, dpi=300)

    # We can even add a single predicted datapoint; it will always be on the line!
    # A linear regression line has an equation of the form y = a + bX
    input = 10  # The explanatory variable: MinTemp
    output = model.intercept_ + (model.coef_ * input)
    print("Predicted value for input", input, ":", output[0])
    ax = sns.scatterplot(x=input, y=output, color="red", marker="X", s=150)
    plt.savefig(modelfile, dpi=300)

    #== Some key indicators for the model quality
    # Mean squared error between true values and predicted values
    print("\nMean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    # The coefficient of determination (r squared), 1 is perfect. 
    print("R squared: %.2f" % r2_score(y_test, y_pred))
    # Explained variance score
    print("Explained variance:", explained_variance_score(y_test, y_pred))


def inspect_residuals(y_test, y_pred): 
    """
    As an additional way to inspect the model, we can plot the residuals. 
    How far is each datapoint from the regression line? 
    The regression line is assumed to be horizontal. 
    A good model will have a rather random distribution of points. 
    If there is still a pattern, than this regularity is missing in the model.
    """
    # Calculate the residuals: for each point, the difference between model and real value.
    residuals = y_test - y_pred
    # Plot this as a scatterplot. 
    plt.figure()
    sns.scatterplot(x=residuals, y=y_pred)
    plt.savefig(join(workdir, "residuals.png"), dpi=300)



# === Main ===

def main(): 
    # Local variables
    datafile = join(workdir, "weather-ww2.csv")
    scatterplotfile = join(workdir, "weather-ww2_scatter.png")
    heatmapfile = join(workdir, "weather-ww2_heatmap.png")
    modelfile = join(workdir, "weather-ww2_model.png")
    # Getting to know the data
    data = load_dataset(datafile)
    get_overview(data)
    inspect_variable(data)
    # Pipeline
    data = select_data(data)
    data = clean_data(data)
    inspect_data(data, scatterplotfile, heatmapfile)
    X_train, X_test, y_train, y_test, y_pred, model = linreg_sklearn(data)
    # Check the model
    inspect_model(X_train, X_test, y_train, y_test, y_pred, model, modelfile)
    inspect_residuals(y_test, y_pred)

main()