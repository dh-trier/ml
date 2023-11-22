"""
Skript zur Illustration von linearer Regression mit einfachen, künstlichen Daten. 

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
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns



# === Parameters / global variables ===

workdir = join(os.path.realpath(os.path.dirname(__file__)))
print(workdir)


# === Functions === 


def create_dataset(): 
    """
    Create some simple, artificial data. 
    We assume the data concerns apple trees: age (years) and yield (kilograms). 
    Returns: Dataframe. 
    """
    datapoints = [
        (1,0.5),
        (2,5),
        (2,8),
        (3,6),
        (4,9),
        (5,8),
        (5,10.5),
        (6,12),
        (7,16),
        (7,14),
        (8,17),
        (8,15),
        (9,16),
        (10,15),
    ]
    columns = ["age", "yield"]
    data = pd.DataFrame(datapoints, columns=columns)
    #print(data)
    return data 


def get_overview(data):
    """
    Get an overview of the data: min, max, mean for each variable.
    """
    print("Data properties for age / x (min, mean, median, max, std):", 
        np.min(data["age"]),
        np.round(np.mean(data["age"])),
        np.median(data["age"]),
        np.max(data["age"]),
        np.round(np.std(data["age"]),2),
        )
    print("Data properties for yield / y (min, mean, median, max, std):", 
        np.min(data["yield"]),
        np.round(np.mean(data["yield"])),
        np.median(data["yield"]),
        np.max(data["yield"]),
        np.round(np.std(data["yield"]),2),
        )

    from scipy import stats
    print(stats.describe(data["age"]))


def plot_data(data): 
    """
    Look at the data using a scatterplot. 
    https://seaborn.pydata.org/generated/seaborn.scatterplot.html
    """
    sns.set_style("whitegrid")
    plt.figure()
    plt.xlim(0, 12)
    plt.ylim(0, 20)
    plt.title("Apple tree yields in kg as a function of age in years")
    sns.scatterplot(
        data=data, 
        x="age", 
        y="yield", 
        color="darkred"
        )
    plt.savefig(join(workdir, "linregsteps_data.png"), dpi=300)
    


def calculate_linreg(data): 
    """
    Perform linear regression manually. 
    """

    # Determine the mean of x and y-axis. 
    xmean = np.mean(data["age"]) 
    ymean = np.mean(data["yield"]) 

    # Determine the sums of the squared deviations: (x - x_mean)^2
    # and the squared sums of the deviations: (x - xmean) x (y - ymean)
    sum_xxdevs = 0
    sum_xydevs = 0
    for row in data.iterrows(): 
        x = row[1]["age"]
        y = row[1]["yield"]

        xdev = x - xmean
        ydev = y - ymean

        sum_xxdevs += xdev * xdev
        sum_xydevs += xdev * ydev

    print("Sum of squared x-deviations:", sum_xxdevs)
    print("Sum of x-deviations x y-deviations:", sum_xydevs)

    # Determine the slope and the intercept
    slope = sum_xydevs / sum_xxdevs
    intercept = ymean - (slope * xmean)
    print("\nSlope and intercept:", np.round(slope, 2), np.round(intercept, 2))

    return slope, intercept 


def plot_linreg(data, slope, intercept): 
    """
    Define the regression line using slope and intercept. 
    Plot the data and the linear regression line. 
    """

    # Define the endpoints of the regression line
    # Formula: y = intercept + (x * slope) 
    
    # Starting point: x = 0
    x = 0 
    y = intercept + (x * slope)
    startpoint = [x,y]
    print(startpoint)

    # End point: x = np.max(age) = 10
    x = 11
    y = intercept + (x * slope)
    endpoint = [x,y]
    print(endpoint)

    # Plot the results: double-layered plot
    sns.set_style("whitegrid")
    plt.figure()
    fig,ax = plt.subplots()
    plt.figure()
    plt.xlim(0, 12)
    plt.ylim(0, 20)
    plt.title("Apple tree age and yields, with regression line")

    # Layer with the scatteplot
    ax = sns.scatterplot(
        data=data, 
        x="age", 
        y="yield", 
        color="darkred"
        )

    # Layer with the regression line
    ax = plt.plot([startpoint[0], endpoint[0]], [startpoint[1], endpoint[1]])      

    # Save the file to disk
    plt.savefig(join(workdir, "linregsteps_regression.png"), dpi=300)


# === Main ===

def main(): 
    data = create_dataset()
    get_overview(data)
    plot_data(data)
    slope, intercept = calculate_linreg(data)
    plot_linreg(data, slope, intercept)

main()
