"""
Skript for testing out the examples from the following seaborn tutorial on regression: 
https://seaborn.pydata.org/tutorial/regression.html
The tutorial uses some datasets built into seaborn. 
"""

# === Imports ===

from matplotlib import pyplot as plt
import seaborn as sns
import os
from os.path import join
import numpy as np

# === Global variables

wd = join(os.path.realpath(os.path.dirname(__file__)))


# === # Datasets

tips = sns.load_dataset("tips")

anscombe = sns.load_dataset("anscombe")





# === (1) Functions for drawing linear regression models === 


def part_one(): 

    # Reglplot
    sns.regplot(data=tips, x="total_bill", y="tip")
    plt.savefig(join(wd, "1-tips_regplot.png"), dpi=300)
    plt.close()

    # Lmplot
    sns.lmplot(x="total_bill", y="tip", data=tips)
    plt.savefig(join(wd, "1-tips_lmplot.png"), dpi=300)
    plt.close()

    # scatterplot
    sns.lmplot(data=tips, x="size", y="tip")
    plt.savefig(join(wd, "1-tips_size-tip.png"), dpi=300)
    plt.close()


    # scatterplot with jitter 
    sns.lmplot(data=tips, x="size", y="tip", x_jitter=.2)
    plt.savefig(join(wd, "1-tips_size-with-jitter.png"), dpi=300)
    plt.close()

    # estimator with confidence interval
    sns.lmplot(data=tips, x="size", y="tip", x_estimator=np.mean)
    plt.savefig(join(wd, "1-tips_size-with-estimator.png"), dpi=300)
    plt.close()

#part_one()

# === (2) Fitting different kinds of models ===  


def part_two(): 

    # Part 1 
    sns.lmplot(
        data=anscombe.query("dataset == 'I'"),
        x="x", 
        y="y",
        ci=None, 
        scatter_kws={"s": 80},
        )
    plt.savefig(join(wd, "2-anscombe_part1.png"), dpi=300)
    plt.close()


    # Part 2
    sns.lmplot(
        data=anscombe.query("dataset == 'II'"),
        x="x", 
        y="y",
        ci=None, 
        scatter_kws={"s": 80},
        )
    plt.savefig(join(wd, "2-anscombe_part2.png"), dpi=300)
    plt.close()


    # Second-order polynomial
    sns.lmplot(
        data=anscombe.query("dataset == 'II'"),
        x="x", 
        y="y",
        ci=None, 
        scatter_kws={"s": 80},
        order = 2,
        )
    plt.savefig(join(wd, "2-anscombe_second-order.png"), dpi=300)
    plt.close()


    # Outliers
    sns.lmplot(
        data=anscombe.query("dataset == 'III'"),
        x="x", 
        y="y",
        ci=None, 
        scatter_kws={"s": 80},
        order = 1,
        )
    plt.savefig(join(wd, "2-anscombe_outliers.png"), dpi=300)
    plt.close()


    # Outliers
    sns.lmplot(
        data=anscombe.query("dataset == 'III'"),
        x="x", 
        y="y",
        ci=None, 
        scatter_kws={"s": 80},
        order = 1,
        robust=True,
        )
    plt.savefig(join(wd, "2-anscombe_outliers-robust.png"), dpi=300)
    plt.close()


    # Tips binary 
    tips["big_tip"] = (tips.tip / tips.total_bill) > .15
    sns.lmplot(
        data=tips,
        x="total_bill", 
        y="big_tip", 
        y_jitter=.03
        )
    plt.savefig(join(wd, "2-tips_binary.png"), dpi=300)
    plt.close()


    # Tips binary: logistic
    tips["big_tip"] = (tips.tip / tips.total_bill) > .15
    sns.lmplot(
        data=tips,
        x="total_bill", 
        y="big_tip", 
        y_jitter=.03,
        logistic=True,
        )
    plt.savefig(join(wd, "2-tips_logistic.png"), dpi=300)
    plt.close()


    # Tips lowess
    sns.lmplot(
        x="total_bill",
        y="tip",
        data=tips,
        lowess=True, 
        line_kws={"color": "C1"}
        )
    plt.savefig(join(wd, "2-tips_lowess.png"), dpi=300)
    plt.close()



    # Residuals: linear
    sns.residplot(
        x="x", 
        y="y", 
        data=anscombe.query("dataset == 'I'"),
        scatter_kws={"s": 80}
        )
    plt.savefig(join(wd, "2-anscombe_residual-I.png"), dpi=300)
    plt.close()

    # Residual:  second order with first-order
    sns.residplot(
        x="x", 
        y="y", 
        data=anscombe.query("dataset == 'II'"),
        scatter_kws={"s": 80}
        )  
    plt.savefig(join(wd, "2-anscombe_residual-II.png"), dpi=300)
    plt.close()
        
#part_two()    



# === (3) Conditioning on other variables === 

def part_three(): 

    sns.lmplot(
        data=tips,
        x="total_bill", 
        y="tip", 
        hue="smoker", 
        )
    plt.savefig(join(wd, "3-tips_hue.png"), dpi=300)
    plt.close()



    sns.lmplot(
        data=tips,
        x="total_bill", 
        y="tip", 
        hue="smoker", 
        markers=["o", "x"],
        )
    plt.savefig(join(wd, "3-tips_hue+markers.png"), dpi=300)
    plt.close()


    sns.lmplot(
        data=tips,
        x="total_bill", 
        y="tip", 
        hue="smoker", 
        col="time",
        )
    plt.savefig(join(wd, "3-tips_hue+time.png"), dpi=300)
    plt.close()


    sns.lmplot(
        data=tips,
        x="total_bill", 
        y="tip", 
        hue="smoker", 
        col="time",
        row="sex",
        )
    plt.savefig(join(wd, "3-tips_hue+time+sex.png"), dpi=300)
    plt.close()

#part_three()


# === (4) Plotting a regression in other contexts ===

def part_four(): 
    # Jointplot
    sns.jointplot(
        x="total_bill", 
        y="tip", 
        data=tips, 
        kind="reg"
        )
    plt.savefig(join(wd, "4-tips_jointplot.png"), dpi=300)
    plt.close()


    # Pairplot

    sns.pairplot(
        tips, 
        x_vars=["total_bill", "size"], 
        y_vars=["tip"],
        hue="smoker", 
        height=5, 
        aspect=.8, 
        kind="reg")            
    plt.savefig(join(wd, "4-tips_pairplot.png"), dpi=300)
    plt.close()

#part_four()




# === (5) Transfer to an actual DH dataset that we need to load ourselves ===  

import pandas as pd

def load_data(): 
    with open(join(wd, "ELTeC-fra.csv"), "r", encoding="utf8") as infile: 
        data = pd.read_csv(infile, sep="\t", index_col="xmlid")
    print(data.head())
    print(data.columns)
    return data


def linear_regression1(data): 
    sns.lmplot(
        data=data,
        x="author-birth",
        y="author-death",
        )
    plt.savefig(join(wd, "5-ELTeC_simple.png"), dpi=300)
    plt.close()


def linear_regression2(data): 
    sns.lmplot(
        data=data,
        x="author-birth",
        y="author-death",
        hue="author-gender",
        )
    plt.savefig(join(wd, "5-ELTeC_hue.png"), dpi=300)
    plt.close()


def linear_regression3(data): 
    sns.lmplot(
        data=data,
        x="reference-year",
        y="numwords",
        )
    plt.savefig(join(wd, "5-ELTeC_refyear-to-numwords.png"), dpi=300)
    plt.close()


def linear_regression4(data): 
    
    # Recode gender as binary numerical value
    data["F"] = pd.get_dummies(data["author-gender"])["F"]
    print(data.head())    
    
    # Remove outlier in numwords: larger than 500.000 words 
    data = data.loc[data["numwords"] < 500000]
    print(data.shape)
    
    sns.lmplot(
        data=data,
        x="numwords",
        y="F",
        logistic=True,
        )
    plt.savefig(join(wd, "5-ELTeC_numwords-gender-logistic.png"), dpi=300)
    plt.close()



def linear_regression4(data): 
    
    # Recode gender as binary numerical value
    data["perspective"] = pd.get_dummies(data["narrative-perspective"])["heterodiegetic"]
    print(data.head())    
    
    # Remove outlier in numwords: larger than 500.000 words 
    data = data.loc[data["numwords"] < 500000]
    print(data.shape)
    
    sns.lmplot(
        data=data,
        x="numwords",
        y="perspective",
        logistic=True,
        )
    plt.savefig(join(wd, "5-ELTeC_numwords-perspective-logistic.png"), dpi=300)
    plt.close()


def linear_regression5(data): 
    
    # Recode gender as binary numerical value
    data["high"] = pd.get_dummies(data["reprint-count"])["high"]
    print(data.head())    
    
    # Remove outlier in numwords: larger than 500.000 words 
    data = data.loc[data["numwords"] < 500000]
    print(data.shape)
    
    sns.lmplot(
        data=data,
        x="numwords",
        y="high",
        logistic=True,
        )
    plt.savefig(join(wd, "5-ELTeC_numwords-reprintcount-logistic.png"), dpi=300)
    plt.close()




def main(): 
    data = load_data()
    linear_regression1(data)
    linear_regression2(data)
    linear_regression3(data)
    linear_regression4(data)
    linear_regression5(data)
main()















