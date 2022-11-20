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

# === Global variables

wd = join(os.path.realpath(os.path.dirname(__file__)))


# === Code === 

# Reglplot
tips = sns.load_dataset("tips")
sns.regplot(x="total_bill", y="tip", data=tips)
plt.savefig(join(wd, "tips-regplot.png"), dpi=300)

# Lmplot
sns.lmplot(x="total_bill", y="tip", data=tips)
plt.savefig(join(wd, "tips-lmplot.png"), dpi=300)

