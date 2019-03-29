from math import sqrt, log
from read_data import *
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def hoeffding(erisk,delta):
    # Get the whoel dataset to find m
    (xvals,yvals) = read_data(1)
    m = xvals.shape[0]
    # Calculate the Hoeffding's Confidence Interval with empirical risk of algorithms
    lower = erisk - sqrt(log(2/delta)/(2*m))
    upper = erisk + sqrt(log(2/delta)/(2*m))

    return (lower,upper)

''' Count data plots
def analysis():
    data = pd.read_csv("spambase.data",header=None)
    labels = data.iloc[:,-1]
    sns.countplot(x=labels)
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.savefig('count.png')
    plt.show()
    plt.close()
'''