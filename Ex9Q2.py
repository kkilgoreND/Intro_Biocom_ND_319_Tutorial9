import numpy
import pandas
from scipy.optimize import minimize
from scipy.stats import norm
from plotnine import *
data=pandas.read_csv("MmarinumGrowth.csv", sep=',')
data.shape
data.columns
