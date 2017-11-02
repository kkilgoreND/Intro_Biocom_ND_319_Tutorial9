import numpy
import pandas
from scipy.optimize import curve_fit
from scipy.stats import norm
from plotnine import *
data=pandas.read_csv("MmarinumGrowth.csv", names=['S','u'])
data.shape
data.columns
S=[0]
u=[1]
#define x as S from csv
x=numpy.array("S")
#define y as u from csv
y=numpy.array("u")
#u=umax*(S/(S+K))