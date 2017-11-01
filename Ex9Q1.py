import numpy
import pandas
from scipy.optimize import minimize
from scipy.stats import norm
from plotnine import *
data=pandas.read_csv("ponzr1.csv", sep=',')
data.shape
data.columns
#subset1=data.loc[data.mutation.isin(['WT','M124K']),:]
#subset2=data.loc[data.mutation.isin(['WT','V456D']),:]
#subset3=data.loc[data.mutation.isin(['WT','I213N']),:]
def null (p,obs):
    B0=p[0]
    sigma=p[1]
    expected=B0
    nll=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll
def alter(p,obs):
    B0=p[0]
    B1=p[1]
    sigma=p[2]
    expected=B0+B1*obs.x
    nll=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll
initialGuess=numpy.array([1,1,1])
df=1
#fitNull=minimize(subset1,initialGuess,method="Nelder-Mead",options={'disp':True},args=df)
#fitAlter=minimize(subset1,initialGuess,method="Nelder-Mead",options={'disp':True},args=df)
#D=2*(fitNull.fun-fitAlter.fun)
#1-scipy.stats.chi2.cdf(x=D,df=1)