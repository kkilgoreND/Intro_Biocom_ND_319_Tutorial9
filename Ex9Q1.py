import numpy
import pandas
from scipy.optimize import minimize
from scipy.stats import norm
from plotnine import *
data=pandas.read_csv('ponzr1.csv')
subset1=data.loc[data.mutation.isin(['WT','M124K']),:]
subset2=data.loc[data.mutation.isin(['WT','V456D']),:]
subset3=data.loc[data.mutation.isin(['WT','I213N']),:]
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
#args=df is throwing an error of not defined
fitNull=minimize(subset1,initialGuess,method="Nelder-Mead",options={'disp':True},args=df)
fitAlter=minimize(subset1,initialGuess,method="Nelder-Mead",options={'disp':True},args=df)
D=2*(fitNull.fun-fitAlter.fun)
1-chi2.cdf(x=D,df=1)

#Chi-squared distribution with one deg of free
#1-scipy.stats.chi2.cdf(x=D,df=1)