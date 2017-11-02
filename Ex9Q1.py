import numpy
import pandas
from scipy.optimize import minimize
from scipy.stats import norm
data=pandas.read_csv("ponzr1.csv", sep=',')
data.shape
data.columns
subset1=data.loc[data.mutation.isin(['WT','M124K']),:]
subset2=data.loc[data.mutation.isin(['WT','V456D']),:]
subset3=data.loc[data.mutation.isin(['WT','I213N']),:]

#def mean (p,obs):
 #   B0=y[0]
 #   sigma=y[1]
 #   expected=B0
 #   nll=-1*norm(expected,sigma).logpdf(obs.y).sum()
 #   return nll
#def like (p,obs):
 #   B0=p[0]
 #   B1=p[1]
 #   sigma=p[2]
 #   expected=B0+B1*obs.x
 #   nll=-1*norm(expected,sigma).logpdf(obs.y).sum()
 #   return nll
#initialGuess=numpy.array([1,1,1])
#fitNull=minimize(null,initialGuess,method="Nelder-Mead",options={'disp':True},args=subset1)
#fitAlter=minimize(alter,initialGuess,method="Nelder-Mead",options={'disp':True},args=subset1)
#D=2*(fitNull.fun-fitAlter.fun)
#1-scipy.stats.chi2.cdf(x=D,df=1)