import numpy
import pandas
from scipy.optimize import minimize
from scipy.stats import norm
data=pandas.read_csv("ponzr1.csv", sep=',')
data.shape
data.columns
#subset1=data.loc[data.mutation.isin(['WT','M124K']),:]
#subset2=data.loc[data.mutation.isin(['WT','V456D']),:]
#subset3=data.loc[data.mutation.isin(['WT','I213N']),:]

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

#to fix the y problem that was arising all you needed to do was change y to p because p is the argument not y.

#likelihood function for null
def null(p,obs):
    B0=p[0]
    sigma=p[1]
    
    expected=B0
    nll=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll
    
#likelihood function for when mutation effects expression
def mut(p,obs):
    B0=p[0]
    B1=p[1]
    sigma=p[2]
    
    expected=B0+B1*obs.x
    nll=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll
#t-test between control and M124K mutation
#data for control vs first mutation (M124K)
data1=data.loc[(data.mutation == "WT") | (data.mutation == "M124K")]
data1.columns=['x', 'y']
data1['x'] = data1['x'].map({'WT': 0, 'M124K': 1})

#parameters with null model
initialGuess=numpy.array([2000,1])
null=minimize(null, initialGuess, method="Nelder-Mead", options={'disp': True}, args=data1)
#print parameters
print(null.x)
#print nll
print(null.fun)

#parameters with mutation model 
initialGuess=numpy.array([2000,1000, 1])
mut=minimize(mut, initialGuess, method="Nelder-Mead", options={'disp': True}, args=data1)
#print parameters
print(mut.x)
#print nll
print(mut.fun)

#difference in nll calculation
D=2*(null.fun-mut.fun)
#test for statistical significance
1-scipy.stats.chi2.cdf(x=D,df=1)
