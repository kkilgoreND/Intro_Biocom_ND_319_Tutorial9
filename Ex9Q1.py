import numpy
import pandas
import scipy
from scipy.optimize import minimize
from scipy.stats import norm
data=pandas.read_csv("ponzr1.csv", sep=',')
data.shape
data.columns
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
#t-test between control and mutations
#data for control vs first mutation (M124K)
data1=data.loc[(data.mutation == "WT") | (data.mutation == "M124K")]
data1.columns=['x', 'y']
data1['x'] = data1['x'].map({'WT': 0, 'M124K': 1})
#data for control vs second mutation (V456D)
data2=data.loc[(data.mutation == "WT") | (data.mutation == "V456D")]
data2.columns=['x', 'y']
data2['x'] = data2['x'].map({'WT': 0, 'V456D': 1})
#data for control vs third mutation (I213N)
data3=data.loc[(data.mutation == "WT") | (data.mutation == "I213N")]
data3.columns=['x', 'y']
data3['x'] = data3['x'].map({'WT': 0, 'I213N': 1})

#parameters with null model; could we have used a for loop here?
initialGuess=numpy.array([2000,1])
null1=minimize(null, initialGuess, method="Nelder-Mead", options={'disp': True}, args=data1)
null2=minimize(null, initialGuess, method="Nelder-Mead", options={'disp': True}, args=data2)
null3=minimize(null, initialGuess, method="Nelder-Mead", options={'disp': True}, args=data3)
#print parameters
print(null1.x)
print(null2.x)
print(null3.x)
#print nll
print(null1.fun)
print(null2.fun)
print(null3.fun)

#parameters with mutation model; ditto about the for loop? 
initialGuess=numpy.array([2000,1000, 1])
mut1=minimize(mut, initialGuess, method="Nelder-Mead", options={'disp': True}, args=data1)
mut2=minimize(mut, initialGuess, method="Nelder-Mead", options={'disp': True}, args=data2)
mut3=minimize(mut, initialGuess, method="Nelder-Mead", options={'disp': True}, args=data3)

#print parameters
print(mut1.x)
print(mut2.x)
print(mut3.x)

#print nll
print(mut1.fun)
print(mut2.fun)
print(mut3.fun)

#difference in nll calculation
D1=2*(null1.fun-mut1.fun)
D2=2*(null2.fun-mut2.fun)
D3=2*(null3.fun-mut3.fun)

#test for statistical significance
print ("The effect of M124K", 1-scipy.stats.chi2.cdf(x=D1,df=1))
print ("The effect of V456D", 1-scipy.stats.chi2.cdf(x=D2,df=1))
print ("The effect of I213N", 1-scipy.stats.chi2.cdf(x=D3,df=1))
