#load packages
import numpy
import pandas
import scipy
from scipy.optimize import minimize
from scipy.stats import norm

#load dataset
data=pandas.read_csv("MmarinumGrowth.csv")
data.shape
data.columns

#define that funky function
def Monod(p,obs):
    max=p[0]
    K=p[1]
    sigma=p[2]
    expected=max*(obs.S/(obs.S+K))
    nll=-1*norm(expected,sigma).logpdf(obs.u).sum()
    return nll
#run the fit function
initialGuess=numpy.array([1,1,1])
fit=minimize(Monod,initialGuess,method="Nelder-Mead",options={'disp':True},args=data)
print(fit.x)
#oh yeah, thanks YY!