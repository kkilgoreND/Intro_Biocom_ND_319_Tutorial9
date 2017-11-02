import numpy
import pandas
import scipy
from scipy.optimize import minimize
from scipy.stats import norm
#add'leafDecomp.csv'
ld=pandas.read_csv("leafDecomp.csv",header=0)

#make new dataframe with x and y as the headers
leaves=ld
leaves.columns=['x', 'y']
leaves.head()

#liklihood func. for constant decomp
def constant(p,obs):
    B0=p[0]
    sigma=p[1]
    expected=B0
    nll=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll
    
#set initial guess
constantGuess=numpy.array([600,1])

#estimate parameters
constantFit=minimize(constant,constantGuess,method="Nelder-Mead",options={'disp':True},args=leaves)
print(constantFit.x)
#print nll
print(constantFit.fun)

#liklihood func. for linear decomp
def linear(p,obs):
    B0=p[0]
    B1=p[1]
    sigma=p[2]
    expected=B0+B1*obs.x
    nll=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll
    
#set initial guesses
linearGuess=numpy.array([10,6,1])
#estimate parameters
linearFit=minimize(linear,linearGuess,method="Nelder-Mead",options={'disp':True},args=leaves)
print(linearFit.x)
#print nll
print(linearFit.fun)

#liklihood function for hump decomp.
def hump(p,obs):
    B0=p[0]
    B1=p[1]
    B2=p[2]
    sigma=p[3]
    expected=B0+B1*obs.x+B2*((obs.x)**2)
    nll=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll
    
#set initial guesses
humpGuess=numpy.array([200,10,-0.2,1])

#estimate parameters
humpFit=minimize(hump,humpGuess,method="Nelder-Mead",options={'disp':True},args=leaves)
print(humpFit.x)
#print nll
print(humpFit.fun)

#difference in nll(constant vs linear)
first_D=2*(constantFit.fun-linearFit.fun)

#test for statistical significance
1-scipy.stats.chi2.cdf(x=first_D,df=1)

#difference in nll(linear vs hump)
second_D=2*(linearFit.fun-humpFit.fun)

#test for statistical significance
1-scipy.stats.chi2.cdf(x=second_D,df=1)

#difference in nll(constant vs hump)
third_D=2*(constantFit.fun-humpFit.fun)

#test for statistical significance
1-scipy.stats.chi2.cdf(x=third_D,df=2)
