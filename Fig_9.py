import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, colors
from math import ceil, factorial, pi
from cmath import exp,log
from scipy.stats import norm
import os
#import pandas as pd
import cvxopt
import time
data_path = ''.join(map(lambda string: string+'\\', os.path.realpath(__file__).split('\\')[:-1]))+'data/'

from Visualization import PopEV_Y_maker, MomentEstimator, H_Estimation
from Algorithms import PLSS_InferMean





if __name__ == "__main__":
    resolution=1/100
    ###################
    
    c = 1/20
    tau,kappa = (0.05,10)
    ExampleNumber=1
    verbose=False

    if 0: #Figure 9a
        g = lambda z: z**3
        d = 20
        NN = 500
        titleInsert = r'$z^{3}$'
        singualrityAtZero=False

    if 0: #Figure 9b
        g = lambda z: exp(z)
        d = 50
        NN = 500
        titleInsert = r'$\exp(z)$'
        singualrityAtZero=False

    if 1: #Figure 9c
        g = lambda z: log(z)
        d = 100
        NN = 500
        titleInsert = r'$\log(z)$'
        singualrityAtZero=True


    ###################


    n = ceil(d/c)
    Intervals = []
    CorrectConfidenceIntervals = 0

    PopEV,Y_List = PopEV_Y_maker(d,n,ExampleNumber,multiplicity=NN)
    truePLSS = sum([g(lam) for lam in PopEV])/d

    for i in range(NN):
        print(i)
        if ExampleNumber==3:
            Y = Y_List[i]
        else:
            _,Y = PopEV_Y_maker(d,n,ExampleNumber)


        estPLSS, mean, variance = PLSS_InferMean(Y,g,tau=tau,kappa=kappa,giveVariance=True,singualrityAtZero=singualrityAtZero)

        ConfidenceLevel = 0.95

        lower = norm.ppf((1-ConfidenceLevel)/2, loc=np.real(estPLSS-mean/n), scale=np.sqrt(np.real(variance/n**2)))
        upper = norm.ppf(1-(1-ConfidenceLevel)/2, loc=np.real(estPLSS-mean/n), scale=np.sqrt(np.real(variance/n**2)))
        Intervals.append((lower,upper))
        if lower <= truePLSS <= upper:
            CorrectConfidenceIntervals+=1

    print('Percentage with correct confidence interval:',100*CorrectConfidenceIntervals/NN,'%')

    fig, ax = plt.subplots(1,1,layout='constrained', figsize=(8, 5.6))

    for i, (lo, hi) in enumerate(Intervals):
        # Mark intervals that *miss* the true parameter in red
        color = "red" if not (lo <= truePLSS <= hi) else "C0"
        ax.hlines(i, lo, hi, colors=color)

    ax.axvline(np.real(truePLSS), linestyle="--", color="black")

    ax.set_title(r'Confidence intervals for g(z) = {}'.format(titleInsert))
    plt.show()


    

    