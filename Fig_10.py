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

from VisualizationTools import PopEV_Y_maker, MomentEstimator, H_Estimation
from Algorithms import PLSS_ConfidenceInterval





if __name__ == "__main__":
    resolution=1/100
    ###################
    
    c = 1/20
    tau,kappa = (0.05,10)
    ExampleNumber=1
    ConfidenceLevel=0.95

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

        d,n = Y.shape
        c = d/n
        S = Y@Y.T/n
        SampEV,_ = np.linalg.eigh(S)

        lower,upper = PLSS_ConfidenceInterval(SampEV,d,n,g,tau=tau,kappa=kappa,ConfidenceLevel=ConfidenceLevel,singualrityAtZero=singualrityAtZero)
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


    

    