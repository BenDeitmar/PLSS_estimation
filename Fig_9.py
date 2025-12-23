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
from Algorithms import PLSS_InferMean





if __name__ == "__main__":
    resolution=1/100
    ###################
    
    c = 1/20
    tau,kappa = (0.05,10)
    ExampleNumber=1
    verbose=False

    if 1: #Figure 8a
        g = lambda z: z**3
        d = 20
        NN = 10000
        titleInsert = r'$z^{3}$'
        singualrityAtZero=False

    if 0: #Figure 8b
        g = lambda z: exp(z)
        d = 50
        NN = 10000
        titleInsert = r'$\exp(z)$'
        singualrityAtZero=False

    if 0: #Figure 8c
        g = lambda z: log(z)
        d = 100
        NN = 10000
        titleInsert = r'$\log(z)$'
        singualrityAtZero=True

    ###################


    n = ceil(d/c)
    Estimators = []
    Differences = []
    Intervals = []
    Means = []
    Variances = []
    CorrectConfidenceIntervals = 0

    PopEV,Y_List = PopEV_Y_maker(d,n,ExampleNumber,multiplicity=NN)

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

        truePLSS = sum([g(lam) for lam in PopEV])/d

        estPLSS, mean, variance = PLSS_InferMean(SampEV,c,g,tau=tau,kappa=kappa,giveVariance=True,singualrityAtZero=singualrityAtZero)

        diff = n*(estPLSS-truePLSS)

        if verbose:
            print('true, diff:', truePLSS, diff)
            print('mean, var:', mean, variance)

        Estimators.append(estPLSS)
        Means.append(mean)
        Variances.append(variance)
        Differences.append(diff)

        ConfidenceLevel = 0.95

        lower = norm.ppf((1-ConfidenceLevel)/2, loc=np.real(mean), scale=np.sqrt(np.real(variance)))
        upper = norm.ppf(1-(1-ConfidenceLevel)/2, loc=np.real(mean), scale=np.sqrt(np.real(variance)))
        if lower<diff and diff<upper:
            CorrectConfidenceIntervals+=1

    print('Percentage with correct confidence interval:',100*CorrectConfidenceIntervals/NN,'%')

    Intervals = np.array(Intervals)
    Differences = np.array(Differences)

    x = np.linspace(min(Differences), max(Differences), 300)
    mu,sigma2 = sum(Means)/NN, sum(Variances)/NN
    #mu,sigma2 = Means[0], Variances[0]
    
    pdf = (1.0 / (np.sqrt(2 * np.pi * sigma2))) * np.exp(- (x - mu)**2 / (2 * sigma2))

    fig, ax = plt.subplots(1,1,layout='constrained', figsize=(8, 5.6))
    ax.hist(Differences, bins=30, density=True, alpha=0.5, edgecolor='black')
    ax.set_title(r'Rescaled estiamtion errors for g(z) = {}'.format(titleInsert))
    #ax.hist(Differences, density=True, alpha=0.5, edgecolor='black')
    ax.plot(x, pdf, linewidth=2)
    plt.show()


    

    