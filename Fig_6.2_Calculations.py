import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, colors
from math import ceil, factorial
from cmath import exp,log
import os
#import pandas as pd
import time
data_path = ''.join(map(lambda string: string+'\\', os.path.realpath(__file__).split('\\')[:-1]))+'data/'

from Visualization import PopEV_Y_maker, MomentEstimator, H_Estimation
from Algorithms import Vectorwise_StieltjesTransformEstimator




if __name__ == "__main__":
    ###################
    c = 2
    ExampleNumber=2
    K = None
    a,b=-np.inf,np.inf

    if 1: #Figure 6a
        z = -1+0.05j
        g = lambda lam: 1/(lam-z)
        g_name='Stieltjes_a'
        resolution=1/100
        ToCompare = {'LedoitWolf', 'ElKaroui','MPI'}
    if 0: #Figure 6b
        z = -0.5+0.1j
        g = lambda lam: 1/(lam-z)
        g_name='Stieltjes_b'
        resolution=1/100
        ToCompare = {'LedoitWolf', 'ElKaroui'}
    if 0: #Figure 6c
        z = 0+0.1j
        g = lambda lam: 1/(lam-z)
        g_name='Stieltjes_c'
        resolution=1/100
        ToCompare = {'LedoitWolf', 'ElKaroui'}
    ###################

    

    
    try:
        d_List = np.load(data_path+'Fig6_d_List.npy')
        AvgTimes_LedoitWolf = np.load(data_path+'Fig6_AvgTimes_LedoitWolf_c={}_Ex{}.npy'.format(c,ExampleNumber))
        LedoitWolf_EstimatedEV_List = []
        AllDataMatrices = []
        AllPopEVs = []
        ChosenDimList = []
        for d in d_List:
            AllDataMatrices.append(np.load(data_path+'Fig6_DataMatrices_c={}_Ex{}_d={}.npy'.format(c,ExampleNumber,int(d))))
            AllPopEVs.append(np.load(data_path+'Fig6_PopEVs_c={}_Ex{}_d={}.npy'.format(c,ExampleNumber,int(d))))
            LedoitWolf_EstimatedEV_List.append(np.load(data_path+'Fig6_LedoitWolf_Estimators_d={}_c={}_Ex{}.npy'.format(int(d),c,ExampleNumber)))
            _,NN = LedoitWolf_EstimatedEV_List[0].shape
    except:
        print('#############################')
        print("Error: could not load the results of the Ledoit-Wolf estimator")
        print("try running Fig_6_Preparation.R with the same choices for c and ExampleNumber first")
        print('#############################')
        assert 0==1

    AllErrors = dict()
    Variances = dict()
    AvgTimes = dict()
    
    for key in ToCompare:
        AllErrors[key] = []
        Variances[key] = []
        AvgTimes[key] = []


    for k in range(len(d_List)):
        d=int(d_List[k])

        n = ceil(d/c)
        print('d=',d)

        TimeDiff = dict()
        Errors = dict()

        for key in ToCompare:
            TimeDiff[key] = 0
            Errors[key] = []

        LedoitWolf_EstimatedEVs = LedoitWolf_EstimatedEV_List[k]

        DataMatrices = AllDataMatrices[k]
        PopEVs = AllPopEVs[k]

        for i in range(NN):
            print(i)
            Y = DataMatrices[i,:,:]
            PopEV = PopEVs[i,:]
            truePLSS = sum([g(lam) for lam in PopEV])/d

            if 'LedoitWolf' in ToCompare:
                LedoitWolf_PLSS = sum([g(lam) for lam in LedoitWolf_EstimatedEVs[:,i]])/d
                Errors['LedoitWolf'].append(LedoitWolf_PLSS-truePLSS)

            if 'ElKaroui' in ToCompare:
                start = time.time()
                positions,weights = H_Estimation(Y)
                end = time.time()
                ElKaroui_PLSS = sum([g(p)*w for p,w in zip(positions,weights)])
                Errors['ElKaroui'].append(ElKaroui_PLSS-truePLSS)
                TimeDiff['ElKaroui'] += end - start

            if 'KongValiant' in ToCompare:
                start = time.time()
                MomentEst = MomentEstimator(Y,K)
                end = time.time()
                Errors['KongValiant'].append(MomentEst-truePLSS)
                TimeDiff['KongValiant'] += end - start

            if 'MPI' in ToCompare:
                start = time.time()
                Y = np.matrix(Y)
                S = Y@Y.H/n
                SampEV,_ = np.linalg.eigh(S)
                CurveEst = Vectorwise_StieltjesTransformEstimator(np.array([z]),SampEV,c,maxIterations=300)[0]
                end = time.time()
                Errors['MPI'].append(CurveEst-truePLSS)
                TimeDiff['MPI'] += end - start

        for key in ToCompare:
            AvgTimes[key].append(TimeDiff[key]/NN)
            AllErrors[key].append(np.abs(Errors[key]))

    if 'LedoitWolf' in ToCompare:
        AvgTimes['LedoitWolf'] = AvgTimes_LedoitWolf

    for key in ToCompare:
        np.save(data_path+'Fig6_AvgTimes_{}_c={}_{}_Ex{}'.format(key,c,g_name,ExampleNumber),np.array(AvgTimes[key]))
        np.save(data_path+'Fig6_AllErrors_{}_c={}_{}_Ex{}'.format(key,c,g_name,ExampleNumber),np.array(AllErrors[key]))
        np.save(data_path+'Fig6_Variances_{}_c={}_{}_Ex{}'.format(key,c,g_name,ExampleNumber),np.array(Variances[key]))
