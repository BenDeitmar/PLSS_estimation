import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, colors
from math import ceil, factorial
from cmath import exp,log
import os
#import pandas as pd
import cvxopt
import time
data_path = ''.join(map(lambda string: string+'\\', os.path.realpath(__file__).split('\\')[:-1]))+'data/'

from Visualization import PopEV_Y_maker, MomentEstimator, H_Estimation
from Algorithms import PLSS_estimator_FullSpectrum




if __name__ == "__main__":
    resolution=1/100
    ToCompare = {'LedoitWolf', 'ElKaroui', 'MPI', 'KongValiant'}
    ###################
    
    if 0: #Figure 7a
        c = 1/10
        tau,kappa = (0.05,10)
        K = 7
        g = lambda z: z**7
        g_name='monomial7'
        ExampleNumber=1
    
    if 0: #Figure 7b
        c = 2
        tau,kappa = (0.1,10)
        K = 5
        g = lambda z: z**5
        g_name='monomial5'
        ExampleNumber=2
    
    if 1: #Figure 7c
        c = 1/4
        tau,kappa = (0.05,25)
        K = 3
        g = lambda z: z**3
        g_name='monomial3'
        ExampleNumber=3
    ###################

    

    
    try:
        d_List = np.load(data_path+'Fig7_d_List.npy')
        AvgTimes_LedoitWolf = np.load(data_path+'Fig7_AvgTimes_LedoitWolf_c={}_Ex{}.npy'.format(c,ExampleNumber))
        LedoitWolf_EstimatedEV_List = []
        AllDataMatrices = []
        AllPopEVs = []
        for d in d_List:
            AllDataMatrices.append(np.load(data_path+'Fig7_DataMatrices_c={}_Ex{}_d={}.npy'.format(c,ExampleNumber,int(d))))
            AllPopEVs.append(np.load(data_path+'Fig7_PopEVs_c={}_Ex{}_d={}.npy'.format(c,ExampleNumber,int(d))))
            LedoitWolf_EstimatedEV_List.append(np.load(data_path+'Fig7_LedoitWolf_Estimators_d={}_c={}_Ex{}.npy'.format(int(d),c,ExampleNumber)))
            _,NN = LedoitWolf_EstimatedEV_List[0].shape
    except:
        print('#############################')
        print("Error: could not load the results of the Ledoit-Wolf estimator")
        print("try running Fig_7_Preparation.R with the same choices for c and ExampleNumber first")
        print('#############################')
        assert 0==1

    AvgErrors = dict()
    Variances = dict()
    AvgTimes = dict()
    
    for key in ToCompare:
        AvgErrors[key] = []
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

            print('true:',truePLSS)

            if 'LedoitWolf' in ToCompare:
                LedoitWolf_PLSS = sum([g(lam) for lam in LedoitWolf_EstimatedEVs[:,i]])/d
                print('LW:',LedoitWolf_PLSS)
                if 0 and d>50:
                    plt.plot(LedoitWolf_EstimatedEVs[:,i],[j/d for j in range(1,d+1)])
                Errors['LedoitWolf'].append(LedoitWolf_PLSS-truePLSS)

            if 'ElKaroui' in ToCompare:
                start = time.time()
                positions,weights = H_Estimation(Y,eta=0.05)
                end = time.time()
                if 0 and d>50:
                    plt.plot(positions,np.cumsum(weights))
                    plt.show()
                    plt.clf()
                ElKaroui_PLSS = sum([g(p)*w for p,w in zip(positions,weights)])
                print('EK:',ElKaroui_PLSS)
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
                CurveEst = PLSS_estimator_FullSpectrum(Y,g,tau=tau,kappa=kappa,resolution=resolution)
                end = time.time()
                print('MPI:',CurveEst)
                Errors['MPI'].append(CurveEst-truePLSS)
                TimeDiff['MPI'] += end - start

        for key in ToCompare:
            AvgTimes[key].append(TimeDiff[key]/NN)
            AvgErrors[key].append(sum(np.abs(Errors[key]))/NN)
            mean = sum(Errors[key])/NN
            Variances[key].append(sum(np.abs(np.array(Errors[key])-mean)**2)/(NN-1))

    if 'LedoitWolf' in ToCompare:
        AvgTimes['LedoitWolf'] = AvgTimes_LedoitWolf

    for key in ToCompare:
        np.save(data_path+'Fig7_AvgTimes_{}_c={}_{}_Ex{}'.format(key,c,g_name,ExampleNumber),np.array(AvgTimes[key]))
        np.save(data_path+'Fig7_AvgErrors_{}_c={}_{}_Ex{}'.format(key,c,g_name,ExampleNumber),np.array(AvgErrors[key]))
        np.save(data_path+'Fig7_Variances_{}_c={}_{}_Ex{}'.format(key,c,g_name,ExampleNumber),np.array(Variances[key]))
