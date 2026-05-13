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

from VisualizationTools import PopEV_Y_maker, KongValiant_MomentEstimator, KongValiant_MomentEstimator_MultipleMoments, BaiChenYao_population_moments, H_Estimation
from Algorithms import PLSS_estimator_FullSpectrum




if __name__ == "__main__":
    resolution=1/100
    ToCompare = {'LedoitWolf', 'ElKaroui', 'MPI', 'KongValiant', 'BaiChenYao'}
    ###################
    
    if 0: #Figure 10a
        c = 1/5
        tau,kappa = (0.05,20)
        K = 10
        data_name='monomialEx1'
        ExampleNumber=1
    
    if 0: #Figure 10b
        c = 2
        tau,kappa = (0.1,20)
        K = 10
        data_name='monomialEx2'
        ExampleNumber=2
    
    if 1: #Figure 10c
        c = 1/4
        tau,kappa = (0.05,25)
        K = 10
        data_name='monomialEx3'
        ExampleNumber=3
    ###################

    
    
    try:
        d_List = np.load(data_path+'Fig10_d_List.npy')
        AvgTimes_LedoitWolf = np.load(data_path+'Fig10_AvgTimes_LedoitWolf_c={}_Ex{}.npy'.format(c,ExampleNumber))
        LedoitWolf_EstimatedEV_List = []
        AllDataMatrices = []
        AllPopEVs = []
        for d in d_List:
            AllDataMatrices.append(np.load(data_path+'Fig10_DataMatrices_c={}_Ex{}_d={}.npy'.format(c,ExampleNumber,int(d))))
            AllPopEVs.append(np.load(data_path+'Fig10_PopEVs_c={}_Ex{}_d={}.npy'.format(c,ExampleNumber,int(d))))
            LedoitWolf_EstimatedEV_List.append(np.load(data_path+'Fig10_LedoitWolf_Estimators_d={}_c={}_Ex{}.npy'.format(int(d),c,ExampleNumber)))
            _,NN = LedoitWolf_EstimatedEV_List[0].shape
    except:
        print('#############################')
        print("Error: could not load the results of the Ledoit-Wolf estimator")
        print("try running Figure_10_1_Preparation.R with the same choices for c and ExampleNumber first")
        print('#############################')
        assert 0==1

    AllErrors = dict()
    AvgTimes = dict()
    
    for key in ToCompare:
        AllErrors[key] = []
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
            truePLSS = np.array([sum([lam**k for lam in PopEV])/d for k in range(K+1)])

            print('true:',truePLSS)

            if 'LedoitWolf' in ToCompare:
                LedoitWolf_PLSS = np.array([sum([lam**k for lam in LedoitWolf_EstimatedEVs[:,i]])/d for k in range(K+1)])
                print('LW:',LedoitWolf_PLSS)
                Errors['LedoitWolf'].append(LedoitWolf_PLSS-truePLSS)

            if 'ElKaroui' in ToCompare:
                start = time.time()
                positions,weights = H_Estimation(Y,eta=0.05)
                end = time.time()
                MomentEstimators = np.zeros(K+1)
                for k in range(K+1):
                    g = lambda x: x**k
                    ElKaroui_PLSS = sum([g(p)*w for p,w in zip(positions,weights)])
                    MomentEstimators[k] = ElKaroui_PLSS
                print('EK:',MomentEstimators)
                Errors['ElKaroui'].append(MomentEstimators-truePLSS)
                TimeDiff['ElKaroui'] += end - start

            if 'KongValiant' in ToCompare:
                start = time.time()
                MomentEst = KongValiant_MomentEstimator_MultipleMoments(Y,K)
                #MomentEst = np.array([KongValiant_MomentEstimator(Y,k) for k in range(K+1)])
                MomentEst[0] = 1
                end = time.time()
                print('KV:',np.array(MomentEst))
                Errors['KongValiant'].append(MomentEst-truePLSS)
                TimeDiff['KongValiant'] += end - start

            if 'BaiChenYao' in ToCompare:
                start = time.time()
                MomentEst = BaiChenYao_population_moments(Y,K)
                end = time.time()
                print('BCY:',MomentEst)
                Errors['BaiChenYao'].append(MomentEst-truePLSS)
                TimeDiff['BaiChenYao'] += end - start

            if 'MPI' in ToCompare:
                start = time.time()
                g = [(lambda x, k=k: x**k) for k in range(K+1)]
                CurveEst = PLSS_estimator_FullSpectrum(Y,g,tau=tau,kappa=kappa,resolution=resolution)
                end = time.time()
                print('MPI:',np.real(CurveEst))
                Errors['MPI'].append(CurveEst-truePLSS)
                TimeDiff['MPI'] += end - start

        for key in ToCompare:
            AvgTimes[key].append(TimeDiff[key]/NN)
            AllErrors[key].append(np.abs(Errors[key]))

    if 'LedoitWolf' in ToCompare:
        AvgTimes['LedoitWolf'] = AvgTimes_LedoitWolf

    for key in ToCompare:
        np.save(data_path+'Fig10_AvgTimes_{}_c={}_{}_Ex{}'.format(key,c,data_name,ExampleNumber),np.array(AvgTimes[key]))
        np.save(data_path+'Fig10_AllErrors_{}_c={}_{}_Ex{}'.format(key,c,data_name,ExampleNumber),np.array(AllErrors[key]))
