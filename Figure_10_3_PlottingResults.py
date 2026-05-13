import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, colors
from math import ceil
from cmath import exp, log
import os
import pandas as pd
import cvxopt
import time
data_path = ''.join(map(lambda string: string+'\\', os.path.realpath(__file__).split('\\')[:-1]))+'data/'


###################
if 0: #Figure 10a
    c = 1/5
    ExampleNumber=1
    #ToCompare = ['LedoitWolf', 'ElKaroui','MPI','KongValiant','BaiChenYao']
    ToCompare = ['MPI', 'BaiChenYao', 'KongValiant']
    g_name='monomialEx1'
    IngoredMoments=1
    TargetMomentEntry=5
    d_ListEntries=-1

if 0: #Figure 10b
    c = 2
    ExampleNumber=2
    #ToCompare = ['LedoitWolf', 'ElKaroui','MPI','KongValiant','BaiChenYao']
    ToCompare = ['MPI', 'BaiChenYao', 'KongValiant']
    g_name='monomialEx2'
    IngoredMoments=2
    TargetMomentEntry=5
    d_ListEntries=-1

if 1: #Figure 10c
    c = 1/4
    ExampleNumber=3
    #ToCompare = ['LedoitWolf', 'ElKaroui','MPI','KongValiant','BaiChenYao']
    ToCompare = ['MPI', 'BaiChenYao', 'KongValiant']
    g_name='monomialEx3'
    IngoredMoments=1
    TargetMomentEntry=5
    d_ListEntries=-1

###################


offsets1 = np.linspace(-10, 10, len(ToCompare))
    
ColorMap = {'LedoitWolf': 'black', 'ElKaroui': 'blue', 'KongValiant': 'purple', 'MPI': 'orange', 'BaiChenYao': 'red'}
LegendMap = {'LedoitWolf': 'Ledoit-Wolf', 'ElKaroui': 'El Karoui', 'KongValiant': 'Kong-Valiant', 'MPI': 'proposed', 'BaiChenYao': 'Bai-Chen-Yao'}

fig, ax = plt.subplots(1,2,layout='constrained', figsize=(16, 4), gridspec_kw={"width_ratios": [2, 1]})


try:
    d_List = np.load(data_path+'Fig10_d_List.npy')
    AvgTimes = dict()
    AllErrors = dict()
    for key in ToCompare:
        AvgTimes[key] = np.load(data_path+'Fig10_AvgTimes_{}_c={}_{}_Ex{}.npy'.format(key,c,g_name,ExampleNumber))
        AllErrors[key] = np.load(data_path+'Fig10_AllErrors_{}_c={}_{}_Ex{}.npy'.format(key,c,g_name,ExampleNumber))
except:
    print('#############################')
    print("Error: could not load the results")
    print("try running Figure_10_1_Preparation.R with the same choices for c and ExampleNumber first")
    print(" then running Figure_10_2_Calculations.py with the same choices for c, ExampleNumber and g_name first")
    print('#############################')
    assert 0==1

ax[0].set_title('Estimation errors for various d-values \n and fixed {}-th moment under Example {}'.format(TargetMomentEntry, ExampleNumber))
ax[1].set_title('Average calculation time for all moments up to K={} \n under Example {}'.format(10, ExampleNumber))


for i in range(len(ToCompare)):
    key = ToCompare[i]
    Errors = AllErrors[key][:,:,IngoredMoments:]

    mean1 = Errors.mean(axis=1)[:,TargetMomentEntry]
    lower1 = np.quantile(Errors[:,:,TargetMomentEntry], 0.1, axis=1)
    upper1 = np.quantile(Errors[:,:,TargetMomentEntry], 0.9, axis=1)
    lower_err1 = mean1 - lower1
    upper_err1 = upper1 - mean1

    ax[0].errorbar(d_List + offsets1[i],mean1,yerr=[lower_err1, upper_err1],alpha=1,fmt='o',capsize=4,color=ColorMap[key],label=LegendMap[key])
    ax[0].semilogy()

    ax[1].semilogy(d_List,AvgTimes[key],color=ColorMap[key],alpha=0.5,linewidth=4,label=LegendMap[key])
    

for i in range(2):
	ax[i].grid(True, alpha=0.3)
	ax[i].legend(loc="upper left")
if ExampleNumber!=3:
    ax[0].legend(loc="upper right")

plt.show()

    

