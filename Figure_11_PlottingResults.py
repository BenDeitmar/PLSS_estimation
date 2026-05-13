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
if 0: #Figure 11a
    c = 1/5
    ExampleNumber=1
    #ToCompare = ['LedoitWolf', 'ElKaroui','MPI','KongValiant','BaiChenYao']
    ToCompare = ['MPI', 'BaiChenYao', 'KongValiant']
    data_name='monomialEx1'
    IngoredMoments=1
    TargetMomentEntry=5
    d_ListEntries=-1

if 0: #Figure 11b
    c = 2
    ExampleNumber=2
    #ToCompare = ['LedoitWolf', 'ElKaroui','MPI','KongValiant','BaiChenYao']
    ToCompare = ['MPI', 'BaiChenYao', 'KongValiant']
    data_name='monomialEx2'
    IngoredMoments=2
    TargetMomentEntry=5
    d_ListEntries=-1

if 1: #Figure 11c
    c = 1/4
    ExampleNumber=3
    #ToCompare = ['LedoitWolf', 'ElKaroui','MPI','KongValiant','BaiChenYao']
    ToCompare = ['MPI', 'BaiChenYao', 'KongValiant']
    data_name='monomialEx3'
    IngoredMoments=1
    TargetMomentEntry=5
    d_ListEntries=-1

###################

offsets0 = np.linspace(-0.2, 0.2, len(ToCompare))
    
ColorMap = {'LedoitWolf': 'black', 'ElKaroui': 'blue', 'KongValiant': 'purple', 'MPI': 'orange', 'BaiChenYao': 'red'}
LegendMap = {'LedoitWolf': 'Ledoit-Wolf', 'ElKaroui': 'El Karoui', 'KongValiant': 'Kong-Valiant', 'MPI': 'proposed', 'BaiChenYao': 'Bai-Chen-Yao'}

fig, ax = plt.subplots(1,1,layout='constrained', figsize=(8, 5.6))


try:
    d_List = np.load(data_path+'Fig10_d_List.npy')
    AvgTimes = dict()
    AllErrors = dict()
    for key in ToCompare:
        AvgTimes[key] = np.load(data_path+'Fig10_AvgTimes_{}_c={}_{}_Ex{}.npy'.format(key,c,data_name,ExampleNumber))
        AllErrors[key] = np.load(data_path+'Fig10_AllErrors_{}_c={}_{}_Ex{}.npy'.format(key,c,data_name,ExampleNumber))
except:
    print('#############################')
    print("Error: could not load the results")
    print("try running Figure_10_1_Preparation.R with the same choices for c and ExampleNumber first")
    print(" then running Figure_10_2_Calculations.py with the same choices for c, ExampleNumber and data_name first")
    print('#############################')
    assert 0==1

ax.set_title('Estimation errors for various moments \n and fixed d = {} under Example {}'.format(int(d_List[d_ListEntries]), ExampleNumber))

#ax[0].plot([0,10],[0]*2,color='gray',alpha=0.3)
#ax[1].plot(d_List,[0]*len(d_List),color='gray',alpha=0.3)
#ax[2].plot(d_List,[0]*len(d_List),color='gray',alpha=0.3)

#ax[0].set_ylim([-0.05*ymax0, ymax0])
#ax[1].set_ylim([-0.05*ymax1, ymax1])

#print(AvgErrors['MPI'])

for i in range(len(ToCompare)):
    key = ToCompare[i]
    Errors = AllErrors[key][:,:,IngoredMoments:]

    mean0 = Errors.mean(axis=1)[d_ListEntries,:]
    lower0 = np.quantile(Errors[d_ListEntries,:], 0.1, axis=0)
    upper0 = np.quantile(Errors[d_ListEntries,:], 0.9, axis=0)
    lower_err0 = mean0 - lower0
    upper_err0 = upper0 - mean0

    ax.errorbar(IngoredMoments+np.array(range(len(mean0))) + offsets0[i],mean0,yerr=[lower_err0, upper_err0],alpha=1,fmt='o',capsize=4,color=ColorMap[key],label=LegendMap[key])
    ax.semilogy()
    

ax.grid(True, alpha=0.3)
ax.legend(loc="upper left")

plt.show()

    

