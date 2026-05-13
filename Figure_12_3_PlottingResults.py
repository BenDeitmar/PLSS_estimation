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
if 1: #Figure 12
    d = 784
    ExampleNumber=3
    ToCompare = ['MPI','LedoitWolf']
    ymax0,ymax1 = 2,0.5
    titleInsert = r'$z^{3}$ and example 3'
    SampVecNr = 1
    PopVecNr = 1

###################
    
ColorMap = {'LedoitWolf': 'black', 'ElKaroui': 'blue', 'KongValiant': 'purple', 'MPI': 'orange', 'MPI_unrestricted': 'red'}
LegendMap = {'LedoitWolf': 'Ledoit-Wolf', 'ElKaroui': 'El Karoui', 'KongValiant': 'Kong-Valiant', 'MPI': 'proposed'}

fig, ax = plt.subplots(1,layout='constrained', figsize=(8, 5.6))

offsets = np.linspace(-10, 10, len(ToCompare)+1)

ax.set_title(r'Estimation error for $|u_1^{\top} v_1|^2$')

try:
    n_List = np.load(data_path+'Fig12_n_List.npy')
    AvgTimes = dict()
    AllErrors = dict()
    AvgErrors = dict()
    Variances = dict()
    for key in ToCompare:
        AvgTimes[key] = np.load(data_path+'Fig12_AvgTimes_vec{}{}_{}_Ex{}.npy'.format(SampVecNr,PopVecNr,key,ExampleNumber))
        AvgErrors[key] = np.load(data_path+'Fig12_AvgErrors_vec{}{}_{}_Ex{}.npy'.format(SampVecNr,PopVecNr,key,ExampleNumber))
        AllErrors[key] = np.load(data_path+'Fig12_AllErrors_vec{}{}_{}_Ex{}.npy'.format(SampVecNr,PopVecNr,key,ExampleNumber))
        Variances[key] = np.load(data_path+'Fig12_Variances_vec{}{}_{}_Ex{}.npy'.format(SampVecNr,PopVecNr,key,ExampleNumber))
    RejectedIndices = np.load(data_path+'Fig12_RejectedIndices_vec{}{}_Ex{}.npy'.format(SampVecNr,PopVecNr,ExampleNumber),allow_pickle=True).item()
except:
    print('#############################')
    print("Error: could not load the results")
    print("try running Figure_12_Preparation.R and then Figure_12_Calculations.py first")
    print('#############################')
    assert 0==1


print([len(RejectedIndices[k]) for k in range(len(n_List))])

ax.plot(n_List,[0]*len(n_List),color='gray',alpha=0.3)

#ax[0].set_ylim([-0.05*ymax0, ymax0])
#ax[1].set_ylim([-0.05*ymax1, ymax1])

#print(AvgErrors['MPI'])

for i in range(len(ToCompare)):
    key = ToCompare[i]
    AllError = AllErrors[key]
    for j in range(len(n_List)):
        Errors = AllError[j,:]
        Errors = Errors[~np.isnan(Errors)]
        if len(Errors)>0:
            mean = Errors.mean()
            lower = np.quantile(Errors, 0.1)
            upper = np.quantile(Errors, 0.9)
            lower_err = mean - lower
            upper_err = upper - mean

            if j==0:
                ax.errorbar([n_List[j] + offsets[i]],mean,yerr=[[lower_err], [upper_err]],fmt='o',capsize=4,color=ColorMap[key],label=LegendMap[key])
            else:
                ax.errorbar([n_List[j] + offsets[i]],mean,yerr=[[lower_err], [upper_err]],fmt='o',capsize=4,color=ColorMap[key])

AllError = AllErrors['LedoitWolf']
for j in range(len(n_List)):
    Errors = AllError[j,:]
    Errors = Errors[RejectedIndices[j]]
    if len(Errors)>0:
        mean = Errors.mean()
        lower = np.quantile(Errors, 0.1)
        upper = np.quantile(Errors, 0.9)
        lower_err = mean - lower
        upper_err = upper - mean
        if j==0:
            ax.text(n_List[j] + offsets[-1]+21,mean,str(len(RejectedIndices[j])),fontsize=14, ha="center", va="bottom",color='red')
            ax.errorbar([n_List[j] + offsets[-1]],mean,yerr=[[lower_err], [upper_err]],fmt='o',capsize=4,color='red',label='LW of rejected')
            #ax.scatter([n_List[j] + offsets[-1]]*len(Errors),Errors,s=16,color='red',label='LW rejected')
        else:
            ax.text(n_List[j] + offsets[-1]+14,mean,str(len(RejectedIndices[j])),fontsize=14, ha="center", va="bottom",color='red')
            ax.errorbar([n_List[j] + offsets[-1]],mean,yerr=[[lower_err], [upper_err]],fmt='o',capsize=4,color='red')
            #ax.scatter([n_List[j] + offsets[-1]]*len(Errors),Errors,s=16,color='red')
    

ax.grid(True, alpha=0.3)
ax.legend(loc="upper right")
ax.semilogy()

plt.show()

    

