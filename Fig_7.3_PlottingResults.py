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
LogarithmicPlot=True


if 0: #Figure 7a
    c = 1/10
    ExampleNumber=1
    ToCompare = {'LedoitWolf', 'ElKaroui','MPI','KongValiant'}
    g_name='monomial7'
    ymax0,ymax1 = 2,0.5
    titleInsert = r'$z^{7}$ and example 1'
if 0: #Figure 7b
    c = 2
    ExampleNumber=2
    ToCompare = {'LedoitWolf', 'ElKaroui','MPI','KongValiant'}
    #ToCompare = {'LedoitWolf', 'ElKaroui','MPI'}
    g_name='monomial5'
    ymax0,ymax1 = 2,0.5
    titleInsert = r'$z^{5}$ and example 2'
if 1: #Figure 7c
    c = 1/4
    ExampleNumber=3
    ToCompare = {'LedoitWolf', 'ElKaroui','MPI','KongValiant'}
    g_name='monomial3'
    ymax0,ymax1 = 2,0.5
    titleInsert = r'$z^{3}$ and example 3'

###################
    
ColorMap = {'LedoitWolf': 'black', 'ElKaroui': 'blue', 'KongValiant': 'purple', 'MPI': 'orange', 'MPI_unrestricted': 'red'}
LegendMap = {'LedoitWolf': 'Ledoit-Wolf', 'ElKaroui': 'El Karoui', 'KongValiant': 'Kong-Valiant', 'MPI': 'new method'}

fig, ax = plt.subplots(1,3,layout='constrained', figsize=(16, 4))

ax[0].set_title(r'Estimation error for g(z) = {}'.format(titleInsert))
ax[1].set_title(r'Empirical variance for g(z) = {}'.format(titleInsert))
ax[2].set_title(r'Time in seconds for g(z) = {}'.format(titleInsert))

try:
    d_List = np.load(data_path+'Fig7_d_List.npy')
    AvgTimes = dict()
    AvgErrors = dict()
    Variances = dict()
    for key in ToCompare:
        AvgTimes[key] = np.load(data_path+'Fig7_AvgTimes_{}_c={}_{}_Ex{}.npy'.format(key,c,g_name,ExampleNumber))
        AvgErrors[key] = np.load(data_path+'Fig7_AvgErrors_{}_c={}_{}_Ex{}.npy'.format(key,c,g_name,ExampleNumber))
        Variances[key] = np.load(data_path+'Fig7_Variances_{}_c={}_{}_Ex{}.npy'.format(key,c,g_name,ExampleNumber))
except:
    print('#############################')
    print("Error: could not load the results")
    print("try running Fig_7_Preparation.R and then Fig_7_Calculations.py with the same choices for c and ExampleNumber first")
    print('#############################')
    assert 0==1

ax[0].plot(d_List,[0]*len(d_List),color='gray',alpha=0.3)
ax[1].plot(d_List,[0]*len(d_List),color='gray',alpha=0.3)
ax[2].plot(d_List,[0]*len(d_List),color='gray',alpha=0.3)

#ax[0].set_ylim([-0.05*ymax0, ymax0])
#ax[1].set_ylim([-0.05*ymax1, ymax1])

#print(AvgErrors['MPI'])

for key in ToCompare:
    ax[0].plot(d_List,AvgErrors[key],color=ColorMap[key],alpha=0.5,linewidth=4,label=LegendMap[key])
    ax[1].plot(d_List,Variances[key],color=ColorMap[key],alpha=0.5,linewidth=4,label=LegendMap[key])
    ax[2].semilogy(d_List,AvgTimes[key],color=ColorMap[key],alpha=0.5,linewidth=4,label=LegendMap[key])
    #fitting of C/n-curves
    #Ignore = 2
    #C1 = sum([y/x for x,y in zip(d_List[Ignore:],AvgErrors[key][Ignore:])])/sum([1/x**2 for x,y in zip(d_List[Ignore:],AvgErrors[key][Ignore:])])
    #ax[0].plot(d_List,[C1/d for d in d_List],color='black',linestyle='dashed',alpha=0.3)


if LogarithmicPlot:
    ax[0].semilogy()
    ax[1].semilogy()


for i in range(3):
	ax[i].grid(True, alpha=0.3)
	ax[i].legend(loc="upper right")

plt.show()

    

