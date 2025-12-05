import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, colors
from math import ceil
from cmath import exp, log
import os
import pandas as pd
import cvxopt
import time
from scipy.optimize import curve_fit
data_path = ''.join(map(lambda string: string+'\\', os.path.realpath(__file__).split('\\')[:-1]))+'data/'


def power_law_model(x, C, alpha):
    return C * x**alpha

def fit_power_law(dL, fL, alpha_bounds=(-np.inf, np.inf)):
    """
    Fit f(x) ≈ C * x^alpha to data, with C>0 and alpha in alpha_bounds.

    Parameters
    ----------
    dL : array-like
        x-values
    fL : array-like
        f(x)-values
    alpha_bounds : tuple (alpha_min, alpha_max), optional
        Bounds for alpha. Use (-np.inf, 0) if you know alpha is negative.

    Returns
    -------
    C_hat : float
    alpha_hat : float
    """
    x = np.asarray(dL, dtype=float)
    y = np.asarray(fL, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size < 2:
        raise ValueError("Need at least two data points to fit.")

    # Initial guess using log–log fit on positive points
    pos = (x > 0) & (y > 0)
    if pos.sum() >= 2:
        X = np.log(x[pos])
        Y = np.log(y[pos])
        a, b = np.polyfit(X, Y, 1)
        C0 = float(np.exp(b))
        alpha0 = float(a)
        if not np.isfinite(C0) or C0 <= 0:
            C0 = 1.0
        if not np.isfinite(alpha0):
            alpha0 = 0.0
    else:
        C0, alpha0 = 1.0, 0.0

    # Bounds: C >= 0, alpha between alpha_bounds
    (amin, amax) = alpha_bounds
    popt, pcov = curve_fit(
        power_law_model,
        x, y,
        p0=[C0, alpha0],
        bounds=([0.0, amin], [np.inf, amax]),
        maxfev=10000
    )

    C_hat, alpha_hat = popt
    return C_hat, alpha_hat


###################
c = 2
ExampleNumber=2
ToCompare = ['LedoitWolf', 'ElKaroui']

#Figure 6a
if 1:
    g_name='Stieltjes_a'
    ymax = 0.75
    i=1
    FitCurve='custom'
#Figure 6b
if 0:
    g_name='Stieltjes_b'
    ymax = 0.025
    i=2
    FitCurve='custom'
#Figure 6c
if 0:
    g_name='Stieltjes_c'
    ToCompare = ['LedoitWolf', 'ElKaroui','MPI']
    ymax = 0.0025
    i=3
    FitCurve='custom'

###################
    
ColorMap = {'LedoitWolf': 'black', 'ElKaroui': 'blue', 'KongValiant': 'purple', 'MPI': 'orange'}
LegendMap = {'LedoitWolf': 'Ledoit-Wolf', 'ElKaroui': 'El Karoui', 'KongValiant': 'Kong-Valiant', 'MPI': 'new method'}

fig, ax = plt.subplots(1,layout='constrained', figsize=(8, 5.6))

offsets = np.linspace(-5, 5, len(ToCompare))

ax.set_title('Estimation error for z{}'.format(i))

try:
    d_List = np.load(data_path+'Fig6_d_List.npy')
    AvgTimes = dict()
    AllErrors = dict()
    Variances = dict()
    for key in ToCompare:
        AvgTimes[key] = np.load(data_path+'Fig6_AvgTimes_{}_c={}_{}_Ex{}.npy'.format(key,c,g_name,ExampleNumber))
        AllErrors[key] = np.load(data_path+'Fig6_AllErrors_{}_c={}_{}_Ex{}.npy'.format(key,c,g_name,ExampleNumber))
        Variances[key] = np.load(data_path+'Fig6_Variances_{}_c={}_{}_Ex{}.npy'.format(key,c,g_name,ExampleNumber))
except:
    print('#############################')
    print("Error: could not load the results")
    print("try running Fig_6_Preparation.R and then Fig_6_Calculations.py with the same choices for c and ExampleNumber first")
    print('#############################')
    assert 0==1

ax.plot(d_List,[0]*len(d_List),color='gray',alpha=0.3)

#ax[0].set_ylim([-0.05*ymax0, ymax0])
#ax[1].set_ylim([-0.05*ymax1, ymax1])


for i in range(len(ToCompare)):
    key = ToCompare[i]
    Errors = AllErrors[key]
    mean = Errors.mean(axis=1)
    lower = np.quantile(Errors, 0.1, axis=1)
    upper = np.quantile(Errors, 0.9, axis=1)
    lower_err = mean - lower
    upper_err = upper - mean

    ax.errorbar(d_List + offsets[i],mean,yerr=[lower_err, upper_err],fmt='o',capsize=4,color=ColorMap[key],label=LegendMap[key])
    if FitCurve=='custom':
        Ignore=0
        if key!='ElKaroui':
            C,alpha = fit_power_law(d_List[Ignore:],mean[Ignore:])
            ax.plot(d_List,[C*d**(alpha) for d in d_List],color=ColorMap[key],linestyle='dashed',alpha=0.3,label=r'fitted $\frac{C}{d^{\alpha}}, \alpha=$'+'{}'.format(np.round(-alpha,decimals=2)))

    if FitCurve=='n': #fitting of C/n-curves
        Ignore = 2
        C1 = sum([y/x for x,y in zip(d_List[Ignore:],mean[Ignore:])])/sum([1/x**2 for x,y in zip(d_List[Ignore:],mean[Ignore:])])
        ax.plot(d_List,[C1/d for d in d_List],color='black',linestyle='dashed',alpha=0.3)
    if FitCurve=='log': #fitting of C/log(n)-curves
        Ignore = 2
        C1 = sum([y/log(x) for x,y in zip(d_List[Ignore:],mean[Ignore:])])/sum([1/log(x)**2 for x,y in zip(d_List[Ignore:],mean[Ignore:])])
        ax.plot(d_List,[C1/log(d) for d in d_List],color='black',linestyle='dashed',alpha=0.3)

if FitCurve=='n':
    ax.plot([],[],color='black',linestyle='dashed',alpha=0.3,label=r'fitted $\frac{C}{d}$-curves')
if FitCurve=='log':
    ax.plot([],[],color='black',linestyle='dashed',alpha=0.3,label=r'fitted $\frac{C}{\log(d)}$-curves')

#upper = ax.get_ylim()[1]
#ax.set_ylim(0, ymax)
ax.semilogy()

plt.grid(True, alpha=0.3)
plt.legend(loc="upper right")

plt.show()

    

