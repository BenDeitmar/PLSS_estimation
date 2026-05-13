import numpy as np
from math import ceil, log, pi
import matplotlib.pyplot as plt

from typing import Sequence
from scipy import integrate, optimize


def sup_cdf_diff_on_interval_fast(f, g, a=-1, b=4.0, n=500) -> float:
    """
    Fast, low-precision approximation of sup_{x in [a,b]} |F(x)-G(x)| given
    vectorized PDFs f and g.

    f, g: callables that accept a NumPy array and return a NumPy array.
    """
    x = np.linspace(a, b, int(n))
    d = np.asarray(f(x), dtype=float) - np.asarray(g(x), dtype=float)

    # cumulative trapezoid: H[k] = integral_a^{x[k]} d(t) dt
    dx = (b - a) / (len(x) - 1)
    H = np.empty_like(d)
    H[0] = 0.0
    H[1:] = np.cumsum(0.5 * (d[:-1] + d[1:]) * dx)

    return float(np.max(np.abs(H)))


def ks_supremum_edf_diff(X: Sequence[float], Y: Sequence[float]) -> float:
    n, m = X.size, Y.size

    # Sort samples
    Xs = np.sort(X)
    Ys = np.sort(Y)

    # Two-pointer scan over the merged sorted unique values
    i = j = 0
    Fx = Fy = 0.0
    prev_absdiff = 0.0
    sup = 0.0

    while i < n or j < m:
        # Next value in the merged order
        if j == m or (i < n and Xs[i] <= Ys[j]):
            v = Xs[i]
        else:
            v = Ys[j]

        # Value just before processing jumps at v (left-limit)
        absdiff_left = abs(Fx - Fy)
        if absdiff_left > sup:
            sup = absdiff_left

        # Consume all ties at v in each sample (the EDF jumps here)
        if i < n and Xs[i] == v:
            ii = i
            while ii < n and Xs[ii] == v:
                ii += 1
            Fx += (ii - i) / n
            i = ii

        if j < m and Ys[j] == v:
            jj = j
            while jj < m and Ys[jj] == v:
                jj += 1
            Fy += (jj - j) / m
            j = jj

        absdiff_right = abs(Fx - Fy)
        if absdiff_right > sup:
            sup = absdiff_right

        prev_absdiff = absdiff_right

    return float(sup)

def apportion_counts(n, weights):
    ws = weights / weights.sum()
    quotas = n * ws
    counts = np.floor(quotas).astype(int)
    r = n - counts.sum()
    remainders = quotas - counts
    idx = np.argsort(remainders)[::-1][:r]
    counts[idx] += 1
    return counts

def getNu(PopEV,c,x,eta,maxIt=200):
    print('MP solving',len(x))
    z = x+eta*1j
    s = np.zeros_like(z)+1j
    for i in range(maxIt):
        s = np.sum(1/(PopEV[np.newaxis,:]*(1-c*z[:,np.newaxis]*s[:,np.newaxis]-c)-z[:,np.newaxis]),axis=1)/d
    return np.imag(s)/pi

def getNu_Theoretical(nodes,w,c,x,eta,maxIt=200):
    print('MP solving',len(x))
    z = x+eta*1j
    s = np.zeros_like(z)+1j
    for i in range(maxIt):
        s = np.sum(w[np.newaxis,:]/(nodes[np.newaxis,:]*(1-c*z[:,np.newaxis]*s[:,np.newaxis]-c)-z[:,np.newaxis]),axis=1)
    return np.imag(s)/pi

d_List = [50*i for i in range(1,11)]
NN=10
supNuCDF_List = []
supSampCDF_List = []

for d in d_List:
    print(d)
    n = d
    c = d/n
    N = ceil(log(d))

    PopEV1 = np.array([0.5+i/2/d for i in range(d)])
    nodes, w = np.polynomial.legendre.leggauss(N)
    nodes, w = (nodes+1)/4+0.5, w/2
    counts = apportion_counts(d,w)
    PopEV2 = []
    for j in range(N):
        PopEV2 += [nodes[j]]*counts[j]
    PopEV2 = np.array(PopEV2)

    eta = 0.001

    Nu1DensityFunction = lambda x: getNu(PopEV1,c,x,eta,maxIt=100)
    Nu2DensityFunction = lambda x: getNu(PopEV2,c,x,eta,maxIt=100)
    #Nu2DensityFunction = lambda x: getNu_Theoretical(nodes,w,c,x,eta,maxIt=100)

    supNuCDF = sup_cdf_diff_on_interval_fast(Nu1DensityFunction, Nu2DensityFunction, n=1000)
    supNuCDF_List.append(supNuCDF)

    X = np.random.normal(size=(d,n))

    T1 = np.diag(np.sqrt(PopEV1))
    S1 = T1@X@X.T@T1/n
    T2 = np.diag(np.sqrt(PopEV2))
    S2 = T2@X@X.T@T2/n

    SampEV1,_ = np.linalg.eigh(S1)
    SampEV2,_ = np.linalg.eigh(S2)

    #SampCDF1 = np.array([sum([1 for lam in SampEV1 if pos >= lam]) for pos in x])/d
    #SampCDF2 = np.array([sum([1 for lam in SampEV2 if pos >= lam]) for pos in x])/d

    supSampCDF = ks_supremum_edf_diff(SampEV1,SampEV2)
    supSampCDF_List.append(supSampCDF)

fig, ax = plt.subplots(1,2,layout='constrained', figsize=(16, 4))

ax[0].set_title(r"Difference in CDFs of $\nu_n^{(1)}$ and $\nu_n^{(2)}$")
ax[0].plot(d_List,supNuCDF_List,alpha=1,linewidth=4,label=r"$||F_{\nu_n^{(1)}} - F_{\nu_n^{(2)}}||_\infty$")

Ignore = 2
C1 = sum([y/x for x,y in zip(d_List[Ignore:],supNuCDF_List[Ignore:])])/sum([1/x**2 for x,y in zip(d_List[Ignore:],supNuCDF_List[Ignore:])])
ax[0].plot(d_List,[C1/d for d in d_List],color='black',linestyle='dashed',alpha=0.5,linewidth=2,label=r"fitted $d \mapsto \frac{C}{d}$")

ax[1].set_title(r"Difference in CDFs of $\hat{\nu}_n^{(1)}$ and $\hat{\nu}_n^{(2)}$")
ax[1].plot(d_List,supSampCDF_List,alpha=1,linewidth=4,label=r"$||F_{\hat{\nu}_n^{(1)}} - F_{\hat{\nu}_n^{(2)}}||_\infty$")

Ignore = 2
C1 = sum([y/x for x,y in zip(d_List[Ignore:],supSampCDF_List[Ignore:])])/sum([1/x**2 for x,y in zip(d_List[Ignore:],supSampCDF_List[Ignore:])])
ax[1].plot(d_List,[C1/d for d in d_List],color='black',linestyle='dashed',alpha=0.5,linewidth=2,label=r"fitted $d \mapsto \frac{C}{d}$")

for i in range(2):
    ax[i].grid(True, alpha=0.3)
    ax[i].legend(loc="upper right")
    
plt.show()