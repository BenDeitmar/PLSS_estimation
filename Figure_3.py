import numpy as np
from math import ceil, log, pi
import matplotlib.pyplot as plt


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

#######################
##upper plots
#d = 20

##lower plots
d = 50
#######################


n = d
c = d/n
N = ceil(log(d))

PopEV1 = np.array([0.5+i/2/d for i in range(d)])
nodes, w = np.polynomial.legendre.leggauss(N)
nodes, w = (nodes+1)/4+0.5, w/2
print(sum(w))
counts = apportion_counts(d,w)
PopEV2 = []
for j in range(N):
    PopEV2 += [nodes[j]]*counts[j]
PopEV2 = np.array(PopEV2)

GrindPoints = 500
x_0 = np.linspace(0,1.5,GrindPoints)
xMin,xMax,eta = 0,3.5,0.001
x = np.linspace(xMin,xMax,GrindPoints)

CDF1 = np.array([sum([1 for lam in PopEV1 if pos >= lam]) for pos in x_0])/d
CDF2 = np.array([sum([1 for lam in PopEV2 if pos >= lam]) for pos in x_0])/d

yNu1 = getNu(PopEV1,c,x,eta)
yNu2 = getNu(PopEV2,c,x,eta)
yNu2 = getNu_Theoretical(nodes,w,c,x,eta)

NuCDF1 = np.cumsum(yNu1)/GrindPoints*(xMax-xMin)
NuCDF2 = np.cumsum(yNu2)/GrindPoints*(xMax-xMin)

X = np.random.normal(size=(d,n))

T1 = np.diag(np.sqrt(PopEV1))
S1 = T1@X@X.T@T1/n
T2 = np.diag(np.sqrt(PopEV2))
S2 = T2@X@X.T@T2/n

SampEV1,_ = np.linalg.eigh(S1)
SampEV2,_ = np.linalg.eigh(S2)

SampCDF1 = np.array([sum([1 for lam in SampEV1 if pos >= lam]) for pos in x])/d
SampCDF2 = np.array([sum([1 for lam in SampEV2 if pos >= lam]) for pos in x])/d

fig, ax = plt.subplots(1,3,layout='constrained', figsize=(16, 4))

ax[0].set_title(r"CDF of $H_n^{(1)}$ and $H_n^{(2)}$"+rf" for d={d} and n={n}")
ax[0].plot(x_0,CDF1,alpha=0.6,linewidth=3,label=r'$F_{H_n^{(1)}}$')
ax[0].plot(x_0,CDF2,alpha=0.6,linewidth=3,label=r'$F_{H_n^{(2)}}$')

ax[1].set_title(r"CDF of $\nu_n^{(1)}$ and $\nu_n^{(2)}$"+rf" for d={d} and n={n}")
ax[1].plot(x,NuCDF1,alpha=0.6,linewidth=3,label=r'$F_{\nu_n^{(1)}}$')
ax[1].plot(x,NuCDF2,alpha=0.6,linewidth=3,label=r'$F_{\nu_n^{(2)}}$')

ax[2].set_title(r"CDF of $\hat{\nu}_n^{(1)}$ and $\hat{\nu}_n^{(2)}$"+rf" for d={d} and n={n}")
ax[2].plot(x,SampCDF1,alpha=0.6,linewidth=3,label=r'$F_{\hat{\nu}_n^{(1)}}$')
ax[2].plot(x,SampCDF2,alpha=0.6,linewidth=3,label=r'$F_{\hat{\nu}_n^{(2)}}$')

for i in range(3):
    ax[i].grid(True, alpha=0.3)
    ax[i].legend(loc="upper right")
plt.show()