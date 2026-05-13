import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import ceil, factorial
from functools import lru_cache
from tensorflow.keras.datasets import mnist as keras_mnist
from matplotlib.colors import LogNorm
import cvxopt

from Algorithms import Vectorwise_StieltjesTransformEstimator, CurveDiscoveryHSpace, CurveDiscoveryNuSpace, IntegrationNodes, findSupp_nu

######################
#helper functions

def pullHaarOd(d):
    X = np.random.normal(size=(d,d))
    _,V = np.linalg.eigh(X@X.T)
    return V

def getMNIST_data(d,ChosenDims=None):
    (x_train, _), (x_test, _) = keras_mnist.load_data()
    data = np.concatenate([x_train.reshape(-1, 28*28), x_test.reshape(-1, 28*28)], axis=0).T/255.0
    if ChosenDims is None:
        ChosenDims = np.random.choice(data.shape[0], size=d, replace=False)
    else:
    	ChosenDims = ChosenDims-1
    return data[ChosenDims,:]

def getMNIST_data_full():
    (x_train, _), (x_test, _) = keras_mnist.load_data()
    data = np.concatenate([x_train.reshape(-1, 28*28), x_test.reshape(-1, 28*28)], axis=0).T/255.0
    X_full = data-np.sum(data,axis=1)[:,np.newaxis]/data.shape[1]
    return X_full

def PopEV_Y_maker(d,n,ExampleNumber,ChosenDims=None,returnSigma=False,multiplicity=1):
    if ExampleNumber==1:
        PopEV = np.array([1 for i in range(d//2)]+[2.5+i/d for i in range(d-d//2)])
        Y = np.diag(np.sqrt(PopEV))@np.random.normal(size=(d,n))
        if returnSigma:
            return np.diag(PopEV), Y
        else:
            return PopEV, Y

    if ExampleNumber==2:
        U = np.matrix(pullHaarOd(d))
        PopEV = np.array([0.5+i/d/2 for i in range(d)])
        Y = U@np.diag(np.sqrt(PopEV))@np.random.choice([-1,1],size=(d,n))
        if returnSigma:
            return U@np.diag(PopEV)@U.H, Y
        else:
            return PopEV, Y

    if ExampleNumber==3:
        data = getMNIST_data(d,ChosenDims=ChosenDims)
        #print(data.shape)
        X_full = data-np.sum(data,axis=1)[:,np.newaxis]/data.shape[1]
        Sigma_surrogate = X_full@X_full.T/data.shape[1]
        PopEV_surrogate,_ = np.linalg.eigh(Sigma_surrogate)
        if multiplicity==1:
            ChosenSamp = np.random.choice(data.shape[1], size=n, replace=False)
            Y = X_full[:,ChosenSamp]
            if returnSigma:
                return Sigma_surrogate, Y
            else:
                return PopEV_surrogate, Y
        else:
            SampleList = []
            for i in range(multiplicity):
                ChosenSamp = np.random.choice(data.shape[1], size=n, replace=False)
                Y = X_full[:,ChosenSamp]
                SampleList.append(Y)
            return PopEV_surrogate,SampleList

    if ExampleNumber==4:
        U = pullHaarOd(d)
        PopEV = np.array([0.5 for i in range(d//2)]+[0.5+i/d for i in range(d-d//2)])
        Y = U@np.diag(np.sqrt(PopEV))@np.random.choice([-1,1],size=(d,n))
        if returnSigma:
            return U@np.diag(PopEV)@U.H, Y
        else:
            return PopEV, Y

def plot_contour_field(
    X, Y, F, ax=None,
    levels=20, cmap_name='Blues',
    vmin=None, vmax=None,
    title=r"Number of iterations until convergence",
    show=True,
    LogScale=False
):
    # figure & axes
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7.8, 5.8))
    ax.set_facecolor("white")
    ax.set_title(title)

    # colormap with transparent NaNs
    try:
        cmap = mpl.colormaps[cmap_name].with_extremes(bad=(1, 1, 1, 0.0))
    except Exception:
        cmap = plt.cm.get_cmap(cmap_name).copy()
        cmap.set_bad((1, 1, 1, 0.0))

    # mask NaNs so they render as transparent/white
    Fm = np.ma.masked_invalid(F)

    if LogScale:
        norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
        a = int(np.ceil(np.log10(vmax))-np.floor(np.log10(vmin)))
        levels = np.logspace(np.floor(np.log10(vmin)), np.ceil(np.log10(vmax)), num=a*10)
        cf = ax.contourf(
            X, Y, Fm,
            levels=levels,
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            antialiased=False,
            norm=norm
        )
        cbar = plt.colorbar(cf, ax=ax)
        exp_min = int(np.floor(np.log10(vmin)))
        exp_max = int(np.ceil(np.log10(vmax)))
        exponents = [-0,-1,-2,-3,-4]
        ticks = [10**e for e in exponents]
        cbar.set_ticks(ticks)
    else:
        cf = ax.contourf(
            X, Y, Fm,
            levels=levels,
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            antialiased=False
        )
        cbar = plt.colorbar(cf, ax=ax, pad=0.02, fraction=0.05)

    # labels & style
    ax.set_xlabel(r"Re($z$)")
    ax.set_ylabel(r"Im($z$)")
    ax.set_facecolor("white")
    ax.grid(False)

    if show:
        plt.tight_layout()
        plt.show()


def _blue_cmap():
    try:
        return mpl.colormaps['Blues'].with_extremes(bad=(1, 1, 1, 0.0))
    except Exception:
        cmap = plt.cm.get_cmap('Blues').copy()
        cmap.set_bad((1, 1, 1, 0.0))
        return cmap

def plot_blue_region(mask, x_lo, x_hi, y_lo, y_hi, ax,
                     alpha=0.45,
                     xlabel=r"Re($z$)", ylabel=r"Im($z$)",
                     show=True,
                     red_evals=None,  # optional: array of sample eigenvalues to plot (red dots on real axis)
                     red_size=16):
    mask = np.asarray(mask, dtype=bool)
    show_arr = np.where(mask, 1.0, np.nan)  # NaN → transparent
    cmap = _blue_cmap()

    

    ax.imshow(show_arr, extent=[x_lo, x_hi, y_lo, y_hi],
              origin='lower', aspect='auto',
              cmap=cmap, vmin=0, vmax=1, alpha=alpha)

    if red_evals is not None:
        red_evals = np.asarray(red_evals, dtype=float)
        ax.scatter(red_evals, np.zeros_like(red_evals),
                   color="red", s=red_size, edgecolor="none", label="Sample eigenvalues")
        ax.legend(loc="upper right")
        #ax.legend(loc="upper right", frameon=False)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)


#Algorithm to find the boundary \bD_{H_n,c_n}(\epsilon,\theta)
def DBoundary(epsilon,theta,PopEV,c,xMin,etaMax,xMax,stepSize=0.005,iterationDepth=10):
    d = len(PopEV)
    Stil = lambda z : sum([1/(lam-z) for lam in PopEV])/d
    N = 1000#ceil((xMax-xMin)/stepSize)
    xList = [xMin+(xMax-xMin)*i/N for i in range(N)]
    bList = []
    if True:
        J = 0
        for x in xList:
            J+=1
            LastNeg = 0
            LastPos = etaMax
            FoundBoundary = False
            for i in range(iterationDepth):
                eta = (LastPos+LastNeg)/2
                z = x+1j*eta
                if np.imag((1-c-c*z*Stil(z))*z) > epsilon and c*abs(z)*np.imag(z*Stil(z))/np.imag((1-c-c*z*Stil(z))*z) < theta:
                    LastPos = eta
                else:
                    LastNeg = eta
                    FoundBoundary = True
            if not FoundBoundary:
                LastPos = 0
            bList.append(LastPos)
    return(xList,bList)

#Calculate Stieltjes transform of discrete measure vectorwise
def Vectorwise_Stieltjes(z,PopEV):
    PopVec = np.array(PopEV)[np.newaxis]
    return np.sum(1/(PopVec.T-z),axis=0)/d

#Algorithm 1 (vectorwise in z and counting iterations)
def VectorwiseStepcounting_MPI(z,SampEV,c,tau=0,kappa=np.inf,maxIteratoions=100,tol=10**(-9)):
    assert (np.imag(z)>0).all()
    d = len(SampEV)
    lastV = np.zeros(len(z))
    zVec = z[np.newaxis]
    SampVec = np.array(SampEV)[np.newaxis]
    V = lastV+1j#np.sum(1/(SampVec.T-zVec),axis=0)/d
    Iterations = 0
    ConvergedEntries = np.zeros(len(z),dtype=np.intc)
    IterationsUntilConvergence = np.zeros(len(z))*np.nan
    while Iterations < maxIteratoions:
        lastV = V
        V = np.sum(SampVec.T/(SampVec.T-(1-c*V)*zVec),axis=0)/d
        ConvergedNow = 1*(abs(V-lastV)<tol)
        IterationsUntilConvergence[np.logical_and(ConvergedNow,np.logical_not(ConvergedEntries))] = Iterations
        ConvergedEntries = ConvergedNow
        Iterations += 1
    s = (V-1)/z
    s[np.isnan(IterationsUntilConvergence)]=np.nan
    s[np.imag((1-c*V)*z)<=0]=np.nan
    s[np.abs(c*z*np.imag(V)/np.imag((1-c*V)*z))>=1]=np.nan
    s[np.min(np.abs(SampEV[:,np.newaxis]-((1-c*V)*z)[np.newaxis,:]),axis=0)<=tau]=np.nan
    s[np.abs((1-c*V)*z)>=kappa]=np.nan
    s[np.abs((1-c*V)*z)<tau]=np.nan
    IterationsUntilConvergence[np.isnan(s)]=np.nan
    return s,IterationsUntilConvergence

######################




# CurveMode 0 -> invisible ; 1 -> curves drawn as lines ; 2 -> scatter plot of the legendre nodes ; 3 -> only one large curve drawn
def VisualizeHSpace(Y,tau=0,kappa=np.inf,resolution=1/100,CurveMode=1,ax=None,show=True,PopEV=None,greenLines=True,Interval=None,theta=1,doCondition_2_13=False):
    Y = np.matrix(Y)
    d,n = Y.shape
    c = d/n
    S = Y@Y.H/n
    SampEV,_ = np.linalg.eigh(S)

    #Use automatic curve dicovery to find the region of interest
    CurveList = np.array(CurveDiscoveryHSpace(SampEV,c,tau=tau,kappa=kappa,doCondition_2_13=doCondition_2_13,PopEV=PopEV))
    LeftEdges, Heights, RightEdges = CurveList[:,0], CurveList[:,1], CurveList[:,2]
    showLeft, showHeight, showRight = min(LeftEdges), max(Heights), max(RightEdges)
    showLeft, showHeight, showRight = showLeft-0.2*(showRight-showLeft), showHeight*1.3, showRight+0.2*(showRight-showLeft)
    
    #create grid
    x = np.arange(showLeft, showRight, resolution)
    y = np.arange(resolution/2, showHeight, resolution/2)
    X, Y = np.meshgrid(x, y)
    Z = X+1j*Y
    orig_shape = Z.shape
    z = Z.ravel()

    print('Number of z for which the population Stieltjes transform estimator must be found:',len(z))
    print('If this number is too large (>100.000), try increasing the resolution-parameter')

    #find the support of nu, if needed for condition (2.13)
    if doCondition_2_13:
        supp_nu = findSupp_nu(SampEV,PopEV,c)
    else:
        supp_nu = None

    #calculate Stieltjes transform estimators
    s = Vectorwise_StieltjesTransformEstimator(z,SampEV,c,tau=tau,kappa=kappa,theta=theta,doCondition_2_13=doCondition_2_13,supp_nu=supp_nu)
    s = s.reshape(orig_shape)
    BlueRegion = (s!=0)

    #apply condition (2.13), if wanted and PopEV available
    if False and doCondition_2_13 and not (PopEV is None):
        #find support of \nu_n
        x,h = DBoundary(0,np.inf,PopEV,c,showLeft,showHeight,showRight)
        z_Boundary = np.array(x)+1j*np.array(h)
        z_Boundary = z_Boundary[np.imag(z_Boundary)>0]
        PopStil = np.sum(1/(PopEV[np.newaxis,:]-z_Boundary[:,np.newaxis]),axis=1)/d
        Phi = (1-c*z_Boundary*PopStil-c)*z_Boundary
        supp_Points = np.real(Phi)

        #find \hat{Phi}_n and apply condition
        zShape = z.reshape(orig_shape)
        hatPhi = (1-c*zShape*s-c)*zShape

        ConditionSet = (np.all(np.abs(hatPhi[..., None] - supp_Points) > tau, axis=-1))
        BlueRegion = np.logical_and(BlueRegion, ConditionSet)



    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5.6))
    ax.set_facecolor("white")
    ax.set_title(rf"$D$(τ,κ,n) in $H$-space for τ={tau}, κ={kappa}")

    ax.plot([showLeft,showRight],[0,0],color='gray',zorder=0)
    plot_blue_region(BlueRegion, showLeft, showRight, resolution, showHeight, ax)
    #ax.scatter(SampEV,[0]*d,color='red',label="Sample eigenvalues")

    if CurveMode==1:
        for L,H,R in zip(LeftEdges, Heights, RightEdges):
            if not Interval is None and L<Interval[1] and R>Interval[0]:
                ax.plot([L,L,R,R],[0,H,H,0],color='red')
            else:
                ax.plot([L,L,R,R],[0,H,H,0],color='black')

    if CurveMode==2:
        for L,H,R in zip(LeftEdges, Heights, RightEdges):
            N = 2*ceil(sqrt(d))
            nodes,_ = IntegrationNodes(N,L,H,R)
            s_nodes = Vectorwise_StieltjesTransformEstimator(nodes,SampEV,c,tau=tau,kappa=kappa)
            Phi_nodes = (1-c*nodes*s_nodes-c)*nodes
            ax.scatter(np.real(nodes),np.imag(nodes),color='black')
            ax.scatter(np.real(Phi_nodes),np.imag(Phi_nodes),color='green')


    if CurveMode==3:
        L,H,R = LeftEdges[0],max(Heights),RightEdges[-1]
        ax.plot([L,L,R,R],[0,H,H,0],color='black')



    ax.set_xlim(showLeft, showRight)
    ax.set_ylim(-0.05*showHeight, showHeight)

    #Stil_UlNu = lambda z : np.sum(1/(SampEV-z))/n-(1-c)/z
    #varphi = lambda x: np.real(-1/Stil_UlNu(x+1j*0.01))
    #ax.scatter([varphi(lam) for lam in SampEV],[-0.025*showHeight]*d)

    if not PopEV is None:
        if greenLines:
            x,h = DBoundary(0,np.inf,PopEV,c,showLeft,showHeight,showRight)
            #print(showLeft,showRight)
            #print(h)
            ax.plot(x,h,color='green',zorder=3,alpha=0.5)
            x,h = DBoundary(0,1,PopEV,c,showLeft,showHeight,showRight)
            ax.plot(x,h,color='green',linestyle='dashed',alpha=0.5)
        ax.scatter(PopEV,[0]*d,color='orange', label=r'supp($H_n$)')
        #ax.legend(loc="upper right", frameon=False)
        ax.legend(loc="upper right")

    if show:
        plt.tight_layout()
        plt.show()

    return None


def VisualizeHNuSpace(Y,tau=0,kappa=np.inf,resolution=1/100,CurveMode=1,show=True,PopEV=None,greenLines=True):
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.6))
    VisualizeHSpace(Y,tau=tau,kappa=kappa,resolution=resolution,CurveMode=CurveMode,ax=ax[0],show=False,PopEV=PopEV,greenLines=greenLines)

    Y = np.matrix(Y)
    d,n = Y.shape
    c = d/n
    S = Y@Y.H/n
    SampEV,_ = np.linalg.eigh(S)

    CurveList = np.array(CurveDiscoveryNuSpace(SampEV,c,tau=tau,kappa=kappa))
    LeftEdges, Heights, RightEdges = CurveList[:,0], CurveList[:,1], CurveList[:,2]
    showLeft, showHeight, showRight = min(LeftEdges), max(Heights), max(RightEdges)
    showLeft, showHeight, showRight = showLeft-0.2*(showRight-showLeft), showHeight*1.3, showRight+0.2*(showRight-showLeft)

    #ax[1].set_facecolor("white")
    ax[1].set_title(rf"possible f-curves for τ={tau}")

    ax[1].plot([showLeft,showRight],[0,0],color='gray',zorder=0)
    #ax[1].scatter(np.real(Phi_nodes),np.imag(Phi_nodes),color='green',label="Phi(gamma_g)")

    for L,H,R in zip(LeftEdges, Heights, RightEdges):
        ax[1].plot([L,L,R,R],[0,H,H,0],color='black')

    ax[1].set_xlim(showLeft, showRight)
    ax[1].set_ylim(-0.05*showHeight, showHeight)

    ax[1].scatter(SampEV,[0]*d,color='red',label="Sample eigenvalues",zorder=3)
    #ax[1].legend(loc="upper right", frameon=False)
    ax[1].legend(loc="upper right")

    if show:
        plt.tight_layout()
        plt.show()

    return None



def ErrorContours(SampEV,c,PopEV,tau=0,kappa=np.inf,resolution=1/100):
    d = len(SampEV)
    CurveList = np.array(CurveDiscoveryHSpace(SampEV,c,tau=tau,kappa=kappa))
    LeftEdges, Heights, RightEdges = CurveList[:,0], CurveList[:,1], CurveList[:,2]
    showLeft, showHeight, showRight = min(LeftEdges), max(Heights), max(RightEdges)
    showLeft, showHeight, showRight = showLeft-0.2*(showRight-showLeft), showHeight*1.3, showRight+0.2*(showRight-showLeft)
    x = np.arange(showLeft, showRight, resolution)
    y = np.arange(resolution, showHeight, resolution/2)
    X, Y = np.meshgrid(x, y)
    Z = X+1j*Y
    orig_shape = Z.shape
    z = Z.ravel()
    print('Number of z for which the population Stieltjes transform estimator must be found:',len(z))
    print('If this number is too large (>100.000), try increasing the resolution-parameter')

    s = Vectorwise_StieltjesTransformEstimator(z,SampEV,c,tau=tau,kappa=kappa)
    s_true = np.sum(1/(PopEV[np.newaxis,:]-z[:,np.newaxis]),axis=1)/d
    Diff = np.abs(s-s_true)
    Diff[s==0] = np.nan
    s = s.reshape(orig_shape)
    Diff = Diff.reshape(orig_shape)

    fig, ax = plt.subplots(1, 1, figsize=(7.8, 5.8))
    plot_contour_field(X,Y,Diff,ax=ax,show=False,title=rf"Estimation error for d={d}",LogScale=True,vmin=10**(-5),vmax=10**0)

    x,h = DBoundary(0,np.inf,PopEV,c,showLeft,showHeight,showRight)
    ax.plot(x,h,color='green',zorder=3)
    x,h = DBoundary(0,1,PopEV,c,showLeft,showHeight,showRight)
    ax.plot(x,h,color='green',linestyle='dashed')
    ax.scatter(PopEV,[0]*d,color='orange', label=r'supp($H$)')
    ax.legend(loc="upper right")

    ax.set_xlim(showLeft, showRight)
    ax.set_ylim(-0.05*showHeight, showHeight)

    plt.tight_layout()
    plt.show()


def IterationContours(SampEV,c,tau=0,kappa=np.inf,resolution=1/100,PopEV=None):
    d = len(SampEV)
    CurveList = np.array(CurveDiscoveryHSpace(SampEV,c,tau=tau,kappa=kappa))
    LeftEdges, Heights, RightEdges = CurveList[:,0], CurveList[:,1], CurveList[:,2]
    showLeft, showHeight, showRight = min(LeftEdges), max(Heights), max(RightEdges)
    showLeft, showHeight, showRight = showLeft-0.2*(showRight-showLeft), showHeight*1.3, showRight+0.2*(showRight-showLeft)
    x = np.arange(showLeft, showRight, resolution)
    y = np.arange(resolution, showHeight, resolution/2)
    X, Y = np.meshgrid(x, y)
    Z = X+1j*Y
    orig_shape = Z.shape
    z = Z.ravel()
    print('Number of z for which the population Stieltjes transform estimator must be found:',len(z))
    print('If this number is too large (>100.000), try increasing the resolution-parameter')

    s,It = VectorwiseStepcounting_MPI(z,SampEV,c,tau=tau,kappa=kappa)
    s = s.reshape(orig_shape)
    It = It.reshape(orig_shape)

    fig, ax = plt.subplots(1, 1, figsize=(7.8, 5.8))
    plot_contour_field(X,Y,It,ax=ax,show=False,title=rf"#Iterations for τ={tau}, κ={kappa}",vmax=100)

    if not PopEV is None:
        x,h = DBoundary(0,np.inf,PopEV,c,showLeft,showHeight,showRight)
        #print(showLeft,showRight)
        #print(h)
        ax.plot(x,h,color='green',zorder=3)
        x,h = DBoundary(0,1,PopEV,c,showLeft,showHeight,showRight)
        ax.plot(x,h,color='green',linestyle='dashed')
        ax.scatter(PopEV,[0]*d,color='orange', label=r'supp($H$)')
        #ax.legend(loc="upper right", frameon=False)
        ax.legend(loc="upper right")

    ax.set_xlim(showLeft, showRight)
    ax.set_ylim(-0.05*showHeight, showHeight)

    plt.tight_layout()
    plt.show()







#########################################################################
#Implements Algorithms 1 from https://doi.org/10.1214/16-AOS1525

def choose(n,k):
    return factorial(n)//factorial(k)//factorial(n-k)

#Kong-Valiant method
def KongValiant_MomentEstimator(X,Moment):
    X = np.matrix(X)
    d,n = X.shape
    A = X.H@X
    G = A.copy()
    G[np.tril_indices(n)] = 0
    for k in range(Moment-1):
        A = G@A
    return np.trace(A)/d/choose(n,Moment)

#Kong-Valiant method all moments up to K
def KongValiant_MomentEstimator_MultipleMoments(X,K):
    X = np.matrix(X)
    d,n = X.shape
    A = X.H@X
    G = A.copy()
    G[np.tril_indices(n)] = 0
    MomentList = [1,np.trace(A)/d/choose(n,1)]
    for Moment in range(2,K+1):
        A = G@A
        MomentList.append(np.trace(A)/d/choose(n,Moment))
    return MomentList




#########################################################################
#implements the algorithm from Subsection 3.2 of https://doi.org/10.1214/07-AOS581

#ElKaroui method
def H_Estimation(X,sig2=None,positions=None,verbose=False,N=None,options={},eta=0.01):
    d,n = X.shape
    c = d/n
    S = X@X.T/n
    SampEV = np.linalg.eigh(S)[0]
    minSamp = min(SampEV)
    if sig2 is None:
        sig2 = max(SampEV)

    epsilon = (sig2-minSamp)/300
    if positions is None:
        positions = np.arange(minSamp,sig2,epsilon)
    NrPositions = len(positions)
    positions = positions[np.newaxis]

    zVec = np.array([lam+eta*1j for lam in SampEV]+[lam-eta*1j for lam in SampEV])[np.newaxis]
    SampVec = np.array(SampEV)[np.newaxis]
    M = (1-c)*(-1/zVec.T)+c/(SampEV-zVec.T)
    vVec = np.sum(M,axis=1)/d
    vVec = vVec[np.newaxis]
    M = positions/(1+positions*vVec.T)

    V = c*M
    v = zVec+1/vVec
    v = v[0,:]

    Q = np.real(V).T@np.real(V)+np.imag(V).T@np.imag(V)
    p = -np.real(V).T@np.real(v)-np.imag(V).T@np.imag(v)
    b = np.matrix(1)
    A = np.ones((1,NrPositions))
    G = -np.eye(NrPositions)
    h = np.zeros(NrPositions)

    Q = Q.astype('float')
    p = p.astype('float')
    A = A.astype('float')
    b = b.astype('float')
    G = G.astype('float')
    h = h.astype('float')

    Q = cvxopt.matrix(Q)
    p = cvxopt.matrix(p)
    A = cvxopt.matrix(A)
    b = cvxopt.matrix(b)
    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)

    tol = 1e-14
    cvxopt.solvers.options['reltol']=tol
    cvxopt.solvers.options['abstol']=tol
    cvxopt.solvers.options['maxiters']=1000
    cvxopt.solvers.options['feastol']=tol
    cvxopt.solvers.options['show_progress'] = False
    sol=cvxopt.solvers.qp(Q, p, G, h, A, b)

    weights = sol['x']

    return positions[0],weights/np.sum(weights)



#########################################################################
#implements the moment estimation method from Section 3 of https://doi.org/10.1111/j.1467-842X.2010.00590.x


@lru_cache(None)
def _weighted_count_partitions(j):
    counts = [0] * j
    out = []

    def rec(remaining_weight, max_part):
        if remaining_weight == 0:
            out.append(tuple(counts))
            return
        if max_part == 0:
            return

        ell = max_part
        for m in range(remaining_weight // ell + 1):
            counts[ell - 1] = m
            rec(remaining_weight - m * ell, ell - 1)
        counts[ell - 1] = 0

    rec(j, j)
    return out


def _phi(j, counts):
    r = sum(counts)
    denom = factorial(j + 1 - r)
    for cnt in counts:
        denom *= factorial(cnt)
    return factorial(j) / denom


def BaiChenYao_population_moments(X, K):
    """
    Estimate population spectral moments beta_0, beta_1, ..., beta_K
    from a p x n data matrix X.

    Returns
    -------
    beta_hat : ndarray, shape (K + 1,)
        beta_hat[0] = 1.
        beta_hat[j] estimates beta_j = int t^j dH(t), for j = 1, ..., K.
    """
    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array with shape (p, n).")
    if not isinstance(K, (int, np.integer)) or K < 0:
        raise ValueError("K must be a nonnegative integer.")

    beta_hat = np.empty(K + 1, dtype=float)
    beta_hat[0] = 1.0

    if K == 0:
        return beta_hat

    p, n = X.shape
    if n < 2:
        raise ValueError("X must contain at least two observations/columns.")

    # Paper assumes zero-mean observations; center rows in practice.
    Xc = X - X.mean(axis=1, keepdims=True)

    # Nonzero eigenvalues of S_n = Xc Xc^* / n are singular_values(Xc)^2 / n.
    # This avoids constructing a possibly huge p x p covariance matrix.
    svals = np.linalg.svd(Xc, compute_uv=False)
    eigvals = (svals ** 2) / n

    c = p / n

    # alpha_hat[j] estimates alpha_j = (1/p) tr(S_n^j).
    alpha_hat = np.empty(K + 1, dtype=float)
    alpha_hat[0] = 1.0
    for j in range(1, K + 1):
        alpha_hat[j] = np.sum(eigvals ** j) / p

    # Recursively invert alpha_j = beta_j + lower-order terms.
    for j in range(1, K + 1):
        correction = 0.0

        for counts in _weighted_count_partitions(j):
            is_beta_j_term = counts[j - 1] == 1 and sum(counts[:-1]) == 0
            if is_beta_j_term:
                continue

            term = (c ** (sum(counts) - 1)) * _phi(j, counts)

            for ell, cnt in enumerate(counts, start=1):
                if cnt:
                    if ell >= j:
                        raise RuntimeError("Unexpected non-triangular term.")
                    term *= beta_hat[ell] ** cnt

            correction += term

        beta_hat[j] = alpha_hat[j] - correction

    return beta_hat





#########################################################################