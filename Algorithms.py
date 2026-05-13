import numpy as np
from math import ceil, sqrt, pi
from scipy.stats import norm

#helper function to check if an object is iterable
def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False

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


# algorithm to find the support of \nu_n
# returns a vector of points that fill out the support
def findSupp_nu(SampEV,PopEV,c):
    d = len(PopEV)

    #Use automatic curve dicovery to find the region of interest
    CurveList = np.array(CurveDiscoveryHSpace(SampEV,c))
    LeftEdges, Heights, RightEdges = CurveList[:,0], CurveList[:,1], CurveList[:,2]
    showLeft, showHeight, showRight = min(LeftEdges), max(Heights), max(RightEdges)
    showLeft, showHeight, showRight = showLeft-0.2*(showRight-showLeft), showHeight*1.3, showRight+0.2*(showRight-showLeft)
    
    #find support of \nu_n
    x,h = DBoundary(0,np.inf,PopEV,c,showLeft,showHeight,showRight)
    z_Boundary = np.array(x)+1j*np.array(h)
    z_Boundary = z_Boundary[np.imag(z_Boundary)>0]
    PopStil = np.sum(1/(PopEV[np.newaxis,:]-z_Boundary[:,np.newaxis]),axis=1)/d
    Phi = (1-c*z_Boundary*PopStil-c)*z_Boundary
    supp_Points = np.real(Phi)

    return supp_Points


# vectorwise algorithm for the population Stieltjes transform estimator as in Definition 1.5
# returns 0, if it could not be found within maxIterations many iterations
# if keyword arguments tau, kappa are given, conidtions (2.11) and (2.12) are also checked
def Vectorwise_StieltjesTransformEstimator(z,SampEV,c,tau=0,kappa=np.inf,maxIterations=100,tol=10**(-12),theta=1,doCondition_2_13=False,supp_nu=None):
    assert (np.imag(z)>0).all()
    d = len(SampEV)
    lastV = np.zeros(len(z))
    zVec = z[np.newaxis]
    SampVec = np.array(SampEV)[np.newaxis]
    V = np.sum(1/(SampVec.T-zVec),axis=0)/d
    Iterations = 0
    while Iterations < maxIterations and np.max(np.abs(V-lastV)) > tol:
        lastV = V
        V = np.sum(SampVec.T/(SampVec.T-(1-c*V)*zVec),axis=0)/d
        Iterations += 1
    s = (V-1)/z
    Phi = (1-c*V)*z
    s[np.abs(V-lastV) > tol]=0
    s[np.abs(c*z*np.imag(V)/np.imag(Phi))>theta]=0
    s[np.imag(Phi)<=0]=0

    # check conditions (2.11) and (2.12), if tau>0 or kappa<inf
    s[np.abs(Phi)>kappa]=0
    s[np.min(np.abs(SampEV[:,np.newaxis]-(Phi)[np.newaxis,:]),axis=0)<=tau]=0

    #apply condition (2.13), if wanted and supp_nu available
    if doCondition_2_13 and not (supp_nu is None):
        #find \hat{Phi}_n and apply condition
        hatPhi = (1-c*z*s-c)*z
        ConditionSet = (np.all(np.abs(hatPhi[..., None] - supp_nu) > tau, axis=-1))
        s[~ConditionSet]=0
    return s


# algorithm for finding admissible curves. Output is a list of curves [(Left,Height,Right),...]
# each curve (Left,Height,Right) will be a tuple of real numbers determining the square curve
# every hole of \hat{D}(tau,kappa,n) will be surrounded by exactly one curve from the output list
def CurveDiscoveryHSpace(SampEV,c,tau=0.01,kappa=np.inf,resolution=10**(-2),extraGrid=False,doCondition_2_13=False,PopEV=None):
    d = len(SampEV)
    sig2 = max(SampEV)

    # define an interval on which holes in \hat{D}(tau,kappa,n) must lie, 
    # sig2 is indeed larger than all population eigenvalues. (this may fail for very small d)
    xMin,xMax = -sig2/2*(2+c+sqrt(c*c+8))-0.1, max(sig2/2*(2+c+sqrt(c*c+8)),sig2+tau)+0.1

    # define grid on which to search for holes in \hat{D}(tau,kappa,n)
    x = np.arange(xMin,xMax,resolution)
    if extraGrid:
        x = np.concatenate([x,(SampEV[1:]+SampEV[:-1])[:-50]/2])
        x = np.sort(x)
    z = x+1j*resolution

    #find the support of nu, if needed for condition (2.13)
    if doCondition_2_13:
        supp_nu = findSupp_nu(SampEV,PopEV,c)
    else:
        supp_nu = None

    # check for holes in \hat{D}(tau,kappa,n)
    s = Vectorwise_StieltjesTransformEstimator(z,SampEV,c,tau=tau,kappa=kappa,doCondition_2_13=doCondition_2_13,supp_nu=supp_nu)
    m = (s == 0)

    # extract left and right edges of curves
    k = 2 + ceil(2*c) # margin which curves keep from holes in \hat{D}(tau,kappa,n) is k*resolution
    LeftEdges = np.real(z[(lambda t,e,pe: (t-1-k)[(t>0)&(e<m.size-1)&(t-pe-1>=2*k+1)])(
        t:=np.flatnonzero((d:=np.diff(np.r_[False, m, False].astype(int)))== 1),
        e:=np.flatnonzero(d==-1)-1,
        np.r_[-1, (np.flatnonzero(d==-1)-1)[:-1]]
    )])
    RightEdges = np.real(z[(lambda t,e,ns: (e+1+k)[(t>0)&(e<m.size-1)&(ns-e-1>=2*k+1)])(
        t:=np.flatnonzero((d:=np.diff(np.r_[False, m, False].astype(int)))== 1),
        e:=np.flatnonzero(d==-1)-1,
        np.r_[t[1:], m.size]
    )])
    assert len(LeftEdges)==len(RightEdges)
    assert len(RightEdges)>0

    #remove artifacts
    LeftEdges=LeftEdges[RightEdges>0]
    RightEdges=RightEdges[RightEdges>0]

    # find heights of curves
    Heights = []
    for L,R in zip(LeftEdges,RightEdges):
        LastNonEx = 0
        LastEx = xMax
        for iteration in range(10):
            y = (LastNonEx+LastEx)/2
            z = np.arange(L,R,resolution)+1j*y
            if (Vectorwise_StieltjesTransformEstimator(z,SampEV,c,tau=tau,kappa=kappa,doCondition_2_13=doCondition_2_13,supp_nu=supp_nu)!=0).all():
                LastEx = y
            else:
                LastNonEx = y
        Heights.append(LastEx+k*resolution)#+np.max([0.05*(R-L),0.1]))

    # left most edge needs extra attention, since holes in \hat{D}(tau,kappa,n) do not behave nicely left of the imaginary axis
    if LeftEdges[0]<=0:
        LastNonEx = LeftEdges[0]
        LastEx = xMin
        for iteration in range(10):
            x = (LastNonEx+LastEx)/2
            z = x+1j*np.arange(resolution,Heights[0],resolution)
            if (Vectorwise_StieltjesTransformEstimator(z,SampEV,c,tau=tau,kappa=np.inf,doCondition_2_13=doCondition_2_13,supp_nu=supp_nu)!=0).all():
                LastEx = x
            else:
                LastNonEx = x
        #print(LastEx)
        LeftEdges[0]=LastEx-2*resolution

    return list(zip(LeftEdges, np.array(Heights), RightEdges))


# algorithm for constructing the nodes and weights for Gauss-Legenre integration of curve integrals
# requires N to be even, so no nodes land on \R. Output is two vectors of 2N values each.
def IntegrationNodes(N,Left,Top,Right):
    assert N%2==0
    x, w = np.polynomial.legendre.leggauss(N)
    x_plus, w_plus = x[N//2:], w[N//2:]

    z_Left = Left+1j*Top*x_plus
    w_Left = -1j*Top*w_plus

    z_Top = Left+(Right-Left)/2*(1+x)+1j*Top
    w_Top = -(Right-Left)/2*w

    z_Right = Right+1j*Top*x_plus
    w_Right = 1j*Top*w_plus

    z_Final = np.hstack([z_Left,z_Top,z_Right])
    w_Final = np.hstack([w_Left,w_Top,w_Right])

    return z_Final, w_Final


# algorithm calculating the sum of PLSS estimators from Definition 2.8 to a given list of curves
# curves from CurveDiscoveryHSpace will usually work. If not, try increasing tau>0
# works vector-wise in g
def PLSS_estimator_IndividualCurves(SampEV,gL,c,CurveList,tau=0.01,kappa=np.inf):
    d = len(SampEV)
    sig2 = max(SampEV)
    N = 2*ceil(sqrt(d))

    if is_iterable(gL):
        g_List = gL
    else:
        g_List = [gL]

    Sum=np.zeros((len(g_List),),dtype=np.complex128)

    for L,H,R in CurveList:
        z, w = IntegrationNodes(N,L,H,R)
        s = Vectorwise_StieltjesTransformEstimator(z,SampEV,c,tau=tau,kappa=kappa)
        if (s==0).any():
            print('A non-admissible curve was passed into PLSS_estimator')
            print('curve in question: (Left, Height, Right) = ',(L,H,R))
            print(r'points not in \hat{D}(tau,kappa,n): ',z[s==0])
            print('If the curve is from CurveDiscoveryHSpace, make sure the parameters (tau,kappa) match')
            print('increasing tau or decreasing the resolution parameter may help with this issue')
        assert (s != 0).all()
        for i in range(len(g_List)):
            g = g_List[i]
            g_Vec = np.array([g(arg) for arg in z])
            g_VecConj = np.array([g(arg) for arg in np.conj(z)])
            Lg = -1/(2*pi*1j)*sum(w*g_Vec*s) + 1/(2*pi*1j)*sum(np.conj(w)*g_VecConj*np.conj(s))
            Sum[i] += Lg
    if is_iterable(gL):
        return Sum
    else:
        return Sum[0]


# algorithm calculating the PLSS estimator over the full population spectrum
# takes the (d, n) data-matrix Y with independent columns
# works vector-wise in g
def PLSS_estimator_FullSpectrum(Y,g,tau=0.01,kappa=np.inf,resolution=10**(-2),SampEV=None):
    d,n = Y.shape
    c = d/n
    if SampEV is None:
        Y = np.matrix(Y)
        S = Y@Y.H/n
        SampEV,_ = np.linalg.eigh(S)
    CurveList = CurveDiscoveryHSpace(SampEV,c,tau=tau,kappa=kappa,resolution=resolution)
    Lg = PLSS_estimator_IndividualCurves(SampEV,g,c,CurveList,tau=tau,kappa=kappa)
    return Lg

    



#algorithm to simultaneously calculate the PLSS estimator and estimate the mean and variance given in Corollary 2.9
def PLSS_InferMean(SampEV,c,g,tau=0.01,kappa=np.inf,giveVariance=False,singualrityAtZero=False,PopEV=None,Curve=None):
    d = len(SampEV)

    if Curve is None:
        Curves = CurveDiscoveryHSpace(SampEV,c,tau=tau,kappa=kappa)
        OuterCurve = (Curves[0][0],max([curve[1] for curve in Curves]),Curves[-1][2])
    else:
        OuterCurve = Curve

    if singualrityAtZero:
        assert OuterCurve[0]>0
    N = 2*2*ceil(np.sqrt(d))
    Lo,Ho,Ro = OuterCurve
    z_Out,w_Out = IntegrationNodes(N,Lo,Ho,Ro)

    # calculate mean
    h = 10**(-5)*1j
    na = np.newaxis

    if PopEV is None:
        PopStil_Est = lambda z: Vectorwise_StieltjesTransformEstimator(z,SampEV,c,tau=tau,kappa=kappa)
    else:
        PopStil_Est = lambda z: np.sum(1/(PopEV[na,:]-z[:,na]),axis=1)/d

    s_Out = PopStil_Est(z_Out)
    Phi_Out = (1-c*z_Out*s_Out-c)*z_Out
    
    z_Out_per = z_Out+h
    s_Out_per = PopStil_Est(z_Out_per)
    Phi_Out_per = (1-c*z_Out_per*s_Out_per-c)*z_Out_per
    PhiDeriv_Out = (Phi_Out_per-Phi_Out)/h
    
    z_Out_minus = z_Out-h
    s_Out_minus = PopStil_Est(z_Out_minus)
    Phi_Out_minus = (1-c*z_Out_minus*s_Out_minus-c)*z_Out_minus
    PhiDerivDeriv_Out = (Phi_Out_per-2*Phi_Out+Phi_Out_minus)/h**2

    meanVec_Out = -PhiDerivDeriv_Out/(2*c*PhiDeriv_Out)
    
    g_Out = np.array([g(arg) for arg in z_Out])
    g_OutConj = np.array([g(arg) for arg in np.conj(z_Out)])
    mean = -1/(2*pi*1j)*np.sum(w_Out*g_Out*meanVec_Out) + 1/(2*pi*1j)*np.sum(np.conj(w_Out)*g_OutConj*np.conj(meanVec_Out))

    # calculate estimator
    estimator = -1/(2*pi*1j)*np.sum(w_Out*g_Out*s_Out) + 1/(2*pi*1j)*np.sum(np.conj(w_Out)*g_OutConj*np.conj(s_Out))
    

    if giveVariance:
        if singualrityAtZero:
            z_Out2,w_Out2 = IntegrationNodes(N-2,Lo,Ho,Ro)
        else:
            delta = 0.1
            Lo2,Ho2,Ro2 = Lo-delta,Ho+delta,Ro+delta
            z_Out2,w_Out2 = IntegrationNodes(N,Lo2,Ho2,Ro2)

        # calculate variance
        s_Out2 = PopStil_Est(z_Out2)
        Phi_Out2 = (1-c*z_Out2*s_Out2-c)*z_Out2
        z_Out2_per = z_Out2+h
        s_Out2_per = PopStil_Est(z_Out2_per)
        Phi_Out2_per = (1-c*z_Out2_per*s_Out2_per-c)*z_Out2_per
        PhiDeriv_Out2 = (Phi_Out2_per-Phi_Out2)/h

        covArray_Out00 = (1/(z_Out[:,na]-z_Out2[na,:])**2 - PhiDeriv_Out[:,na]*PhiDeriv_Out2[na,:]/(Phi_Out[:,na]-Phi_Out2[na,:])**2)/c**2
        covArray_Out01 = (1/(z_Out[:,na]-np.conj(z_Out2)[na,:])**2 - PhiDeriv_Out[:,na]*np.conj(PhiDeriv_Out2)[na,:]/(Phi_Out[:,na]-np.conj(Phi_Out2)[na,:])**2)/c**2
        g_Out2 = np.array([g(arg) for arg in z_Out2])
        g_Out2Conj = np.array([g(arg) for arg in np.conj(z_Out2)])
        M = (w_Out*g_Out)[:,na]*(np.conj(w_Out2)*g_Out2Conj)[na,:]*covArray_Out01 - (w_Out*g_Out)[:,na]*(w_Out2*g_Out2)[na,:]*covArray_Out00
        variance = 1/(pi**2)*np.real(np.sum(M))

        return estimator, mean, variance

    return estimator, mean



# algorithm for constructing asymptotic confidence intervals to PLSS (see Subsection 5.3)
def PLSS_ConfidenceInterval(SampEV,d,n,g,tau=0.01,kappa=np.inf,ConfidenceLevel=0.95,singualrityAtZero=False):
    c = d/n
    estimator, mean, variance = PLSS_InferMean(SampEV,c,g,tau=tau,kappa=kappa,giveVariance=True,singualrityAtZero=singualrityAtZero)
    lower = norm.ppf((1-ConfidenceLevel)/2, loc=np.real(estimator-mean/n), scale=np.sqrt(np.real(variance/n**2)))
    upper = norm.ppf(1-(1-ConfidenceLevel)/2, loc=np.real(estimator-mean/n), scale=np.sqrt(np.real(variance/n**2)))
    return (lower,upper)



# algorithm calculating a log-determinant estimator by the methods introduced here
# when applicable, calculates an estimator for the log-determinant, 
#      that is close to the estimator from https://doi.org/10.1016/j.jmva.2015.02.003 up to an O(1/n^2)-difference
def LogDet_estimator(Y,tau=0.01,kappa=np.inf,resolution=10**(-2),tz=-100,SampEV=None,Sigma=None,PopEV=None):
    d,n = Y.shape
    estimator, mean = PLSS_InferMean(Y,lambda z : np.log(z),tau=tau,kappa=kappa,giveVariance=False,singualrityAtZero=True)
    return estimator-mean/n







############################################################################
## Algorithms realizing the GLSS estimation from Section D in the supplement

# algorithm for infering valid f-curves from the already discovered admissible g-curves
# condition (c) of Definition D.1 may be ignored in practice, since the functions are
#     only evaluated at discrete nodes on the curve images, which practically never overlap
def CurveDiscoveryNuSpace(SampEV,c,tau=0.01,kappa=np.inf,g_curves=None):
    d = len(SampEV)
    if g_curves is None:
        g_curves = CurveDiscoveryHSpace(SampEV,c,tau=tau,kappa=kappa)
    N = 2*ceil(sqrt(d))
    NuBoxes = []
    for L,H,R in g_curves:
        nodes,_ = IntegrationNodes(N,L,H,R)
        s_nodes = Vectorwise_StieltjesTransformEstimator(nodes,SampEV,c)
        Phi_nodes = (1-c*nodes*s_nodes-c)*nodes
        NuBoxes.append((min(np.real(Phi_nodes)),max(np.imag(Phi_nodes)),max(np.real(Phi_nodes))))
    tau = 0.1
    f_curves = []
    for i in range(len(NuBoxes)):
        Lb,Hb,Rb = NuBoxes[i]
        if i==0:
            L = Lb-tau
        else:
            L = (Lb+NuBoxes[i-1][2])/2
        if i==len(NuBoxes)-1:
            R = Rb+tau
        else:
            R = (Rb+NuBoxes[i+1][0])/2
        H = Hb+tau
        f_curves.append((L,H,R))
    return f_curves


# algorithm calculating the GLSS estimator from Definition D.2
# takes functions f,g and lists g_curves,f_curves each of the form [(L,H,R),...]
def GLSS_estimator(SampEV,c,g,f,g_curves,f_curves,tau=0.01,kappa=np.inf):
    d = len(SampEV)
    N = 2*ceil(sqrt(d))
    Sum = 0
    for g_curve in g_curves:
        for f_curve in f_curves:
            L_f,H_f,R_f = f_curve
            L_g,H_g,R_g = g_curve
            zf,wf = IntegrationNodes(N,L_f,H_f,R_f)
            zg,wg = IntegrationNodes(N,L_g,H_g,R_g)
            na,co = np.newaxis, np.conj
            sf = c*np.sum(1/(SampEV[:,na]-zf[na,:]),axis=0)/d-(1-c)/zf
            sg = Vectorwise_StieltjesTransformEstimator(zg,SampEV,c,tau=tau,kappa=kappa)
            fz = np.array([f(z) for z in zf])
            fCoz = np.array([f(np.conj(z)) for z in zf])
            gz = np.array([g(z) for z in zg])
            gCoz = np.array([g(np.conj(z)) for z in zg])
            kz1z2 = (zf[na,:]*sf[na,:]**2+(1-c)*sf[na,:]+c*sg[:,na])/c/(zg[:,na]*sf[na,:]+1)/zf[na,:]
            kz1Coz2 = (co(zf[na,:]*sf[na,:]**2)+(1-c)*co(sf)[na,:]+c*sg[:,na])/c/(zg[:,na]*co(sf)[na,:]+1)/co(zf)[na,:]
            kCoz1z2 = (zf[na,:]*sf[na,:]**2+(1-c)*sf[na,:]+c*co(sg)[:,na])/c/(co(zg)[:,na]*sf[na,:]+1)/zf[na,:]
            kCoz1Coz2 = co(kz1z2)
            Sum += np.sum((fz*wf)[na,:]*(gz*wg)[:,na]*kz1z2)/(4*pi**2)
            Sum -= np.sum((fCoz*co(wf))[na,:]*(gz*wg)[:,na]*kz1Coz2)/(4*pi**2)
            Sum -= np.sum((fz*wf)[na,:]*(gCoz*co(wg))[:,na]*kCoz1z2)/(4*pi**2)
            Sum += np.sum((fCoz*co(wf))[na,:]*(gCoz*co(wg))[:,na]*kCoz1Coz2)/(4*pi**2)
    return Sum




