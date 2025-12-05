import numpy as np
from math import ceil

from Visualization import ErrorContours, PopEV_Y_maker

######################
c = 1/20
ExampleNumber=1
resolution = 1/300
tau,kappa = (0.01,10)
    
d = 20
#d = 200
#d = 2000

######################

n = ceil(d/c)
PopEV,Y = PopEV_Y_maker(d,n,ExampleNumber)
S = Y@Y.T/n
SampEV,_ = np.linalg.eigh(S)

ErrorContours(SampEV,c,PopEV,resolution=resolution)
