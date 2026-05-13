import numpy as np
from math import ceil

from VisualizationTools import ErrorContours, PopEV_Y_maker

######################
c = 1/10
ExampleNumber=1
resolution = 1/100

##Figure 5a
d = 20

##Figure 5b
#d = 200

##Figure 5c
#d = 2000

######################

n = ceil(d/c)
PopEV,Y = PopEV_Y_maker(d,n,ExampleNumber)
S = Y@Y.T/n
SampEV,_ = np.linalg.eigh(S)

ErrorContours(SampEV,c,PopEV,resolution=resolution)
