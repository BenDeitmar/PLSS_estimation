import numpy as np
from math import ceil

from Visualization import VisualizeHSpace, PopEV_Y_maker


######################
if 0: #Figure 4a
    d = 200
    c = 1/20
    ExampleNumber=1
    resolution = 1/300
    tau,kappa = (0.05,10)

if 1: #Figure 4b
    d = 100
    c = 2
    ExampleNumber=2
    resolution = 1/40
    tau,kappa = (0.1,10)

if 0: #Figure 4c
    d = 784
    c = 1/4
    ExampleNumber=3
    resolution = 1/100
    tau,kappa = (0.05,25)
######################

n = ceil(d/c)
PopEV,Y = PopEV_Y_maker(d,n,ExampleNumber)


VisualizeHSpace(Y,tau=tau,kappa=kappa,resolution=resolution,PopEV=PopEV)

