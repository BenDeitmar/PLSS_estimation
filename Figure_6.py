import numpy as np
from math import ceil
import matplotlib.pyplot as plt

from VisualizationTools import VisualizeHSpace, PopEV_Y_maker


######################
if 1: #Figure 6a
    d = 200
    c = 1/10
    ExampleNumber=1
    resolution = 1/100
    tau,kappa = (0.05,10)
    ShowAdmissibleCurves=False

if 0: #Figure 6b
    d = 100
    c = 2
    ExampleNumber=2
    resolution = 1/40
    tau,kappa = (0.1,10)
    ShowAdmissibleCurves=False

if 0: #Figure 6c
    d = 784
    c = 1/4
    ExampleNumber=3
    resolution = 1/200
    tau,kappa = (0.025,25)
    ShowAdmissibleCurves=False
######################

n = ceil(d/c)
PopEV,Y = PopEV_Y_maker(d,n,ExampleNumber)

fig, ax = plt.subplots(2,1,layout='constrained', figsize=(8, 2*5.6))

if ShowAdmissibleCurves:
    CurveMode = 1
else:
    CurveMode = 0

VisualizeHSpace(Y,tau=tau,kappa=kappa,resolution=resolution,PopEV=PopEV,CurveMode=CurveMode,doCondition_2_13=False,ax=ax[0],show=False)
VisualizeHSpace(Y,tau=tau,kappa=kappa,resolution=resolution,PopEV=PopEV,CurveMode=CurveMode,doCondition_2_13=True,ax=ax[1],show=False)

ax[0].set_title(rf"$D$(τ,κ,n) in $H$-space for τ={tau}, κ={kappa} without condition (2.13)")
ax[1].set_title(rf"$D$(τ,κ,n) in $H$-space for τ={tau}, κ={kappa} with condition (2.13)")

plt.show()

