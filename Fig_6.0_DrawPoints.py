import numpy as np
from math import ceil
import matplotlib.pyplot as plt

from Visualization import VisualizeHSpace, PopEV_Y_maker

######################
if 1: #Figure 6 points
    d = 200
    c = 2
    ExampleNumber=2
    resolution = 1/40
    tau,kappa = (0.1,10)
######################

n = ceil(d/c)
PopEV,Y = PopEV_Y_maker(d,n,ExampleNumber)

fig, ax = plt.subplots(1, 1, figsize=(8, 5.6))

VisualizeHSpace(Y,tau=tau,kappa=kappa,resolution=resolution,PopEV=PopEV,CurveMode=0,ax=ax,show=False,theta=1)

#ax.scatter([1.3,1.7,2.9],[0.1,0.1,0.1],color='red')
#ax.annotate(r'$z_1$',(1.33,0.12), fontsize=35, textcoords='data')
#ax.annotate(r'$z_2$',(1.73,0.12), fontsize=35, textcoords='data')
#ax.annotate(r'$z_3$',(2.93,0.12), fontsize=35, textcoords='data')

ax.scatter([0,-0.5,-1],[0.1,0.1,0.1],color='red')
ax.annotate(r'$z_3$',(0.03,0.12), fontsize=35, textcoords='data')
ax.annotate(r'$z_2$',(-0.5+0.03,0.12), fontsize=35, textcoords='data')
ax.annotate(r'$z_1$',(-1+0.03,0.12), fontsize=35, textcoords='data')

plt.tight_layout()
plt.show()
