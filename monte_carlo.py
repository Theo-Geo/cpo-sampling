#%%
"""
minimum working example for sampling a set of euler angles that represent an ODF on SO(3)

@author: Theo Häußler
"""

import numpy as np 
rng = np.random.default_rng()
r = np.random
from matplotlib import pyplot as plt 
import sys
#%%
path2helpers = "./ev-sampling/"
sys.path.append(path2helpers)
from helpers import struc2, EA2vectors
from helpers import get_proj, plot_a2_odf, plotS2point
from helpers import sample_dist


#%% import example data 

cpo_data = np.genfromtxt("CPO-test.dat", skip_header=1)

id_mask = cpo_data[:,0] == 8 # 3 or whatever index or file you find

euler_angles = cpo_data[id_mask, 2:].T
vf = cpo_data[id_mask, 1]

axes_ens = EA2vectors(*euler_angles, deg=True) 
a2_obs = np.array([struc2(ax_ens, vf) for ax_ens in axes_ens])

#%% an unused idealized fabric state aligned with the coordinate frame 
# I used this one to play around with the inversion
# states that approach p=1 or a perfect single maximum harder to sample

p = 0.6 
q = 0.5 
a2_ideal = np.array([np.diag(np.roll([p, (1-p)*(1-q), (1-p)*q], i)) 
                     for i in range(3)])


#%% sampling

ensemble, S, n_iter = sample_dist(a2_obs, 500, 2e-4)

#%% misfit evolution 

figS, axS = plt.subplots(1,1, figsize=(4,3), dpi=400) 
axS.plot(np.arange(n_iter), S)
axS.set(xlabel="Number of iterations", ylabel=r"misfit $S$", yscale="log", xlim=(0,n_iter))

#%% plotting 
# this in not a kde. The odf thats plotted here 
# only has the information of the second order 
# structure tensor 

axes_sampled = EA2vectors(*ensemble)
a2_sampled = np.array([struc2(ax_ens) for ax_ens in axes_sampled])

fig = plt.figure(dpi=200)
lvl= np.linspace(0,5,11)/(4*np.pi)

for i in range(3):
    ax1 = plt.subplot(2,3,i+1, projection=get_proj()); ax1.set_global()
    plot_a2_odf(ax1, a2_obs[i], lvl=lvl)
    ax1.annotate(f"{np.int64(np.eye(3)[i])}", xy=(0.33, 1.05), xycoords="axes fraction")
    plotS2point(ax1, axes_ens[i], s=vf*1000, c="b", alpha = 0.2)
    plotS2point(ax1, -axes_ens[i], s=vf*1000, c="b",alpha = 0.2)
    
    ax2 = plt.subplot(2,3,i+4, projection=get_proj()); ax2.set_global()
    surf = plot_a2_odf(ax2, a2_sampled[i], lvl=lvl)
    plotS2point(ax2, axes_sampled[i], s=1, c="b", alpha=0.2)
    plotS2point(ax2, -axes_sampled[i], s=1, c="b", alpha=0.2)
    
    if i == 0: 
        ax1.annotate("observed", xy=(-0.15,0.3), xycoords="axes fraction", rotation=90) 
        ax2.annotate("sampled", xy=(-0.15,0.3), xycoords="axes fraction", rotation=90) 
    
    
cb = ax2.inset_axes([1.2,0,0.15, 2.2])
fig.colorbar(surf, cax=cb, extend="max", label=r"multiples of uniform")
cb.set_yticks(lvl, np.round(lvl*4*np.pi,1))












