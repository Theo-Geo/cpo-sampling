#%%
"""
Sampling with the accept reject method based 
on the ode of each of the axes. 
The use of this is discouraged. 
"""
import numpy as np 
from matplotlib import pyplot as plt 
import sys

#%%
path2helpers = "./cpo-sampling/"
sys.path.append(path2helpers)
from helpers import struc2, EA2vectors, odf, get_lm, struc2coeff
from helpers import plotS2point, cart2sph, get_proj
from helpers import plot_a2_odf

#%% import example data 

cpo_data = np.genfromtxt("CPO-test.dat", skip_header=1)

id_mask = cpo_data[:,0] == 8 # 3 or whatever index or file you find

euler_angles = cpo_data[id_mask, 2:].T
vf = cpo_data[id_mask, 1]

axes_ens = EA2vectors(*euler_angles, deg=True) #EA2rotmat(*euler_angles.T, deg=True).transpose(0,2,1)
a2_obs = np.array([struc2(ax_ens, vf) for ax_ens in axes_ens])
nlm_obs = np.array([struc2coeff(a2_obs[i]) for i in range(3)])


#%% trying to sample  

def sample_dist(nlm_, N:int):
    lm2 = get_lm(2)
    # first pick a batch of random angles 
    phi1_set = 2*np.pi*np.random.uniform(0,1,N)
    theta_set  = np.arccos(1-2*np.random.uniform(0,1,N)) # colatitude
    phi2_set = 2*np.pi*np.random.uniform(0,1,N)
    
    # pick a batch of random numbers one for each axis and angle trio
    p_rand = 1/(4*np.pi)*np.random.uniform(-1.5,6,(3,N))
    
    # convert angles to rotmats 
    axes_ens_r = EA2vectors(phi1_set, theta_set, phi2_set)
    
    lats = np.empty((3, *phi1_set.shape)); lons = np.empty((3, *phi1_set.shape))
    for i in range(3): lats[i], _, lons[i] = cart2sph(axes_ens_r[i])
    
    p_odf = np.array([odf(nlm_[i], lm2, lats[i], lons[i]) for i in range(3)])
    
    mask = np.sum(p_odf >= p_rand, axis=0)==3
    N_acc = np.sum(mask)
    
    phi1_samples = phi1_set[mask]
    theta_samples = theta_set[mask]
    phi2_samples = phi2_set[mask]
    
    print(f"sampling rate: {N_acc/N*100} %")
    print("number of final samples:", N_acc)
    
    return np.array([phi1_samples, theta_samples, phi2_samples])

#%% sampling some of the nlm states 

p = 1; q = 0.5 
a2_ideal = np.array([np.diag(np.roll([p, (1-p)*(1-q), (1-p)*q], i)) for i in range(3)])
nlm_ideal = np.array([struc2coeff(a2_ideal[i]) for i in range(3)])

EA_sampled = sample_dist(nlm_obs, 10000)
axes_sampled = EA2vectors(*EA_sampled)

a2_sampled = np.array([struc2(ax_) for ax_ in axes_sampled])

#%% plotting

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
