import numpy as np 
r = np.random
import matplotlib.ticker as mticker
from numba import njit # not necessary, but will make code slower
import cartopy.crs as ccrs # necessary for plotting 
import scipy.special as sp

#%% main functions 

def sample_dist(a2_obs:np.ndarray, N_grains:int, tolerance:float, max_iterations:int=100000):
    """
    sample an orientation distribution in SO(3) 
    represented by the second order structure tensors of 3 axes

    Parameters
    ----------
    a2_obs : np.ndarray of shape (3,3,3)
        array of second order structure tensors that are to be sampled
        first dimension (100, 010, 001)
        second and third dimension structure tensor
    N_grains : int
        number of grains that are supposed to be generated
    tolerance : float
        tolerance for the RMS difference between structure tensors
        recommened is something around 2e-4 
        higher will lead to a very low acceptance rate
    max_iterations : int, optional
        maximum iterations of drawing a random orientation. 
        The default is 100000.

    Returns
    -------
    ensemble : np.ndarray of shape (3,N_grains)
        ensemble of euler angles (phi1, theta, phi2)
    S_arr : np.ndarray of length n_iter
        array of misfits 
    n_iter : int 
        final number of iterations

    """
    ensemble = draw_uniform_EA(N_grains)
    axes_old = EA2vectors(*ensemble)
    axes_new = np.copy(axes_old)
    
    S_old, a2_dif_old = misfit_axes(axes_old, a2_obs, sigma=tolerance)
    print(S_old)

    S_arr = np.array([]) # misfit function
    # dif_arr = np.array([]) # absolute differences

    max_iterations = max_iterations
    n_acc = 0; n_iter = 0
    a2_dif_old = 10
    
    while a2_dif_old > tolerance:
        S_arr = np.append(S_arr, S_old)
        n_iter +=1
        if n_iter>max_iterations: 
            print("Warning: maximum number of iterations reached") 
            break
        #dif_arr = np.append(dif_arr, a2_dif_old)
        random_index = r.randint(0,N_grains-1)
        random_EA = draw_uniform_EA(1)
        # only doing this for 3 vectors at a time instead of 3*N_grains
        axes_new[:,random_index] = EA2vectors(*random_EA) 
         
        # compute misfit and RMS difference
        S_new, a2_dif_new = misfit_axes(axes_new, a2_obs, sigma=tolerance)
        p_acc = np.e**(min(S_old - S_new, 0))
        
        if p_acc > r.random(): 
            axes_old = np.copy(axes_new)
            ensemble[:,random_index] = random_EA[:,0] 
            S_old = np.copy(S_new)
            a2_dif_old = np.copy(a2_dif_new)
            n_acc += 1
        else: axes_new = np.copy(axes_old) # reset to unperturbed state
        
    print("final residual:", a2_dif_old)
    print("number of iterations:", n_iter)
    print("acceptance rate:", n_acc/n_iter)
    
    return ensemble, S_arr, n_iter


@njit # python but c compiled, if removed the code  will run but slower
def misfit_axes(axes:np.ndarray, a2_data:np.ndarray, sigma:float): 
    """
    misfit of an ensemble of axes 
    with respect to some observed second order structure tensors

    Parameters
    ----------
    axes : np.ndarray of shape (3,N_grains,3)
        ensemble of axes 
        first dimension (100, 010, 001)
        second dimension grain i 
        third dimension (x,y,z)
    a2_data : np.ndarray
        observed array of second order structure tensors
        first dimension (100, 010, 001)
        second and third dimension structure tensor
    sigma : float
        standart deviation on one observed value 

    Returns
    -------
    S : float 
        misfit 
    RMS : float
        rootmean squared difference of observed structure tensors 
        and modeled structure tensors
    """
    N_g = len(axes[0])
    sigma_mean = sigma/np.sqrt(N_g) # error on a mean derived from N_g samples
    delta_a2 = 0.0
    #_, evecs_100 = np.linalg.eigh(struc2(axes[0]))
    for i in range(3): 
        a2_model = struc2(axes[i])
        dev_2 = (a2_data[i] - a2_model)**2
        delta_a2 += (dev_2[0,0] + dev_2[1,1] + dev_2[2,2] +
              dev_2[0,1] + dev_2[0,2] + dev_2[1,2])/(3*6)
    S = delta_a2/sigma_mean**2
    return S, np.sqrt(delta_a2)

def draw_uniform_EA(N_samples:int=1):
    """
    the uniform distribution is sin(theta)/8pi^2
    """
    phi1_r = 2*np.pi*np.random.uniform(0,1, N_samples)
    theta_r  = np.arccos(1-2*np.random.uniform(0,1,N_samples)) # "colatitude"
    phi2_r = 2*np.pi*np.random.uniform(0,1, N_samples)
    return np.array([phi1_r, theta_r, phi2_r])


#%% conversion

def EA2vectors(phi1,theta,phi2, deg=False): 
    
    # convert angles to rotmats
    rot_mats = EA2rotmat(phi1, theta, phi2, passive=False, deg=deg)
    # then to vectors | array with shape ((100,010,001), phi1.shape) 
    axes_ens = np.array([rot_mats[...,i] for i in range(3)])
    
    return axes_ens

def EA2rotmat(phi1, theta, phi2, deg:bool=False, passive:bool=False):
    # the zxz-convention R = R^z_a R^x_b R^z_c according to wikipedia is used
    
    rot_matrix = np.empty((3,3) if len(np.atleast_1d(phi2))==1 
                          else (len(phi2),3,3), dtype=float)

    if not passive: phi1, phi2 = np.copy(phi2), np.copy(phi1)
    if deg: phi1 = np.deg2rad(phi1); phi2 = np.deg2rad(phi2); theta = np.deg2rad(theta)

    rot_matrix[..., 0,0] = np.cos(phi2)*np.cos(phi1) -np.cos(theta)*np.sin(phi1)*np.sin(phi2);
    rot_matrix[..., 0,1] = -np.cos(phi2)*np.sin(phi1) -np.cos(theta)*np.cos(phi1)*np.sin(phi2);
    rot_matrix[..., 0,2] = -np.sin(phi2)*np.sin(theta);

    rot_matrix[..., 1,0] = np.sin(phi2)*np.cos(phi1) +np.cos(theta)*np.sin(phi1)*np.cos(phi2);
    rot_matrix[..., 1,1] = -np.sin(phi2)*np.sin(phi1) +np.cos(theta)*np.cos(phi1)*np.cos(phi2);
    rot_matrix[..., 1,2] = np.cos(phi2)*np.sin(theta);

    rot_matrix[..., 2,0] = -np.sin(theta)*np.sin(phi1);
    rot_matrix[..., 2,1] = -np.sin(theta)*np.cos(phi1);
    rot_matrix[..., 2,2] = np.cos(theta);
    
    assert any(np.atleast_1d(rot_matrix[...,2,2]) <= 1.0), "Invalid rotation matrix:np.cos(theta) > 1"
    
    if passive: return rot_matrix
    elif len(np.atleast_1d(phi2))==1: return rot_matrix.T
    else: return rot_matrix.transpose(0, 2, 1)
    
def rotmat2EA(rotmat): 
    # the zxz-convention R = R^z_a R^x_b R^z_c according to wikipedia is used
    
    phi1 = np.arctan2(rotmat[2,0], rotmat[1,0])
    theta = np.arccos(rotmat[0,0])
    phi2 = np.arctan2(rotmat[0,2], -rotmat[0,1])
    
    return phi1, theta, phi2

def orthogonalize(matrix): 
    # decompose matrix into orthogonal matrix and triangular one
    Q, R = np.linalg.qr(matrix)
    
    return Q

@njit
def struc2(n_ens, weights=None):
    a_tens = np.zeros((3,3))    
    if weights is None: weights = np.ones(len(n_ens))
    for n, w in zip(n_ens,weights): a_tens += np.outer(n,n)*w
    return a_tens/np.sum(weights)

#%% functions for spectral expansion coefficients 
# work from Rathmann and Lilien 2021 - 2025 
# for more information visit https://nicholasmr.github.io/specfab/

def odf(nlm_:np.ndarray, lm_:np.ndarray, lat:np.ndarray, lon:np.ndarray):
    """
    orientation distribution function from spectral expansion

    Parameters
    ----------
    nlm_ : np.ndarray of type complex
        spectral expanison coefficients
    lm_ : np.ndarray 
        indices for expansion coefficients 
    lat : np.ndarray or float
        latitude in radians as grid or single value
    lon : np.ndarray or float
        longitude in radians as grid or single value

    Returns
    -------
    F : np.ndarray or float
        probability of latitude and longitude pair(s)
        derived from the odf expressed in terms of spectral expansion coefficients

    """
    colat = np.pi/2-lat
    nlmlen = len(nlm_)
    F = np.real(np.sum([ nlm_[ii]*sp.sph_harm(lm_[1][ii], lm_[0][ii], lon, colat) 
                        for ii in np.arange(nlmlen) ], axis=0))
    return F

def probabilities_of_EA(phi1, theta, phi2, nlm_axes, lm): 
    """
    
    Parameters
    ----------
    phi1 : float or np.ndarray
        first Euler angle in the ZXZ convention in radians from [0,2pi]
    theta : float or np.ndarray
        second Euler angle in the ZXZ convention in radians from [0,pi]
    phi2 : float or np.ndarray
        third Euler angle in the ZXZ convention in radians from [0,2pi]
    nlm_axes : complex np.ndarray of shape (3,nlm_len)
        spherical harmonics expansion coefficients
    lm : TYPE
        DESCRIPTION.

    Returns
    -------
    probabilities : np.ndarray
        normalized probability of for each axis distribution

    """
    axes = EA2vectors(phi1, theta, phi2)
    lat, _, lon = cart2sph(axes)
    probabilities = np.array([odf(nlm_axes[i], lm, lat[i], lon[i])
                              for i in range(3)])
    return probabilities

def get_lm(L):
    # indices of the spectral expansion series 
    # l <= L;       -l <= m <=l
    l_list = [];    m_list = []
    l = 0
    while l <= L:
        m = -l
        while m <= l:
            l_list.append(l)
            m_list.append(m)
            m +=1
        l+=2 # for antipodal ensembles only even model are non zero
    return np.array([l_list, m_list])


def discretize(nlm, lm, latres, lonres):

    """
    Sample distribution on equispaced lat--lon grid
    """

    vcolat = np.linspace(0,   np.pi, latres) # vector
    vlon   = np.linspace(0, 2*np.pi, lonres) # vector
    lon, colat = np.meshgrid(vlon, vcolat) # gridded (matrix)
    lat = np.pi/2-colat

    nlmlen_from_nlm = len(nlm)
    nlmlen_from_lm = lm.shape[1]
    if nlmlen_from_nlm != nlmlen_from_lm: 
        nlmlen = np.amin([nlmlen_from_nlm,nlmlen_from_lm]) # pick smallest common range and continue (this is probably what the user wants)
        #warnings.warn('sfplt.discretize(): dimensions of nlm (%i) and lm (%i) do not match, setting nlm_len=%i'%(nlmlen_from_nlm, nlmlen_from_lm, nlmlen))
    else:
        nlmlen = nlmlen_from_nlm
        
    F = np.real(np.sum([ nlm[ii]*sp.sph_harm(lm[1][ii], lm[0][ii], lon, colat) 
                        for ii in np.arange(nlmlen) ], axis=0))

    return (F, lat, lon)

def struc2coeff(a2:np.ndarray): 
    """
    convert structure tensors to spectral exansion coefficients
    This is a rip off of the work of N.M. Rathmann and D. Lillien 
    for more information see https://nicholasmr.github.io/specfab/
    
    Parameters
    ----------
    a2 : np.ndarray of shape (3,3)
        second order structure tensor

    Returns
    -------
    nlm : np.ndarray of length 6
        spectral expansion coefficients up to lowest order 
        n^0_0, n^{-2}_2, n^{-1}_2, n^0_2, n^1_2, n^2_2

    """
    nlm = np.empty(6, dtype=np.complex256)
    nlm[0] = 1/(2.*np.sqrt(np.pi))
    nlm[3] = -0.25*((-2.0)*np.sqrt((5.0)) + (3.0)*np.sqrt((5.0))*a2[0,0] + 3.0*np.sqrt((5.0))*a2[1,1])/np.sqrt(np.pi)
    nlm[4] = -0.5*(np.sqrt(7.5)*a2[0,2]*np.sqrt(np.pi**(-1.0))) + complex(0, 0.5)*np.sqrt(7.5)*a2[1,2]*np.sqrt(np.pi**(-1.0))
    nlm[5] = -0.125*((-1.0)*np.sqrt((30.0))*a2[0,0] + np.sqrt((30.0))*a2[1,1])/np.sqrt(np.pi) - complex(0,0.5)*np.sqrt(7.5)*a2[0,1]*np.sqrt(np.pi**(-1.0))
    nlm[1] = +np.conj(nlm[5]) # n^{-2}_2 = conjugate(n^{2}_2) 
    nlm[2] = -np.conj(nlm[4]) # n^{-1}_2 = conjugate(n^{1}_2)
    
    return nlm

def plot_a2_odf(ax, a2:np.ndarray, lvl=None, lat_res:int=50, lon_res:int=100): 
    if lvl is None: lvl= np.linspace(0,5,11)/(4*np.pi)
    nlm = struc2coeff(a2); lm2 = get_lm(2)
    odf_ideal, lat_grid, lon_grid = discretize(nlm, lm2, lat_res, lon_res)
    
    surf = plotS2contourf(ax, np.rad2deg(lon_grid), np.rad2deg(lat_grid), 
                      odf_ideal, cmap="Greys", levels=lvl) 
    return surf 


#%% plotting functions

def get_proj(rot=30, incl=60): return ccrs.Orthographic(rot, 90-incl)

def add_grid(ax, geo): 
    grid_kwargs = {'ylocs':np.arange(-90,90+30,30), 
                   'xlocs':np.arange(0,360+45,45), 'linewidth':0.5, 
                   'color':'black', 'alpha':0.25, 'linestyle':'-'} 
    gl = ax.gridlines(crs=geo, **grid_kwargs)
    gl.xlocator = mticker.FixedLocator(np.array([-135, -90, -45, 0, 90, 45, 135, 180])) # grid line longitudes

def add_coord_axes(ax, geo, fontsize): 
    coord_kwargs = dict(ha='center', va='center', transform=geo, 
                        color='tab:red', fontsize=fontsize)
    ax.text(0,  0, '$x$', **coord_kwargs)
    ax.text(90, 0, '$y$', **coord_kwargs)
    ax.text(180,  0, '$-x$', **coord_kwargs)
    ax.text(279, 0, '$-y$', **coord_kwargs)
    ax.text(0, 90, '$z$', **coord_kwargs)
    #ax.text(0, -90, '$z$', **coord_kwargs)

def plotS2contourf(ax, lon, lat, data, coord_ax_fs=15, grid=True, *args, **kwargs):
    geo = ccrs.PlateCarree()
    if grid:                add_grid(ax, geo)
    if coord_ax_fs != 0:    add_coord_axes(ax, geo, coord_ax_fs)
        
    return ax.contourf(lon, lat, data, transform=geo, *args, **kwargs)

def plotS2point(ax, v, coord_ax_fs=0, grid=False, *args, **kwargs):
    geo = ccrs.PlateCarree()
    if grid:                add_grid(ax, geo)
    if coord_ax_fs != 0:    add_coord_axes(ax, geo, coord_ax_fs)
    # Plot point on S2: wraps plt.plot()
    lat, colat, lon = cart2sph(v, deg=True)
    return ax.scatter(lon, lat, transform=geo, *args, **kwargs)

def cart2sph(v, deg=False):
    # Cartesian vector(s) --> spherical coordinates
    x, y, z = v[...,0], v[...,1], v[...,2]
    lon = np.arctan2(y,x)
    colat = np.arctan2(np.sqrt(np.power(x,2) + np.power(y,2)), z)
    lat = np.pi/2 - colat
    if deg: return (np.rad2deg(lat), np.rad2deg(colat), np.rad2deg(lon))  
    else:   return (lat, colat, lon)
        
def sph2cart(colat, lon, deg=False):
    # Spherical coordinates --> Cartesian vector(s)
    if deg: colat, lon = np.deg2rad(colat), np.deg2rad(lon)
    colat, lon = np.atleast_1d(colat), np.atleast_1d(lon)
    r = np.array([ [np.cos(lon[ii])*np.sin(colat[ii]), np.sin(lon[ii])*np.sin(colat[ii]), +np.cos(colat[ii])] for ii in range(len(colat)) ]) # radial unit vector
    return r










