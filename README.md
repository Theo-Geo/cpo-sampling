# sampling for worldbuilder demo 

In this demo I attempt to sample an ensemble of euler angles from a set of second order structure tensors. Work related to the spherical harmonics expansion is based on the work by [Rathmann and Lilien 2021-2026](https://nicholasmr.github.io/specfab/).

## requirements 
the following python packages are required:    
numpy, scipy, matplotlib, cartopy 

## statistics 

The second order structure tensor with respect to any orientation distribution function (ODF) $n(\theta,\phi)$ of normal axes $\bm{n}$ is defined as
```math
\bm{a}^{(2)}_n = \int_{S^2}\bm{n}\otimes\bm{n}\ n(\theta, \phi) \ \mathrm{d}\Omega
```
and can be approximated by a discrete ensemble of axes $\{\bm{n}_i\}_{i=1}^{N_g}$ with volume fractions $f_i$ that sum up to one,
```math
\bm{a}^{(2)}_n = \sum_i^{N_g}\bm{n}_i\otimes\bm{n}_i f_i
```
To obtain the second order structure tensors of the set of axes (100, 010, 001) from Euler angles, one can convert the euler angles to an **active** rotation matrix and multiply the axes by the rotation matrix. This gives 3 sets of axes each the length of the number of grains. 

### spectral expansion of an ODF 

The odf on $S^2$ can be expanded in terms of spherical harmonics with the expansion coefficients $n^m_l$,
```math
n(\theta, \phi) = \sum_{l=0}^{L}\sum_{m=-l}^{l} n_l^m Y_l^m(\theta,\phi) \qquad \text{with} \qquad Y_l^m \propto P_l^m(\cos{(\theta)}) \mathrm{e}^{i m \phi} \text{ ,}
```
here $L$ is a maximum truncation. For infinite L one approcahes an exact description, but also has to consider more and more expansion coefficients. The least information is given by the expansion coefficients up to order $L=2$.

One can also derive a relation between the entries of the structure tensor of order $L$ and the expansion coefficients of that order or lower. In this example I will use the odf one can derive from the second order structure tensor such that we will need the expansion coefficients $\{n^0_0, n^{-2}_2, n^{-1}_2, n^0_2, n^1_2, n^2_2\}$ for each axis to evaluate the odf at any given point. To derive $n^m_l(\bm{a}^{(2)})$ one can first find $\bm{a}^{(2)}(n^m_l)$ (see [Rathmann 2024](https://doi.org/10.1029%2F2024GC011831) equ. 8) and invert it analyticall.
    
## sampling a uniform distribution 

The ZXZ convention for euler angles is used. The uniform distribution of Euler angles representing SO(3) is given by, [(e.g. Bunge 1982, p.32)](http://www.ebsd.info/pdf/Bunge_TextureAnalysis.pdf)
```math
g(\phi_1, \theta, \phi_2) = \frac{\sin(\theta)}{8\pi^2}
```
Splitting it up into 3 separate distributions for each angle and using [inverse transform sampling](https://en.wikipedia.org/wiki/Inverse_transform_sampling) one can sample a homogeneusly distributed set of Euler angles from uniformly distributed random numbers $(u_1,u_2,u_3)$ over $[0,1)$ with, 
```math
\begin{align}
    \phi_1 &= 2 \pi u_1 \\
    \theta &= \arccos(1-2u_2) \\
    \phi_2 &= 2 \pi u_3 \\
\end{align}
```
    
## sampling methods 

### maximum likelihood monte carlo sampling

I propose to construct a misfit $S$ between the modeled structure tensors and the observed structure tensors as follows
```math
\begin{align}
    \Delta_i(\phi_1, \theta, \phi_2) &:= \bm{a}^{(2)}_{\rm obs, i} - \bm{a}^{(2)}_{\rm modeled, i}(\phi_1, \theta, \phi_2)\qquad \qquad i = (100,\ 010,\ 001) \\
    RMS &= \sqrt{\frac{1}{3\cdot 6}\sum_{i=1}^3 \Delta^2_{i,1,1} + \Delta^2_{i,2,2} + \Delta^2_{i,3,3} +\Delta^2_{i,2,3} + \Delta^2_{i,1,3} + \Delta^2_{i,1,2}}\\
    S(\phi_1, \theta, \phi_2) &= \frac{RMS^2}{\sigma_\Delta^2} \text{ ,}
\end{align}
```
with $\sigma_\Delta$ being the allowed standart deviation of $\Delta$, which in turn means the allowed standart error of $\bm{a}^{(2)}_{\rm modeled, i}$ with respect to $\bm{a}^{(2)}_{\rm obs, i}$. The standart error on $\bm{a}^{(2)}_{\rm obs}$ is an error on the mean and scales with the number of samples (or in this case grains) as $\sigma_{a^{(2)}} = \sigma_{\bm{n}_i\otimes \bm{n}_i}/\sqrt{N_g}$. The error on each component of the structure tensor is assumed to be the same and can be prescribed with a tolerance value. If the fluctuations in each component of $\Delta$ are on the order of $\Delta \sim \sigma_{\Delta}$ the misfit is one the order of one.

All of this is a little handwavy the most important part is that $\sigma_\Delta \propto 1/\sqrt{N_g}$ to get a consistent result from the sampling irrespective of the required size of samples. 

The algorith to sample is based on the maximum likelyhood principe such that the likelyhood of an ensemble of euler angles is given by, 
$$
L = \exp(-S(\phi_1, \theta,\phi_2)) \text{ .}
$$
The likelyhood is maximal at one when the misfit is zero. In markov chain monte carlo simulations one usually perturbs the ensemble and then evaluates if the perturbed ditribution is accepted based on the likelihood ratio. The acceptance probability in this case is given by, (This is from a lecture on inverse problems, but I need some sources for this.)
```math 
p_{acc} = L_{new}/L_{old} = \exp(\min(S_{old}-S_{new}, 0)
```
The min is just teher to avoid over flow problems. 
When the misfit of the new distribution is smaller than the old one, $p_{acc}=1$ and the change is always accepted, whereas when $S_{old} < S_{new}$ the $p_{acc}$ is smaller than one. A monte carlo algorithm to sample a set of euler angles (EA) then looks as follows: 

0. draw $N_g$ number of EA $(\phi_1, \theta, \phi_2)$ from a homogeneous distribution and assign a misfit $S_{old}$
1. draw one new trio of EA from a homogeneous distribution on SO(3)
2. draw a random grain index  
3. replace the EA at that index with the new trio 
4. compute a new misfit $S_{new}$(EA')
5. compute the acceptance probability from $S_{new}$(EA') and $S_{old}$(EA)
6. draw a random number p in [0,1) 
7. accept the change if $p<p_{acc}$ else **undo** changes
8. repeat from 1. until tolerance reached or $S\sim \mathcal{O}(1)$

This method works quite well, but has problems with strongly peaked fabrics. 

### accept reject method 

Every set of angles has a probability assigned to them according to the 3 ODFs one can reconstruct from the structure tensor of each axis.

algorithm: 
1. pick random set of 3 axes (or one trio of EA) 
2. calculate probabilites for each axis 
3. draw 3 random numbers $p_1, p_2, p_3$
4. accept the ones where all 3 probabilities are larger than the random number 
5. repeat until number of sampled reached

The tricky part here is that the probability density one gets from the spectral expansion integrates to one, but also yields negative values. I am not yet sure how to deal with that. There seems to be limits to $n(\theta,\phi)\in\frac{1}{4\pi}[-1.5,6]$ such that the random numbers $p_1, p_2, p_3$ were all drawn from the interval $\frac{1}{4\pi}[-1.5,6]$. This still leads to okay results but does not reflect the actual fabric strength. 

# benchmark 

This is still a work in progress but the idea i had by now would be: 
1. evolve the fabric for some time
2. extract second order structure tensors for each axis
3. resample from structure tensors 
4. continuiue fabric development 
5. compare "interrupted" cpo evolution vs. not interrupted 

