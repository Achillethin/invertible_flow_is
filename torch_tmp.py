#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 15:18:46 2020

@author: achillethin
"""

#!/usr/bin/env python
# coding: utf-8

# In[32]:


from torch_target import Gaussian_target, Gaussian_mixture, Uniform, BNAF_examples
from torch_trajectory import trajectories
from tqdm import tqdm
import torch
import torch.nn as nn

#import particles
from scipy.stats import multivariate_normal, multinomial
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20, 10)
# import autograd.numpy as np  # Thinly-wrapped numpy
#from autograd import grad
import pdb
import numpy as np

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import MCMC, NUTS, HMC
# ### How to use gaussian mixture target

# In[33]:

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torchType = torch.float32

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


args = dotdict({})


# In[3]:
args.dim = 2
args.mu = torch.zeros(args.dim, dtype = torchType, device = device) +1
args.cov = 1*torch.eye(args.dim, dtype = torchType, device = device) ###Covariance matrix
#args.num_gauss = 2
#args.locs = torch.tensor([[0.,1.], [1., 0]], dtype = torchType, device = device) ###list of tensors for different mu
#args.covs =[0.01*torch.eye(2,  dtype = torchType, device = device)] *args.num_gauss ##list of tensors of different cov matrix


#gauss = Gaussian_mixture(args)
gauss = Gaussian_target(args)
args.qmin = -15
args.qmax = 15
args.pmin = -15
args.pmax = 15

###prior for p
argsp = dotdict({})
argsp.mu = torch.zeros(args.dim, dtype = torchType, device = device)
argsp.cov = torch.eye(args.dim, dtype = torchType, device = device)
#args.prior_p = Gaussian_target(argsp)

args.elbo = False

args.hmax = 400
args.dt = .1
args.n_particles = 100
args.alpha = np.exp(-.1)

args.T = 100



def grad_gauss(q):
    f = (q - args.mu) @ args.cov.inverse()
    return f



def A(k):
    cond = (k < args.T) & (k > -args.T)
    return (cond)*1


a = A(torch.arange(-2*args.T+1, 2*args.T))

# In[35]:


trajs = trajectories(args, a)


#print(L(t['position'][:,args.T:(3*args.T-1)],t['momentum'][:,args.T:(3*args.T-1)] ))
# print(t['weight'])
# print(np.sum(L(t['position'][:, args.T:(3*args.T-1)],
#               t['momentum'][:, args.T:(3*args.T-1)])*t['weight'], axis=1))
# print('Naive Uniform Monte Carlo estimate:', ((args.qmax-args.qmin)**args.dim) *
#      np.mean(gm.get_density(args.qmin + np.random.rand(1000000, 2) * (args.qmax-args.qmin))))


# In[ ]:

###Parameters of the problems (fixed) : dimension, sigma_target (lipschitz constant of grad_potential)
##
###Parameters to tune through ELBO
args.elbo = False

n_attempts = 1
naive = []
our = []
args.dt = 0.1
args.n_particles = 100
args.alpha = np.exp(-0.1)

#a = A(torch.arange(-1*args.T+1, 1*args.T, dtype = torchType, device = device ))
args.train_a = False
a =torch.tensor(np.ones(2* args.T-1), 
                 dtype = torchType, device = device, 
                 requires_grad = (args.elbo&args.train_a))
trajs = trajectories(args, a)


def L(q, p):
    
    return gauss.get_density(q) *(args.qmax-args.qmin)**(args.dim)
    # return 1

argsq = dotdict({})
argsq.x_min = args.qmin
argsq.x_max = args.qmax
argsq.dim = args.dim
argsq.device = args.device
              
prior_q = Uniform(argsq)
# In[14]:
for i in tqdm(range(n_attempts)):
    naive.append(torch.mean((args.qmax-args.qmin)**(args.dim)*gauss.get_density(prior_q.get_samples(args.n_particles))))
    #pdb.set_trace()   
    trajs.prop_trajectories(gauss, grad_gauss)
    our.append(trajs.Z_f(L))

our_arr = np.array(our)  # *(args.qmax-args.qmin)**args.dim
naive_arr = np.array(naive)  # *(args.qmax-args.qmin)**args.dim
print('Our Estimate: ', np.mean(our_arr), 'std', np.std(our_arr))
print('Naive Estimate: ', np.mean(naive_arr), 'std', np.std(naive_arr))



# our_arr
# naive


#%%

# t = trajs.trajs
# i=int(np.random.rand()*args.n_particles)
# #i=140
# f, axes = plt.subplots(nrows=2,ncols=2)
# ax1, ax2 = axes[0]
# ax3,ax4 = axes[1]
# tau_m = int(t['tau_m'][i])
# ax1.scatter(t['position'][i][:,0].numpy(), t['position'][i][:,1].numpy())
# ax1.set_title('Trajectory in 2D')
# L_k_i = L(t['position'][i].numpy(), t['momentum'][i].numpy()) 
# ax2.plot( L_k_i)
# ax2.set_title('Likelihood')
# w_i = t['weight'][i].numpy()
# ax3.plot( w_i)
# ax3.set_title('Weights')
# ax4.plot((L_k_i*w_i).numpy())
# ax4.set_title('Weighted Likelihood')
# print(np.sum(w_i))
# print(np.sum((L_k_i*w_i).numpy()))
# #print(L_k)




#%%

### ELBO: Logarithm of the normalizing constant is computed efficiently in Z_f_log with logsumexp
args.train_a = False
args.elbo = True

args.T = 10

args.dt = 0.1#torch.tensor(0.1, dtype = torchType, device = device)#, requires_grad= args.elbo) ###dt must be strictly positive
args.alpha = np.exp(-0.1)#torch.tensor(1, dtype = torchType, device = device)#, requires_grad= args.elbo)
a =torch.tensor(np.ones(2* args.T-1), 
                 dtype = torchType, device = device, 
                 requires_grad = (args.elbo&args.train_a))
args.n_optim = 100
args.learning_rate = 1e-4
args.n_particles=1


def log_L(q):
    #K = (args.qmax-args.qmin)**(args.dim) * 1/np.sqrt((2*np.pi)**(args.dim)*np.product(args.sigma))
    
    #return K * np.exp(-0.5*np.sum((q-args.mu)**2/args.sigma,axis=2))
    return gauss.get_logdensity(q) #+(args.dim)*torch.log(torch.tensor(args.qmax-args.qmin, dtype = torchType, device = device)) 
   

#if args.train_a:
#    parameters = list([args.a, args.dt, args.gamma])
#else:
#    parameters = list([args.dt, args.gamma])
trajs = trajectories(args, a)

parameters = trajs.parameters()

optimizer = torch.optim.Adam(params=parameters, lr=args.learning_rate)

def grad_gauss(z, target = gauss):
    z = z.detach().requires_grad_(True)
    gradu = torch.autograd.grad(gauss.get_logdensity(z), z)[0]
    return gradu

#%%
with torch.autograd.detect_anomaly():
    #args.dt = args.dt_log.exp()
    #pdb.set_trace()
    trajs.prop_trajectories(gauss, grad_gauss)
    elbo = trajs.Z_f_log(log_L)
    print(elbo)
    print(trajs.dt)
    print(trajs.alpha)
    (-elbo).backward()
    optimizer.step()
    optimizer.zero_grad()




#%%
  
std_normal = torch.distributions.Normal(loc=torch.tensor(0., device=device),
                                                scale=torch.tensor(1., device=device))
std_normal_pyro = torch.distributions.Normal(loc=torch.tensor(0., device=device),
                                                scale=torch.tensor(1., device=device))

n_samples = 39900
n_warmup = 100
args.dim = 2
points_nuts =np.zeros((4,n_samples , args.dim))
points_hmc =np.zeros((4,n_samples , args.dim))


distrib = ['t1']
for idx, name in enumerate(distrib):
    args.bnaf_data = name  
    target = BNAF_examples(args)

    def potential_fn(z):
        z = z['points']
        return -target.get_logdensity(z)
    
    init_samples = std_normal_pyro.sample((1, 2))
    init_params = {'points': init_samples}
    
    hmc_kernel = HMC(potential_fn = potential_fn, step_size=0.3, num_steps= 20, adapt_step_size= False, adapt_mass_matrix= False)
    
    mcmc_hmc = MCMC(hmc_kernel,
                initial_params=init_params,
                num_samples=n_samples,
                warmup_steps=n_warmup)
    
    mcmc_hmc.run()
    z_hmc = mcmc_hmc.get_samples()['points'].squeeze().cpu().numpy()
    points_hmc[idx] = z_hmc

#%%
np.save('../samples/hmc_rezende_03', points_hmc)


#%%
args.dim =4
args.num_gauss = 8
args.locs = 2*torch.tensor([[2.,1.,0,0],[-2.,-1.,0,0],
                      [1.,0.,0,0],[-1.,0.,0,0],
                      [3.,0.,1,0],[-3.,0.,-1,0],
                      [0.,0.,0,1],[0.,0.,0,-1]], dtype = torchType, device = device) ###list of tensors for different mu
args.covs =[0.05*torch.eye(4,  dtype = torchType, device = device)] *args.num_gauss ##list of tensors of different cov matrix
args.p_gaussians = torch.tensor([.5,.5,.5,.5,.5,.5,.5,.5], dtype = torchType, device = device)

gauss = Gaussian_mixture(args)
points_hmc =np.zeros((4,n_samples , args.dim))

#%%

def potential_fn(z):
    z = z['points']
    return -gauss.get_logdensity(z)

init_samples = std_normal_pyro.sample((1, args.dim))
init_params = {'points': init_samples}

hmc_kernel = HMC(potential_fn = potential_fn, step_size=0.01, num_steps= 25, adapt_step_size= False, adapt_mass_matrix= False)

mcmc_hmc = MCMC(hmc_kernel,
            initial_params=init_params,
            num_samples=n_samples,
            warmup_steps=n_warmup)

mcmc_hmc.run()
z_hmc = mcmc_hmc.get_samples()['points'].squeeze().cpu().numpy()
points_hmc[idx] = z_hmc


#%%
np.save('../samples/hmc_corr_gauss_05', points_hmc)


#%%
args.mu = torch.tensor(np.array([0,0]), dtype = torchType, device = device)
s = np.array([[10,0], [0, 0.001]])
args.cov =torch.tensor( s @ np.array([[2**(-1/2),  2**(-1/2)],[-2**(-1/2), 2**(-1/2)]]), dtype = torchType, device = device)
args.dim = 2

gauss = Gaussian_target(args)
points_hmc =np.zeros((4,n_samples , args.dim))

#%%

def potential_fn(z):
    z = z['points']
    return -gauss.get_logdensity(z)

init_samples = std_normal_pyro.sample((1, args.dim))
init_params = {'points': init_samples}

hmc_kernel = HMC(potential_fn = potential_fn, step_size=0.03, num_steps= 25, adapt_step_size= False, adapt_mass_matrix= False)

mcmc_hmc = MCMC(hmc_kernel,
            initial_params=init_params,
            num_samples=n_samples,
            warmup_steps=n_warmup)

mcmc_hmc.run()
z_hmc = mcmc_hmc.get_samples()['points'].squeeze().cpu().numpy()
points_hmc[idx] = z_hmc


#%%
np.save('../samples/hmc_corr_gauss_003', points_hmc)

