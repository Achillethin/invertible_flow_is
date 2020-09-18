#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 14:48:33 2020

@author: achillethin
"""



import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import pdb
import pyro
from pyro.infer.mcmc import HMC, MCMC, NUTS
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt

import os
import sys

from data import Dataset

from models import Encoder, Decoder
from torch_target import Gaussian_target, NN_bernoulli, MNIST_target

from torch_trajectory import trajectories


from scipy.special import logsumexp



#%%
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torchType = torch.float32

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def set_seeds(rand_seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(rand_seed)
    np.random.seed(rand_seed)
    random.seed(rand_seed)

seed = 322 # 1337 #
set_seeds(seed)

#%%
args = {}
args['data'] = "mnist" # 'lfw' # 
args['data_h'] = 28 # 32
args['data_w'] = 28 # 32
args['data_c'] = 1 # 3
args['n_data'] = -1 # whole dataset
args['train_data_size'] = 50000 # size of training data
args['n_val'] = 10000 # size of validation data # 1000
args['n_batch'] = 100 # batch size for training
args['n_batch_test'] = 10 # batch size for testing
args['n_batch_val'] = 10000 # batch size for validation # 1000
args['z_dim'] = 64
args = dotdict(args)
args.vds = 1000 ## Validation data set
args.train_data_size = 15000
args.train_batch_size = 100
args.test_batch_size = 10 ## Batch size test
args.val_batch_size = 100 ## batch size validation
#%%




dataset = Dataset(args, device)

decoder = torch.load('../../../models/decoder_mnist_Z4.pt', map_location=device)
decoder.eval()
decoder

encoder = torch.load('../../../models/encoder_mnist_Z4.pt', map_location=device)
encoder.eval()
encoder

sampling_distr = torch.distributions.Normal(loc=torch.tensor(0., dtype=torchType, device=device),
                                            scale=torch.tensor(1., dtype=torchType, device=device))

#%%
n_samples = 10000
batch_train = None
for batch in dataset.next_train_batch(): # cycle over batches
    batch_train = batch

ind = [2, 56, 74] # was ind = 2 (3_0) ind = 56 (3_1) ind = 74 (3_2)
i = 2
fixed_image = torch.tensor(batch_train[i], device=device)
# for i in ind:
#     fixed_image = torch.cat([fixed_image, batch_train[i][None, ...]], dim=0)
# plt.imshow(fixed_image[1].permute(1, 2, 0)[:, :, 0].cpu(), 'gray')
#%%

# def log_p_z_x(z, fixed_image=fixed_image):
#     #z = z['points']
# #     pdb.set_trace()
#     p_x_given_z_logits = decoder(z)
#     p_x_given_z = torch.distributions.Bernoulli(logits=p_x_given_z_logits[0])
#     expected_log_likelihood = 0
#     for i in range(len(fixed_image)):
#         expected_log_likelihood += torch.sum(p_x_given_z.log_prob(fixed_image[i][None]), [1, 2, 3])
#     log_density = expected_log_likelihood + sampling_distr.log_prob(z).sum(1)
#     return log_density

args.T = 20
def A(k):
    cond = (k < args.T) & (k > -args.T)
    torch_one = torch.tensor(1, dtype = torchType)
    return cond*torch_one

#pdb.set_trace()
#a = A(torch.arange(-args.T+1, args.T, dtype= torchType, device = device))


def log_p_z_x(z, fixed_image=fixed_image):
    #z = z['points']
    #pdb.set_trace()
    if len(z.shape)>1:
        p_x_given_z_logits = decoder(z)[0]
    else:
        p_x_given_z_logits = decoder(z[None,:])[0]
    p_x_given_z = torch.distributions.Bernoulli(logits=p_x_given_z_logits)
    expected_log_likelihood = torch.sum(p_x_given_z.log_prob(fixed_image), [1, 2, 3])
    log_density = expected_log_likelihood #+ sampling_distr.log_prob(z).sum()
    return log_density


def grad_U(z, fixed_image = fixed_image):
    z = z.detach().requires_grad_(True)
    gradu = torch.autograd.grad(log_p_z_x(z, fixed_image=fixed_image), z)[0]
    return gradu
#%%
    
#We want now to estimate the normalizing constant of the function target_energy(z, fixed_image=fixed_image)
###Three ways to do this
###Naive Monte Carlo estimator sum p(z) p(x|z) dz
###Importance sampling sum(p(z) p(x|z)/q_x(z) ) with z sampled from q_x(z)
###Our method 
naive = []
our = []
IS = []
n_attempts = 100
args.dim = 4
args.alpha = 0.9
args.elbo = False
args.dt = 0.01
args.n_particles = 100
#args.qmax = 4
#args.qmin = -4
argsp = dotdict({})
argsp.mu = torch.zeros(args.dim, dtype = torchType, device = device)
argsp.cov = torch.eye(args.dim, dtype = torchType, device = device)
args.prior_q = Gaussian_target(argsp)

args.pmin = -20
args.pmax = 20
args.T = 100
a =torch.tensor(np.ones(2* args.T-1), 
                 dtype = torchType, device = device)
argsp = dotdict({})
argsp.mu = torch.zeros(args.dim, dtype = torchType, device = device)
argsp.cov = 20*torch.eye(args.dim, dtype = torchType, device = device)
#args.prior_p = Gaussian_target(argsp)
mu, sigma = encoder(fixed_image[None])
prior_q = torch.distributions.MultivariateNormal(loc=torch.zeros(args.dim, dtype = torchType, device = device), 
                                                 covariance_matrix=torch.eye(args.dim, dtype = torchType, device = device))

def log_L(z, fixed_image=fixed_image):
    p_x_given_z_logits = decoder(z.reshape(-1, args.dim))[0].reshape(args.n_particles, -1, 28,28)
    p_x_given_z = torch.distributions.Bernoulli(logits=p_x_given_z_logits)
    expected_log_likelihood = torch.sum(p_x_given_z.log_prob(fixed_image), [2, 3])
    log_density = expected_log_likelihood #+ sampling_distr.log_prob(z).sum()
    return log_density

trajs = trajectories(args, a)

#%%
for i in tqdm(range(n_attempts)):
    naive.append(torch.logsumexp(log_L(prior_q.sample((args.n_particles,)))[0], dim = 0)-np.log(args.n_particles))
    #pdb.set_trace()  
    trajs.prop_trajectories(log_p_z_x, grad_U)
    our.append(trajs.Z_f_log(log_L).cpu().detach().numpy())
    print(logsumexp(our)-np.log(len(our)))
    #z = trajs.trajs['position'].cpu().detach().numpy()

#%%

importance = []
#%%

for i in tqdm(range(n_attempts)):
    u = prior_q.sample((args.n_particles,))
    sigmaimp = 1.2*sigma
    importance.append(torch.logsumexp(log_L(mu+sigmaimp*u)[0]-(prior_q.log_prob(u)-torch.sum(torch.log(sigmaimp))), dim = 0)-np.log(args.n_particles))



    
    
    
    
    