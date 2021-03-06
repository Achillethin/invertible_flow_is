#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 15:19:26 2020

@author: achillethin
"""
import math
import pdb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from scipy.stats import gamma, invgamma
torchType = torch.float32
import hamiltorch

class Target(nn.Module):
    """
    Base class for a custom target distribution
    """

    def __init__(self, kwargs):
        super(Target, self).__init__()
        self.device = kwargs.device
        self.torchType = torchType
        self.device_zero = torch.tensor(0., dtype=kwargs.torchType, device=self.device)
        self.device_one = torch.tensor(1., dtype=kwargs.torchType, device=self.device)

    def get_density(self, x):
        """
        The method returns target density, estimated at point x
        Input:
        x - datapoint
        Output:
        density - p(x)
        """
        # You should define the class for your custom distribution
        raise NotImplementedError

    def get_logdensity(self, x):
        """
        The method returns target logdensity, estimated at point x
        Input:
        x - datapoint
        Output:
        log_density - log p(x)
        """
        # You should define the class for your custom distribution
        raise NotImplementedError

    def get_samples(self, n):
        """
        The method returns samples from the distribution
        Input:
        n - amount of samples
        Output:
        samples - samples from the distribution
        """
        # You should define the class for your custom distribution
        raise NotImplementedError

class Gaussian_target(Target):
    """
    1 gaussian (multivariate)
    """

    def __init__(self, kwargs):
        super(Gaussian_target, self).__init__(kwargs)
        self.device = kwargs.device
        self.torchType = torchType        
        self.mu = kwargs['mu']  # list of locations for each of these gaussians
        self.cov = kwargs['cov']  # list of covariance matrices for each of these gaussians
        self.dist = torch.distributions.MultivariateNormal(loc=self.mu, covariance_matrix=self.cov)

    def get_density(self, x):
        """
        The method returns target density
        Input:
        x - datapoint
        Output:
        density - p(x)
        """
         
        return self.get_logdensity(x).exp()

    def get_logdensity(self, x):
        """
        The method returns target logdensity
        Input:
        x - datapoint
        Output:
        log_density - log p(x)
        """
        return self.dist.log_prob(x)
    
    def get_samples(self, n=1):
        """
        The method returns samples from the distribution
        Input:
        n - amount of samples
        Output:
        samples - samples from the distribution
        """
        # You should define the class for your custom distribution
        
        return self.dist.sample((n,))



class Gaussian_mixture(Target):
    """
    Mixture of n gaussians (multivariate)
    """

    def __init__(self, kwargs):
        super(Gaussian_mixture, self).__init__(kwargs)
        self.device = kwargs.device
        self.torchType = torchType
        self.num = kwargs['num_gauss']
        self.pis = kwargs['p_gaussians']
        self.locs = kwargs['locs']  # list of locations for each of these gaussians
        self.covs = kwargs['covs']  # list of covariance matrices for each of these gaussians
        self.peak = [None] * self.num
        for i in range(self.num):
            self.peak[i] = torch.distributions.MultivariateNormal(loc=self.locs[i], covariance_matrix=self.covs[i])

    def get_density(self, x):
        """
        The method returns target density
        Input:
        x - datapoint
        z - latent vaiable
        Output:
        density - p(x)
        """
        density = self.get_logdensity(x).exp()
        return density

    def get_logdensity(self, x):
        """
        The method returns target logdensity
        Input:
        x - datapoint
        z - latent vaiable
        Output:
        log_density - log p(x)
        """
        log_p = torch.tensor([], device=self.device)
        for i in range(self.num):
            log_paux = (torch.log(self.pis[i]) + self.peak[i].log_prob(x)).view(-1, 1)
            log_p = torch.cat([log_p, log_paux], dim=-1)
        log_density = torch.logsumexp(log_p, dim=1)  # + torch.tensor(1337., device=self.device)
        return log_density



class Uniform(Target):
    ##Uniform on a box of dimension dim
    def __init__(self,kwargs):
        super(Uniform, self).__init__(kwargs)
        self.device = kwargs.device
        self.torchType = torchType
        self.xmin = kwargs.x_min
        self.xmax = kwargs.x_max
        self.dim = kwargs.dim
        self.dist = torch.distributions.Uniform(self.xmin,self.xmax)
        
    def get_density(self, x):
        """
        The method returns target density, estimated at point x
        Input:
        x - datapoint
        Output:
        log p(x)
        """
        
        return self.dist.log_prob(x).exp()
        # You should define the class for your custom distribution

    def get_logdensity(self,  x):
        """
        The method returns target logdensity, estimated at point x
        Input:
        x - datapoint
        Output:
        log_density - log p(x)
        """
        return self.dist.log_prob(x)

    def get_samples(self, n=1):
        """
        The method returns samples from the distribution
        Input:
        n - amount of samples
        Output:
        samples - samples from the distribution
        """
        # You should define the class for your custom distribution
        samples =self.dist.sample((n, self.dim))
        return samples                                     



class NN_bernoulli(Target):
    """
    Density for NN with Bernoulli output
    """

    def __init__(self, kwargs, model, device):
        super(NN_bernoulli, self).__init__(kwargs, device)
        self.decoder = model
        self.prior = torch.distributions.Normal(loc=self.device_zero, scale=self.device_one)

    def get_density(self, x, z):
        """
        The method returns target density
        Input:
        x - datapoint
        z - latent vaiable
        Output:
        density - p(x, z)
        """
        density = self.get_logdensity(x).exp()
        return density

    def get_logdensity(self, x, z, prior=None, args=None, prior_flow=None):
        """
        The method returns target logdensity
        Input:
        x - datapoint
        z - latent vaiable
        Output:
        log_density - log p(x, z)
        """
        p_x_given_z_logits = self.decoder(z)
        p_x_given_z = torch.distributions.Bernoulli(logits=p_x_given_z_logits[0])
        if (len(x.shape) == 4):
            expected_log_likelihood = torch.sum(p_x_given_z.log_prob(x), [1, 2, 3])
        else:
            expected_log_likelihood = torch.sum(p_x_given_z.log_prob(x), 1)
        if prior_flow:
            log_likelihood = expected_log_likelihood
            log_density = log_likelihood + prior(args, z, prior_flow)
        else:
            log_density = expected_log_likelihood + self.prior.log_prob(z).sum(1)
        return log_density


class MNIST_target(Target):
    def __init__(self, kwargs, device):
        super(MNIST_target, self).__init__(kwargs, device)
        self.true_x = kwargs["true_x"]  # the image we're trying to reconstruct
        self.decoder = kwargs[
            "decoder"]  # SOTA network with saved weights, which gives a good approximation of p_theta(z | x)
        self.prior = torch.distributions.Normal(loc=self.device_zero, scale=self.device_one)
        self.decoder.eval()

    def get_logdensity(self, z, x=None):
        """
        The method returns target logdensity
        Input:
        x - datapoint
        z - latent vaiable
        Output:
        log_density - log p(x, z)
        """
        # pdb.set_trace()
        if x is not None:
            if not torch.all(torch.eq(x, self.true_x)):
                raise AttributeError
            x = self.true_x
        else:
            pdb.set_trace()
        p_x_given_z_logits = self.decoder(z)
        p_x_given_z = torch.distributions.Bernoulli(logits=p_x_given_z_logits[0])
        expected_log_likelihood = 0
        # pdb.set_trace()
        for i in range(len(x)):
            expected_log_likelihood += torch.sum(p_x_given_z.log_prob(x[i][None]), [1, 2, 3])
        log_density = expected_log_likelihood + self.prior.log_prob(z).sum(1)
        return log_density

    def get_density(self, z, x=None):
        return self.get_logdensity(z).exp()

    def get_samples(self, n=1):
        # we can approximate by a gaussian for eps small enough
        # return self.std_normal.sample((n, self.dim))
        raise NotImplementedError

    def show_x(self, x=None):
        if not x:
            x = self.true_x
        # pdb.set_trace()
        np_image = x.cpu().numpy()
        plt.imshow(np_image[0, 0, :, :])
        plt.show()

    def get_pictures(self, z, nrow=5):
        x_logit = self.decoder(z)[0]
        #         print(x_logit.shape)
        out = torchvision.utils.make_grid(x_logit, nrow)
        #         print(out.shape)
        out = out.cpu().numpy()[0]
        plt.imshow(out)
        plt.show()

class BNAF_examples(Target):

    def __init__(self, kwargs):
        super(BNAF_examples, self).__init__(kwargs)
        self.data = kwargs.bnaf_data
        self.max = kwargs.qmax
        self.min = kwargs.qmin

    def get_logdensity(self, z, x=None):
        #if (torch.max(z)>self.max)|(torch.min(z)<self.min):
        #    return torch.tensor(-math.inf, dtype = torchType, requires_grad= True)
        if self.data == 't1':
            if len(z.shape) == 1:
                z = z.view(1, 2)
            z_norm = torch.norm(z, 2, 1)
            add1 = 0.5 * ((z_norm - 2) / 0.4) ** 2
            add2 = - torch.log(torch.exp(-0.5 * ((z[:, 0] - 2) / 0.6) ** 2) + \
                               torch.exp(-0.5 * ((z[:, 0] + 2) / 0.6) ** 2) + 1e-9)
            return -add1 - add2

        elif self.data == 't2':
            if len(z.shape) == 1:
                z = z.view(1, 2)
            w1 = torch.sin(2 * np.pi * z[:, 0] / 4)
            return -0.5 * ((z[:, 1] - w1) / 0.4) ** 2
        elif self.data == 't3':
            if len(z.shape) == 1:
                z = z.view(1, 2)
            w1 = torch.sin(2 * np.pi * z[:, 0] / 4)
            w2 = 3 * torch.exp(-0.5 * ((z[:, 0] - 1) / 0.6) ** 2)
            in1 = torch.exp(-0.5 * ((z[:, 1] - w1) / 0.35) ** 2)
            in2 = torch.exp(-0.5 * ((z[:, 1] - w1 + w2) / 0.35) ** 2)
            return torch.log(in1 + in2 + 1e-9)
        elif self.data == 't4':
            if len(z.shape) == 1:
                z = z.view(1, 2)
            w1 = torch.sin(2 * np.pi * z[:, 0] / 4)
            w3 = 3 * torch.sigmoid((z[:, 0] - 1) / 0.3)
            in1 = torch.exp(-0.5 * ((z[:, 1] - w1) / 0.4) ** 2)
            in2 = torch.exp(-0.5 * ((z[:, 1] - w1 + w3) / 0.35) ** 2)
            return torch.log(in1 + in2 + 1e-9)
        else:
            raise RuntimeError

    def get_density(self, z, x=None):
        density = self.distr.log_prob(z).exp()
        return density

    def get_samples(self, n):
        return torch.stack(hamiltorch.sample(log_prob_func=self.get_logdensity, params_init=torch.zeros(2),
                                             num_samples=n, step_size=.1, num_steps_per_sample=20))