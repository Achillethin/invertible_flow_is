import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import pdb
torchType = torch.float32

class trajectory(nn.Module):
    def __init__(self, dim, qmin, qmax, pmin, pmax, hmax, alpha, dt, a, T, device, elbo, prior_q=None, prior_p=None):
        super(trajectory, self).__init__()
        self.dim = dim
        self.qmin = qmin
        self.qmax = qmax
        self.pmin = pmin
        self.pmax = pmax
        self.hmax = hmax
        self.alpha = alpha
        self.dt = dt
        self.a = a
        self.T = T  # Linked to a : truncation
        # There could be a T_- and a T_+ -- not necessarily symmetric
        self.prior_q = prior_q
        self.prior_p = prior_p
        # self.h_fin = kwargs.hmax0
        # self.target = target
        # self.trajectory = list()
        # self.qinit = self.qmin
        # self.pinit = self.pmin
        # self.E_init =kwargs.hmax0
        # self.natt=0

    def kinetic(self, p):
        # Kinetic energy of the particle
        return 1/2 * torch.sum(p**2)

    def test_appartenance(self, q, p, target):
        ###Function that tests whether (q, p) \in O
        if (self.prior_p==None)&(self.prior_q==None):
            bool1 = (torch.max(q) <= self.qmax) & (torch.max(p) <= self.pmax)
            bool2 = (torch.min(q) >= self.qmin) & (torch.min(p) >= self.pmin)
        elif (self.prior_q==None):
            bool1 = (torch.max(q) <= self.qmax) #& (np.max(p) <= self.pmax)
            bool2 = (torch.min(q) >= self.qmin) #& (np.min(p) >= self.pmin)
        elif (self.prior_p==None):
            bool1 = (torch.max(p) <= self.pmax) #& (np.max(p) <= self.pmax)
            bool2 = (torch.min(p) >= self.pmin) #& (np.min(p) >= self.pmin)
        else:
            bool1 = torch.tensor(True)
            bool2 =torch.tensor( True)
        #bool3 = (-target.get_logdensity(q)<self.hmax)
        bool3= torch.tensor(True)
        return bool1 & bool2&bool3



    def criterion_forward_stop(self, q, p, target, hmin=None):
        # Supplementary criterion for forward stopping of the oarticles
        # original paper: force inferior to some threshold -- local minimum
        # When stop, put all remaining weight "sum product jacobians on last point ??
        # if we don't move further away ?
        if hmin is not None:
            return self.hamiltonian(q, p, target) < hmin
        else:
            return False

    def init_particle(self, target):
        # Initialise through rejection sampling on prior
        # There could be here usage of a Markov kernel for new local moves
        n_att = 0
        # for n in range(self.n_particles):
        # Add self.prior hererather than uniform
        unif = torch.distributions.Uniform(0,1)
        if (self.prior_q == None)&(self.prior_p == None):
            qs = self.qmin + unif.sample((self.dim,)) * (self.qmax-self.qmin)
            ps = self.pmin + unif.sample((self.dim,)) * (self.pmax-self.pmin)
            while not self.test_appartenance( qs, ps, target):
                qs = self.qmin + \
                    unif.sample((self.dim,)) * (self.qmax-self.qmin)
                ps = self.pmin + \
                    unif.sample((self.dim,)) * (self.pmax-self.pmin)
                n_att += 1
            self.qinit = qs
            self.pinit = ps
            self.natt = n_att
        elif (self.prior_q==None):
            qs = self.qmin + unif.sample((self.dim,)) * (self.qmax-self.qmin)
            ps = self.prior_p.get_samples()
            while not self.test_appartenance( qs, ps, target):
                qs = self.qmin + unif.sample((self.dim,)) * (self.qmax-self.qmin)
                ps = self.prior_p.get_samples()
                n_att += 1
            self.qinit = qs
            self.pinit = ps
            self.natt = n_att
        elif (self.prior_p==None):
            ps = self.pmin + unif.sample((self.dim,)) * (self.pmax-self.pmin)
            qs = self.prior_q.get_samples()
            while not self.test_appartenance( qs, ps, target):
                ps = self.pmin + unif.sample((self.dim,)) * (self.pmax-self.pmin)
                qs = self.prior_q.get_samples()
                n_att += 1
            self.qinit = qs
            self.pinit = ps
            self.natt = n_att
        else:
            qs = self.prior_q.get_samples()
            ps = self.prior_p.get_samples()
            while not self.test_appartenance( qs, ps, target):
                qs = self.prior_q.get_samples()
                ps = self.prior_p.get_samples()
                n_att += 1
            self.qinit = qs
            self.pinit = ps
            self.natt = n_att
        return qs, ps

    def prop_trajectory(self, phi_forward, phi_backward, target, grad_U, jac):
        # Compute trajectory from given initial position
        i_back = 0
        i_for = 0
        q_currb = self.qinit
        p_currb = self.pinit
        q_currf = self.qinit
        p_currf = self.pinit
        # Initialise list with initial data
        p_tot = torch.zeros((4*self.T-1, self.dim))
        q_tot = torch.zeros((4*self.T-1, self.dim))
        rho_tot = torch.zeros((4*self.T-1))
        p_tot[2*self.T-1] = self.pinit
        q_tot[2*self.T-1] = self.qinit
        # Weights
        if (self.prior_q == None)&(self.prior_p == None):
            rho_tot[2*self.T-1] = 1
        elif (self.prior_q == None):
            rho_tot[2*self.T-1] =  self.prior_p.get_density(p_currb)
        elif (self.prior_p == None):
            rho_tot[2*self.T-1] =  self.prior_q.get_density(p_currb)
        else:
            rho_tot[2*self.T-1] = self.prior_q.get_density(q_currb)* self.prior_p.get_density(p_currb)
        bool_while = (self.test_appartenance(
            q_currb, p_currb, target) & torch.tensor(i_back > -2*self.T+1))
        while bool_while:
            # Propagate backward
            i_back -= 1  # Exponent of the function applied
            q_currb, p_currb = phi_backward(q_currb, p_currb, grad_U)
            appartenance = self.test_appartenance( q_currb, p_currb, target)
            if appartenance:
                if (self.prior_q == None)&(self.prior_p == None):
                    rho_tot[2*self.T+i_back-1] =  jac(i_back)
                elif (self.prior_q == None):
                    rho_tot[2*self.T+i_back-1] = self.prior_p.get_density(p_currb) * jac(i_back)
                elif (self.prior_p == None):
                    rho_tot[2*self.T+i_back-1] = self.prior_q.get_density(p_currb) * jac(i_back)
                else:
                    rho_tot[2*self.T+i_back-1] = self.prior_q.get_density(
                        q_currb) * self.prior_p.get_density(p_currb) * jac(i_back)
                p_tot[2*self.T+i_back-1]=p_currb
                q_tot[2*self.T+i_back-1]=q_currb
            bool_while = appartenance & (i_back > -2*self.T+1)
        # Reverse position to have it increasing with k
        stop = self.criterion_forward_stop(q_currb, p_currb, target)
        keep_on = torch.tensor(not stop)
        bool_while = torch.tensor(i_for < 2*self.T-1) & (
            self.test_appartenance(q_currf, p_currf, target)) & (keep_on)

        while bool_while:
            # Propagate forward trajectory
            i_for += 1  # Exponent of the function applied underneath
            q_currf, p_currf = phi_forward(q_currf, p_currf, grad_U)
            appartenance = self.test_appartenance(q_currf, p_currf, target)
            if appartenance:
                if (self.prior_q == None)&(self.prior_p == None):
                    rho_tot[2*self.T+i_for-1] =  jac(i_for)
                elif (self.prior_q == None):
                    rho_tot[2*self.T+i_for-1] = self.prior_p.get_density(p_currf) * jac(i_for)
                elif (self.prior_p == None):
                    rho_tot[2*self.T+i_for-1] = self.prior_q.get_density(p_currf) * jac(i_for)
                else:
                    rho_tot[2*self.T+i_for-1] = self.prior_q.get_density(
                        q_currf) * self.prior_p.get_density(p_currf) * jac(i_for)
                p_tot[2*self.T+i_for-1]=p_currf
                q_tot[2*self.T+i_for-1]=q_currf
            stop = self.criterion_forward_stop(q_currb, p_currb, target)
            keep_on = torch.tensor(not stop)
            bool_while = torch.tensor(i_for < 2*self.T-1) & (appartenance) & (keep_on)
        # Now total trajectory is computed
        # Return in desired format: dict of length \tau, containing
        # q_k, p_h, E_k, k for each time step, i.e k in [\tau^-, \tau^+]
        self.tau_m = i_back
        self.tau_p = i_for
        weight = self.compute_w(self.a, rho_tot, self.T)
        # for k in range(len(k_tot)):
        #    traj.append(list([q_tot[k],p_tot[k],E_tot[k],k_tot[k], weight[k]]))
        ###Return now only trajectory part i am interested in
        traj = {'positions': q_tot[self.T:(3*self.T-1)],
                'momenta': p_tot[self.T:(3*self.T-1)],
                'tau_m': self.tau_m,
                'weights': weight,
                'rho': rho_tot[self.T:(3*self.T-1)]
                }
        self.traj = traj
        return traj

    def define_traj(self, phi_forward, phi_backward, target, grad_U, jac):
        # Overall function for new trajectory
        pin, qin = self.init_particle(target)
        self.prop_trajectory(phi_forward, phi_backward, target, grad_U, jac)
        return 

    @staticmethod
    def compute_w(a, rho_tot, T):
##numpy.convolve(rho, a, mode='same') returns array of length 4T-1
        # rho is not logarithmic
        ###To do convolve but error ?
        #den_weight = np.convolve(rho_tot, a[::-1], mode='same')
        den_weight = F.conv1d(rho_tot.view(1, 1, -1), a.view(1, 1, -1), padding = T-1).view(-1)
        num_weight = torch.flip(a, (0,))*rho_tot[T:3*T-1]
##Index 0 is 2T-1 (python notations) here
        ###Write it by hand
        # den_weight1 = np.zeros(np.shape(num_weight))
        # for k in range(T,3*T-1): ###Support of a -- on python arrays
        #     den_weight1[k] = 0
        #     for tj in range(-T+1,T): ###Support of a #Shift of 2T-1 with array stockage
        #         den_weight1[k]+= a[::-1][tj+2*T-1]*rho_tot[tj+k]
        # # ##Test simple
        # print(den_weight[T:3*T-1]-den_weight1[T:3*T-1]) 
        w = num_weight/den_weight[T:3*T-1]
        w[w!=w]=0.
        return w


class trajectories(nn.Module):
    def __init__(self, kwargs, a=None):
        super(trajectories, self).__init__()
        self.dim = kwargs.dim
        self.qmin = kwargs.qmin
        self.qmax = kwargs.qmax
        self.pmin = kwargs.pmin
        self.pmax = kwargs.pmax
        self.hmax = kwargs.hmax
        self.device =kwargs.device
        self.elbo = kwargs.elbo
        self.alpha = nn.Parameter(torch.tensor(kwargs.alpha, dtype = torchType, device = self.device), 
                                  requires_grad= self.elbo)
        self.dt = nn.Parameter(torch.tensor(kwargs.dt, dtype = torchType, device = self.device), 
                               requires_grad= self.elbo)
        self.n_particles = kwargs.n_particles
        self.a = a
        self.T = kwargs.T
        self.prior_q = kwargs.prior_q
        self.prior_p = kwargs.prior_p
        # Renormalizing the prior
        # self.tau_m = 0
        # self.tau_p = 0
        # self.h_fin = kwargs.hmax0
        # self.target = target
        # self.trajectory = list()
        # self.qinit = self.qmin
        # self.pinit = self.pmin
        # self.E_init =kwargs.hmax0
        # self.natt=0
  # Sigma covariance matrix of p ==Id here
# Where to input grad_U here ??

    def phi_forward(self, q, p, grad_U):
        # Define operators for all trajectories
        pt = self.alpha * p
        p1 = pt - self.dt * grad_U(q)
        q1 = q + self.dt * p1
        return q1, p1

    def phi_backward(self, q, p, grad_U):
       # inverse operator
        q1 = q - self.dt * p
        pt = self.alpha * p
        p1 = pt + self.dt*self.alpha * grad_U(q1)
        return q1, p1

    def jac(self, k, init_point=None):
        # Jacobian of the operator to the power k
        return torch.pow(self.alpha, k * self.dim)

    
    def hamiltonian(self, q, p, target):
        # Hamiltonian of the particle: sum of potential and kinetic energies
        # Linked to posterior densities and not prior
        potential = - target.get_logdensity(q)
        return self.kinetic(p) + potential


    def prop_trajectories(self,  target, grad_U):
        # Propagate n_particles trajectories
        # Returns a list with all dict for single trajectories
        rho_n = torch.zeros((self.n_particles, 2*self.T-1))
        q_n = torch.zeros((self.n_particles, 2*self.T-1,self.dim))
        p_n = torch.zeros((self.n_particles, 2*self.T-1,self.dim))
        w_n = torch.zeros((self.n_particles, 2*self.T-1))
        tau_m = torch.zeros(self.n_particles)
        for i in range(self.n_particles):
            # Sample the trajectory
            traj_aux = trajectory(self.dim, self.qmin, self.qmax, self.pmin,
                                  self.pmax, self.hmax, self.alpha, self.dt, self.a, self.T,
                                  self.device, self.elbo, self.prior_q, self.prior_p)
            traj_aux.define_traj(self.phi_forward, self.phi_backward, target, grad_U, self.jac)
            rho_n[i]=traj_aux.traj['rho']
            q_n[i]=traj_aux.traj['positions']
            p_n[i]=traj_aux.traj['momenta']
            w_n[i]=traj_aux.traj['weights']
            tau_m[i]=traj_aux.traj['tau_m']
            # print('tau_m: ', traj_aux.tau_m)
            # print('tau_p :', traj_aux.tau_p)
        self.trajs = {'position': q_n,
                        'momentum': p_n,
                        'tau_m': tau_m,
                        'weight': w_n,
                        'rho': rho_n
                        }
        return self.trajs

    def prop_trajectories_gibbs(self, I_curr, target, grad_U):
        # Propagate n_particles trajectories
        # Returns a list with all dict for single trajectories
        rho_n = torch.zeros((self.n_particles, 2*self.T-1))
        q_n = torch.zeros((self.n_particles, 2*self.T-1,self.dim))
        p_n = torch.zeros((self.n_particles, 2*self.T-1,self.dim))
        w_n = torch.zeros((self.n_particles, 2*self.T-1))
        tau_m = torch.zeros(self.n_particles)
        for i in range(self.n_particles):
            # Sample the trajectory
            if i == I_curr:
                rho_n[i]=self.trajs['rho'][I_curr]
                q_n[i]=self.trajs['position'][I_curr]
                p_n[i]=self.trajs['momentum'][I_curr]
                w_n[i]=self.trajs['weight'][I_curr]
                tau_m[i]=self.trajs['tau_m'][I_curr]
            else:
                traj_aux = trajectory(self.dim, self.qmin, self.qmax, self.pmin,
                                  self.pmax, self.hmax, self.alpha, self.dt, self.a, self.T,
                                  self.prior_q, self.prior_p)
                traj_aux.define_traj(self.phi_forward, self.phi_backward, target, grad_U, self.jac)
                rho_n[i]=traj_aux.traj['rho']
                q_n[i]=traj_aux.traj['positions']
                p_n[i]=traj_aux.traj['momenta']
                w_n[i]=traj_aux.traj['weights']
                tau_m[i]=traj_aux.traj['tau_m']
                # print('tau_m: ', traj_aux.tau_m)
                # print('tau_p :', traj_aux.tau_p)
        self.trajs = {'position': q_n,
                        'momentum': p_n,
                        'tau_m': tau_m,
                        'weight': w_n,
                        'rho': rho_n
                        }
        return self.trajs
    
    def Z_f_i(self, f):
        # Do the mean over trajectory with corresponding weights
        zfi = f(self.trajs['position'],self.trajs['momentum'])*self.trajs['weight']
        if self.T>1:
            zf = torch.sum(zfi, dim = 1)
        else:
            zf = zfi
        return zf
    
    def Z_f_i_log(self, f):
        # Do the mean over trajectory with corresponding weights
        zfi = torch.log(f(self.trajs['position'],self.trajs['momentum']))+torch.log(self.trajs['weight'])
        if self.T>1:
            zf = torch.logsumexp(zfi, dim = 1)
        else:
            zf = zfi
        return zf

    def Z_f_log(self, f):
        # Do the mean over trajectory with corresponding weights
        ##f is logarithmic here
        #pdb.set_trace()
        zfi = f(self.trajs['position']).view(self.n_particles, -1)+torch.log(self.trajs['weight'])
        zf = torch.logsumexp(zfi, dim = 1)
        return torch.logsumexp(zf, -1) #- np.log(self.n_particles)

    def Z_f(self, f):
         #zf = list()
         #for i in range(self.n_particles):
         #    f_k = f(self.trajs['position'][i,self.T:(3*self.T-1)], self.trajs['momentum'][i,self.T:(3*self.T-1)])      
         #    #print(f_k)
         #    zf.append(np.sum(f_k*self.trajs['weight'][i]))
        # print(f_k*self.trajs['weight'])
        # print(zfi)
        zfi = f(self.trajs['position'], self.trajs['momentum'])*self.trajs['weight']
        if self.T>1:
            zf = torch.sum(zfi, dim = 1)
        else:
            zf = zfi
        return torch.mean(zf)
