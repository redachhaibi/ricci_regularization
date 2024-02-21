import torch
import numpy as np
import torchvision

import torch.func as TF
from functorch import jacrev,jacfwd
import matplotlib.pyplot as plt
import functools

# jacfwd
"""
def metric_jacfwd(u, function):
    u = u.unsqueeze(0) # newline
    jac = jacfwd(function)(u).squeeze().reshape(-1,u.shape[-1])
    #jac = jacfwd(function)(u).squeeze()
    # squeezing is needed to get rid of 1-dimentions
    metric = torch.matmul(jac.T,jac)
    return metric
"""

def metric_jacfwd(u, function, latent_space_dim=2):
    u = u.reshape(-1,latent_space_dim)
    jac = jacfwd(function)(u)
    jac = jac.reshape(-1,latent_space_dim)
    metric = torch.matmul(jac.T,jac)
    return metric

metric_jacfwd_vmap = TF.vmap(metric_jacfwd)

def metric_der_jacfwd (u, function):
    metric = functools.partial(metric_jacfwd, function=function)
    dg = jacfwd(metric)(u).squeeze()
    # squeezing is needed to get rid of 1-dimentions 
    # occuring when using jacfwd
    return dg
metric_der_jacfwd_vmap = TF.vmap(metric_der_jacfwd)

def Ch_jacfwd (u, function, eps = 0.01):
    g = metric_jacfwd(u,function)
    #g_inv = torch.inverse(g)
    d = g.shape[0]
    g_inv = torch.inverse(g + eps*torch.eye(d))
    dg = metric_der_jacfwd(u,function)
    Ch = 0.5*(torch.einsum('im,mkl->ikl',g_inv,dg)+
              torch.einsum('im,mlk->ikl',g_inv,dg)-
              torch.einsum('im,klm->ikl',g_inv,dg)
              )
    return Ch
Ch_jacfwd_vmap = TF.vmap(Ch_jacfwd)

def Ch_der_jacfwd (u, function):
    Ch = functools.partial(Ch_jacfwd, function=function)
    dCh = jacfwd(Ch)(u).squeeze()
    return dCh

Ch_der_jacfwd_vmap = TF.vmap(Ch_der_jacfwd)

# Riemann curvature tensor (3,1)
def Riem_jacfwd(u, function):
    Ch = Ch_jacfwd(u, function)
    Ch_der = Ch_der_jacfwd(u, function)

    Riem = torch.einsum("iljk->ijkl",Ch_der) - torch.einsum("ikjl->ijkl",Ch_der)
    Riem += torch.einsum("ikp,plj->ijkl", Ch, Ch) - torch.einsum("ilp,pkj->ijkl", Ch, Ch)
    return Riem

def Ric_jacfwd(u, function):
    Riemann = Riem_jacfwd(u, function)
    Ric = torch.einsum("cacb->ab",Riemann)
    return Ric

Ric_jacfwd_vmap = TF.vmap(Ric_jacfwd)

def Sc_jacfwd (u, function, eps = 0.01):
    metric = metric_jacfwd(u, function=function)
    Ricci = Ric_jacfwd(u, function=function)
    d = metric.shape[0]
    metric_inv = torch.inverse(metric + eps * torch.eye(d))
    #metric_inv = torch.inverse(metric)
    Sc = torch.einsum('ab,ab',metric_inv,Ricci)
    return Sc
Sc_jacfwd_vmap = TF.vmap(Sc_jacfwd)

# jacrev
def metric_jacrev(u, function):
    jac = jacrev(function)(u).squeeze()
    # squeezing is needed to get rid of 1-dimentions
    metric = torch.matmul(jac.T,jac)
    return metric

"""
def metric_jacrev(u, function, latent_space_dim=2):
    u = u.reshape(-1,latent_space_dim)
    jac = jacrev(function)(u)
    jac = jac.reshape(-1,latent_space_dim)
    metric = torch.matmul(jac.T,jac)
    return metric
"""

metric_jacrev_vmap = TF.vmap(metric_jacrev)

def metric_der_jacrev (u, function):
    metric = functools.partial(metric_jacrev, function=function)
    dg = jacrev(metric)(u).squeeze()
    # squeezing is needed to get rid of 1-dimentions 
    # occuring when using jacrev
    return dg
metric_der_jacrev_vmap = TF.vmap(metric_der_jacrev)

def Ch_jacrev (u, function):
    g = metric_jacrev(u,function)
    g_inv = torch.inverse(g)
    dg = metric_der_jacrev(u,function)
    Ch = 0.5*(torch.einsum('im,mkl->ikl',g_inv,dg)+
              torch.einsum('im,mlk->ikl',g_inv,dg)-
              torch.einsum('im,klm->ikl',g_inv,dg)
              )
    return Ch
Ch_jacrev_vmap = TF.vmap(Ch_jacrev)

def Ch_der_jacrev (u, function):
    Ch = functools.partial(Ch_jacrev, function=function)
    dCh = jacrev(Ch)(u).squeeze()
    return dCh

Ch_der_jacrev_vmap = TF.vmap(Ch_der_jacrev)

# Riemann curvature tensor (3,1)
def Riem_jacrev(u, function):
    Ch = Ch_jacrev(u, function)
    Ch_der = Ch_der_jacrev(u, function)

    Riem = torch.einsum("iljk->ijkl",Ch_der) - torch.einsum("ikjl->ijkl",Ch_der)
    Riem += torch.einsum("ikp,plj->ijkl", Ch, Ch) - torch.einsum("ilp,pkj->ijkl", Ch, Ch)
    return Riem

def Ric_jacrev(u, function):
    Riemann = Riem_jacrev(u, function)
    Ric = torch.einsum("cacb->ab",Riemann)
    return Ric

Ric_jacrev_vmap = TF.vmap(Ric_jacrev)

def Sc_jacrev (u, function):
    metric = metric_jacrev(u, function=function)
    Ricci = Ric_jacrev(u, function=function)
    metric_inv = torch.inverse(metric)
    Sc = torch.einsum('ab,ab',metric_inv,Ricci)
    return Sc
Sc_jacrev_vmap = TF.vmap(Sc_jacrev)


# polynomial local diffeomorphysm of R^2
def my_fun_polinomial(u):
    u = u.flatten()
    x = u[0]
    y = u[1]

    x_out = x**2 + y + 37*x
    y_out = y**3+x*y

    x_out = x_out.unsqueeze(0)
    y_out = y_out.unsqueeze(0)
    output = torch.cat((x_out, y_out),dim=-1)
    output = output.flatten()
    return output

# Functions with sphere and Lobachevsky plane pullback metrics
# Sphere embedding
# Input: u is a 2d-vector with longitude and lattitude
# Outut: output contains the 3d coordinates of sphere and padded with zeros (781 dimension)
#        -> 784 dim in total
# u = (\theta, \phi)
# ds^2 = (d\theta)^2 + sin^2(\theta)*(d\phi)^2
def my_fun_sphere(u):
    u = u.flatten()
    
    x = torch.sin(u[0])*torch.cos(u[1])
    y = torch.sin(u[0])*torch.sin(u[1])
    z = torch.cos(u[0])

    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    z = z.unsqueeze(0)
    output = torch.cat((x, y, z),dim=-1)
    output = torch.cat((output.unsqueeze(0),torch.zeros(781).unsqueeze(0)),dim=1)
    output = output.flatten()
    return output

# Hyperbolic plane embedding
# Partial embedding (for y>c) of Lobachevsky plane to R^3 
# (formally here it is R^784)
# ds^2 = 1/y^2(dx^2 + dy^2)
# http://www.antoinebourget.org/maths/2018/08/08/embedding-hyperbolic-plane.html
def my_fun_lobachevsky(u, c=0.01):
    u = u.flatten()
    x = u[0]
    y = u[1]
    t = torch.acosh(y/c)
    x0 = t - torch.tanh(t)
    x1 = (1/torch.sinh(t))*torch.cos(x/c)
    x2 = (1/torch.sinh(t))*torch.sin(x/c)
    output = torch.cat((x0.unsqueeze(0),x1.unsqueeze(0),x2.unsqueeze(0)),dim=-1)
    output = torch.cat((output.unsqueeze(0),torch.zeros(781).unsqueeze(0)),dim=1)
    output = output.flatten()
    return output


# as a class!
class RiemannianGeometry():
    def __init__(self, latent_space_dim, function, AD_method, eps = 0):
        self.latent_space_dim = latent_space_dim
        self.function = function
        self.eps = eps
        self.AD_method = AD_method
    def metric(self, point):
        #print()
        point = point.reshape(-1,self.latent_space_dim)
        jac = self.AD_method(self.function)(point)
        jac = jac.reshape(-1,self.latent_space_dim)
        metric = torch.matmul(jac.T,jac)
        return metric
    def metric_vmap(self,tensor):
        return TF.vmap(self.metric)(tensor)
    def metric_der(self, point):
        metric = self.metric
        dg = self.AD_method(metric)(point).squeeze()
        # squeezing is needed to get rid of 1-dimentions 
        # occuring when using jacfwd
        return dg
    def metric_der_vmap(self,tensor):
        return TF.vmap(self.metric_der)(tensor)
    def Ch (self, point):
        g = self.metric(point)
        g_inv = torch.inverse(g + self.eps*torch.eye(self.latent_space_dim))
        dg = self.metric_der(point)
        Ch = 0.5*(torch.einsum('im,mkl->ikl',g_inv,dg)+
                torch.einsum('im,mlk->ikl',g_inv,dg)-
                torch.einsum('im,klm->ikl',g_inv,dg)
                )
        return Ch
    def Ch_vmap(self,tensor):
        return TF.vmap(self.Ch)(tensor)
    def Ch_der (self, point):
        dCh = self.AD_method(self.Ch)(point).squeeze()
        return dCh
    def Ch_der_vmap(self,tensor):
        return TF.vmap(self.Ch_der)(tensor)
    # Riemann curvature tensor (3,1)
    def Riem(self, point):
        Ch = self.Ch(point)
        Ch_der = self.Ch_der(point)

        Riem = torch.einsum("iljk->ijkl",Ch_der) - torch.einsum("ikjl->ijkl",Ch_der)
        Riem += torch.einsum("ikp,plj->ijkl", Ch, Ch) - torch.einsum("ilp,pkj->ijkl", Ch, Ch)
        return Riem

    def Ric(self, point):
        Riemann = self.Riem(point)
        Ric = torch.einsum("cacb->ab",Riemann)
        return Ric
    def Ric_vmap(self,tensor):
        return TF.vmap(self.Ric)(tensor)

    def Sc (self, point):
        metric = self.metric(point)
        Ricci = self.Ric(point)
        metric_inv = torch.inverse(metric + self.eps * torch.eye(self.latent_space_dim))
        Sc = torch.einsum('ab,ab',metric_inv,Ricci)
        return Sc
    def Sc_vmap(self,tensor):
        return TF.vmap(self.Sc)(tensor)
    
