import torch
import numpy as np
import torchvision

import torch.func as TF
from functorch import jacrev,jacfwd
import matplotlib.pyplot as plt
import functools

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

def Ch_jacfwd (u, function):
    g = metric_jacfwd(u,function)
    g_inv = torch.inverse(g)
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

# Functions with sphere and Lobachevsky plane pullback metrics

# Sphere embedding
# Input: u is a 2d-vector with longitude and lattitude
# Outut: output contains the 3d coordinates of sphere and padded with zeros (781 dimension)
#        -> 784 dim in total
def my_fun_sphere(u):
    u = u.flatten()
    output = torch.cat((torch.sin(u[0])*torch.cos(u[1]).unsqueeze(0),
    torch.sin(u[0])*torch.sin(u[1]).unsqueeze(0),
    torch.cos(u[0]).unsqueeze(0)),dim=-1)
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
