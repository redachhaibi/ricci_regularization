import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt

def make_grid(numsteps, xshift = 0.0, yshift = 0.0):
    
    xs = torch.linspace(-1.5, 1.5, steps = numsteps) + xshift
    ys = torch.linspace(-1.5, 1.5, steps = numsteps) + yshift
    #uniform_grid = torch.cartesian_prod(xs,ys)

    # true grid starts from left bottom corner. x is the first to increase
    tgrid = torch.cartesian_prod(ys, xs)
    tgrid = tgrid.roll(1,1)
    return tgrid

def draw_frob_norm_tensor_on_grid(plot_name,tensor_on_grid, numsteps = 100,xshift = 0.0, yshift = 0.0):
    Frob_norm_on_grid = tensor_on_grid.norm(dim=(1,2)).view(numsteps,numsteps)
    #Frob_norm_on_grid = metric_on_grid.norm(dim=(1,2)).view(numsteps,numsteps)
    Frob_norm_on_grid = Frob_norm_on_grid[1:-1,1:-1].detach()

    fig, ax = plt.subplots()
    im = ax.imshow(Frob_norm_on_grid,origin="lower")

    cbar = ax.figure.colorbar(im)
    
    ax.set_xticks((Frob_norm_on_grid.shape[0]-1)*(np.linspace(0,1,num=11)),labels=(np.linspace(-1.5,1.5,num=11)+xshift).round(1))
    ax.set_yticks((Frob_norm_on_grid.shape[1]-1)*(np.linspace(0,1,num=11)),labels=(np.linspace(-1.5,1.5,num=11)+yshift).round(1))
    plt.xlabel( "x coordinate")
    plt.ylabel( "y coordinate")
    plt.axis('scaled')

    ax.set_title(plot_name)
    fig.tight_layout()
    plt.show()
    return plt