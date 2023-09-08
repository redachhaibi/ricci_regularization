import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt

def make_grid(numsteps, 
              xcenter = 0.0, ycenter = 0.0,
              xsize =  3.0, ysize = 3.0,
              xshift = 0.0, yshift = 0.0):
    
    xs = torch.linspace(xcenter - 0.5*xsize, xcenter + 0.5*xsize, steps = numsteps) + xshift
    ys = torch.linspace(ycenter - 0.5*ysize, ycenter + 0.5*ysize, steps = numsteps) + yshift
    
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

def draw_scalar_on_grid(scalar_on_grid,plot_name="my_plot", 
        xcenter = 0.0, ycenter = 0.0,
        xsize =  3.0, ysize = 3.0,
        xshift = 0.0, yshift = 0.0,
        numsteps = 100, numticks=5,
        tick_decimals = 2):

    scalar_on_grid = scalar_on_grid.detach()

    fig, ax = plt.subplots()
    im = ax.imshow(scalar_on_grid,origin="lower")

    cbar = ax.figure.colorbar(im)
    
    xticks = np.linspace(xcenter - 0.5*xsize, xcenter + 0.5*xsize, numticks) 
    yticks = np.linspace(ycenter - 0.5*ysize, ycenter + 0.5*ysize, numticks)

    #xtick_labels = torch.round(xticks+xshift,decimals=tick_decimals)
    #ytick_labels = torch.round(yticks+yshift,decimals=tick_decimals)
    #print(xtick_labels)

    # this weird thing is done because 
    # torch.rand + torch.tolist does not do the rounding in fact!!
    
    xtick_labels = (xticks+xshift).tolist()
    ytick_labels = (yticks+yshift).tolist()

    xtick_labels = [ '%.{0}f'.format(tick_decimals) % elem for elem in xtick_labels ]
    ytick_labels = [ '%.{0}f'.format(tick_decimals) % elem for elem in ytick_labels]

    ticks_places = np.linspace(0, 1, numticks)*(numsteps-1)

    ax.set_xticks(ticks_places,labels = xtick_labels)
    ax.set_yticks(ticks_places,labels = ytick_labels)

    plt.xlabel( "x coordinate")
    plt.ylabel( "y coordinate")
    plt.axis('scaled')

    ax.set_title(plot_name)
    fig.tight_layout()
    plt.show()
    return plt