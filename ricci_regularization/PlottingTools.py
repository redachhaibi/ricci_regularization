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


# plot recostructions for mnist dataset
def plot_ae_outputs(test_dataset,encoder,decoder,n=10,D=784):
    plt.figure(figsize=(16,4.5))
    targets = test_dataset.targets.numpy()
    t_idx = {i:np.where(targets==i)[0][0] for i in range(n)}
    for i in range(n):
      ax = plt.subplot(2,n,i+1)
      img = test_dataset[t_idx[i]][0].unsqueeze(0)
      #encoder.eval()
      #decoder.eval()
      with torch.no_grad():
         #rec_img  = decoder(encoder(img))
         rec_img  = decoder(encoder(img.reshape(1,D))).reshape(1,28,28)
      plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
        ax.set_title('Original images')
      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')  
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
         ax.set_title('Reconstructed images')
    plt.show()   
# borrowed from https://gist.github.com/jakevdp/91077b0cae40f8f8244a
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)
def plot3losses(mse_train_list,uniform_train_list,curv_train_list):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6,18))
    
    axes[0].semilogy(mse_train_list, color = 'tab:red')
    axes[0].set_ylabel('MSE')
    
    axes[1].semilogy(uniform_train_list, color = 'tab:olive')
    axes[1].set_ylabel('Uniform loss')
    
    axes[2].semilogy(curv_train_list, color = 'tab:blue')
    axes[2].set_ylabel('Curvature')
    for i in range(3):
        axes[i].set_xlabel('Batches')
    #fig.show()
    plt.show()
    return fig,axes