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
    plt.figure(figsize=(16*n/10,4.5))
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

def plot_ae_outputs_selected(test_dataset,encoder,decoder,selected_labels = None,D=784,dpi=400):
    if selected_labels is None:
        n = 10
    else:
        n = len(selected_labels)
    plt.figure(figsize=(6 * n / 10, 1.5), dpi=dpi)
    targets = test_dataset.targets.numpy()
    t_idx = {i: np.where(targets == i)[0][0] for i in selected_labels}
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        img = test_dataset[t_idx[selected_labels[i]]][0].unsqueeze(0)
        with torch.no_grad():
            rec_img = decoder(encoder(img.reshape(1, D))).reshape(1, 28, 28)
        plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Original images', fontsize=6)
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Reconstructed images', fontsize=6)
    plt.show() 

# borrowed from https://gist.github.com/jakevdp/91077b0cae40f8f8244a
def discrete_cmap(N, base_cmap=None, bright_colors = False):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    if (N==2) and (bright_colors ==True):
        color_list = base(np.array([0.3,0.85]))

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

from scipy import signal
def plot9losses(mse_train_list,curv_train_list,g_inv_train_list,numwindows1=50,numwindows2=200):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18,18))
    
    win50 = signal.windows.hann(numwindows1)
    win200 = signal.windows.hann(numwindows2)

    axes[0,0].semilogy(mse_train_list, color = 'tab:red')
    axes[0,0].set_ylabel('MSE')

    axes[0,1].semilogy(signal.convolve(mse_train_list, win50, mode='same') / sum(win50), color = 'tab:red')
    #axes[0,1].set_ylabel('MSE')

    axes[0,2].semilogy(signal.convolve(mse_train_list, win200, mode='same') / sum(win200), color = 'tab:red')
    #axes[0,2].set_ylabel('MSE')
    
    axes[1,0].semilogy(curv_train_list, color = 'tab:olive')
    axes[1,0].set_ylabel('Curvature')

    axes[1,1].semilogy(signal.convolve(curv_train_list, win50, mode='same') / sum(win50), color = 'tab:olive')
    #axes[1,1].set_ylabel('Curvature')

    axes[1,2].semilogy(signal.convolve(curv_train_list, win200, mode='same') / sum(win200), color = 'tab:olive')
    #axes[1,2].set_ylabel('Curvature')
    
    axes[2,0].semilogy(g_inv_train_list, color = 'tab:blue')
    axes[2,0].set_ylabel('$\|G^{-1}\|_F$')

    axes[2,1].semilogy(signal.convolve(g_inv_train_list, win50, mode='same') / sum(win50), color = 'tab:blue')
    #axes[2,1].set_ylabel('$\|G^{-1}\|_F$')

    axes[2,2].semilogy(signal.convolve(g_inv_train_list, win200, mode='same') / sum(win200), color = 'tab:blue')
    #axes[2,2].set_ylabel('$\|G^{-1}\|_F$')

    for i in range(3):
        for j in range(3):
            if i==2:
                axes[i,j].set_xlabel('Batches')
    fig.show()
    return fig,axes

import matplotlib.colors as mcolors
def plotlosses(**dict_of_losses):
    number_of_plots = len(dict_of_losses)
    fig, axes = plt.subplots(nrows=number_of_plots, ncols=1, figsize=(6, number_of_plots*6))
    i = 0
    color_iterable = iter(mcolors.TABLEAU_COLORS)
    for name,loss_list in dict_of_losses.items():
        if number_of_plots == 1:
            axes = [axes]
        axes[i].semilogy(loss_list, color = next(color_iterable))
        axes[i].set_ylabel(name)
        axes[i].set_xlabel('Batches')
        i += 1
    plt.show()
    return fig,axes

def plotfromdict(dict_of_losses):
    number_of_plots = len(dict_of_losses)
    fig, axes = plt.subplots(nrows=number_of_plots, ncols=1, figsize=(6, number_of_plots*6))
    i = 0
    color_iterable = iter(mcolors.TABLEAU_COLORS)
    for name,loss_list in dict_of_losses.items():
        if number_of_plots == 1:
            axes = [axes]
        try:
            newcolor = next(color_iterable)
        except StopIteration:
            color_iterable = iter(mcolors.TABLEAU_COLORS)
            newcolor = next(color_iterable)
        axes[i].semilogy(loss_list, color = newcolor)
        axes[i].set_ylabel(name)
        axes[i].set_xlabel('Batches')
        i += 1
    plt.show()
    return fig,axes

def plotsmart(dictplots):
    number_of_plots = len(dictplots)
    fig, axes = plt.subplots(nrows=number_of_plots, ncols=1, figsize=(4, number_of_plots*4))
    i = 0
    color_iterable = iter(mcolors.TABLEAU_COLORS)
    for plot_name,plot_info in dictplots.items():
        if number_of_plots == 1:
            axes = [axes]
        
        for legend, curve in plot_info["data"].items(): 
            try:
                newcolor = next(color_iterable)
            except StopIteration:
                color_iterable = iter(mcolors.TABLEAU_COLORS)
                newcolor = next(color_iterable)
            #end except
                
            if (legend=="max")|(legend=="min"):
                linestyle = 'dashed'
            else:
                linestyle = 'solid'
            axes[i].semilogy(curve, color = newcolor,label = legend,ls = linestyle)
        #end for
        axes[i].set_ylabel(plot_info["yname_latex"])
        axes[i].set_xlabel('Batches')
        axes[i].legend(loc="lower left")
        i += 1
    plt.show()
    return fig,axes


def translate_dict(dict2print, include_curvature_plots = True, eps = 0):
    dictplots = {
    "plot1": {
        "yname_latex": "MSE",
        "data": {
            "MSE": dict2print["MSE"]
        }
    },
    "plot2": {
        "yname_latex": "$\mathcal{L}_\mathrm{unif}$",
        "data": {
            "$\widehat\mathcal{L}_\mathrm{unif}$": dict2print["uniform_loss"]
        }
    }
    }
    if include_curvature_plots == True:
        dictplots["plot3"] = {
            "yname_latex": "$\mathcal{L}_\mathrm{curv}$",
            "data": {
                "$\widehat\mathcal{L}_\mathrm{curv}$": dict2print["curvature_loss"]
            }
        }
        dictplots["plot4"] = {
            "yname_latex": "$R^2$",
            "data": {
                "mean": dict2print["curv_squared_mean"],
                "max": dict2print["curv_squared_max"]
            }
        }
    dict2 = {

    "plot5": {
        "yname_latex": "$\det(g)$",
        "data": {
            "mean": dict2print["g_det_mean"],
            "max": dict2print["g_det_max"],
            "min": dict2print["g_det_min"]
        }
    },

    "plot6": {
        "yname_latex": r"$\|g_{reg}^{-1}\|_F, \ \varepsilon = $" + f"{eps}",
        "data": {
            "mean": dict2print["g_inv_norm_mean"],
            "max": dict2print["g_inv_norm_max"]
        }
    },

    "plot7": {
        "yname_latex": r"$\|\nabla \Psi \|^2_F = \mathrm{tr} (g) $",
        "data": {
            "mean": dict2print["decoder_jac_norm_mean"],
            "max": dict2print["decoder_jac_norm_max"]
        }
    },

    "plot8": {
        "yname_latex": r"$\|\nabla \Phi \|^2_F $",
        "data": {
            "mean": dict2print["encoder_jac_norm_mean"],
            "max": dict2print["encoder_jac_norm_max"]
        }
    }
    }
    
    dictplots = {**dictplots,**dict2}
    return dictplots

def PlotSmartConvolve(dictplots, numwindows1 = 50, numwindows2 = 200):
    number_of_plots = len(dictplots)
    fig, axes = plt.subplots(nrows=number_of_plots, ncols=3, figsize=(4*3, number_of_plots*4))
    
    win = [signal.windows.hann(1), signal.windows.hann(numwindows1), signal.windows.hann(numwindows2)]  # convolution window size
    i = 0
    color_iterable = iter(mcolors.TABLEAU_COLORS)
    for plot_name,plot_info in dictplots.items():
        #if number_of_plots == 1:
        #    axes = [axes]
        
        for legend, curve in plot_info["data"].items(): 
            try:
                newcolor = next(color_iterable)
            except StopIteration:
                color_iterable = iter(mcolors.TABLEAU_COLORS)
                newcolor = next(color_iterable)
            #end except
                
            if legend=="max":
                linestyle = 'dashed'
            else:
                linestyle = 'solid'
            for j in range(3):
                axes[i,j].semilogy(signal.convolve(curve, win[j], mode='same') / sum(win[j]), color = newcolor,label = legend,ls = linestyle)
                #axes[i,j].semilogy(curve, color = newcolor,label = legend,ls = linestyle)
                axes[i,j].set_xlabel('Batches')
            #end for
        #end for
        axes[i,0].set_ylabel(plot_info["yname_latex"])
        axes[i,0].legend(loc="lower left")
        i += 1
    plt.show()
    return fig,axes