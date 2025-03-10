import torch, math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import signal

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

def plot_ae_outputs_selected(test_dataset,encoder,decoder,targets=None,selected_labels = None,D=784,dpi=400):
    if selected_labels is None:
        n = 10
    else:
        n = len(selected_labels)
    plt.figure(figsize=(6 * n / 10, 1.5), dpi=dpi)
    if targets == None:
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
    return ax

def discrete_cmap_upd(N, base_cmap=None):
    """Create an N-bin discrete colormap from the given base colormap."""
    base = plt.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap = matplotlib.colors.ListedColormap(color_list[:N], name=f"{base.name}_{N}")
    return cmap

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
    axes[1].set_ylabel('Equidistribution loss')
    
    axes[2].semilogy(curv_train_list, color = 'tab:blue')
    axes[2].set_ylabel('Curvature')
    for i in range(3):
        axes[i].set_xlabel('Batches')
    #fig.show()
    plt.show()
    return fig,axes

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


def translate_dict(dict_losses_to_plot, eps = 0):
    dictplots = {
    "plot1": {
        "yname_latex": "MSE",
        "data": {
            "MSE": dict_losses_to_plot["MSE"]
        }
    },
    "plot2": {
        "yname_latex": "$\mathcal{L}_\mathrm{equi}$",
        "data": {
            "$\widehat\mathcal{L}_\mathrm{equi}$": dict_losses_to_plot["Equidistribution"]
        }
    },
    "plot3": {
        "yname_latex": "$\mathcal{L}_\mathrm{contractive}$",
        "data": {"$\widehat\mathcal{L}_\mathrm{contractive}$": dict_losses_to_plot.get("Contractive")}
    },
    "plot4": {
        "yname_latex": "$\mathcal{L}_\mathrm{curv}$",
        "data": {"$\widehat\mathcal{L}_\mathrm{curv}$": dict_losses_to_plot.get("Curvature")}
    },
    "plot5": {
        "yname_latex": "$\det(g)$",
        "data": {k: dict_losses_to_plot.get(k) for k in ["g_det_mean", "g_det_max", "g_det_min"]}
    },
    "plot6": {
        "yname_latex": rf"$\|g_{{reg}}^{{-1}}\|_F, \ \varepsilon = {eps}$",
        "data": {k: dict_losses_to_plot.get(k) for k in ["g_inv_norm_mean", "g_inv_norm_max"]}
    },
    "plot7": {
        "yname_latex": r"$\|\nabla \Psi \|^2_F = \mathrm{tr} (g) $",
        "data": {k: dict_losses_to_plot.get(k) for k in ["decoder_jac_norm_mean", "decoder_jac_norm_max"]}
    },
    "plot8": {
        "yname_latex": r"$\|\nabla \Phi \|^2_F $",
        "data": {k: dict_losses_to_plot.get(k) for k in ["encoder_jac_norm_mean", "encoder_jac_norm_max"]}
    },
    "plot9": {
            "yname_latex": "$R^2$", 
            "data": {k: dict_losses_to_plot.get(k) for k in ["curv_squared_mean", "curv_squared_max"]}
    }
    }
    # Merge and filter out empty plots
    # Iterate over dict2 and collect the keys to remove
    keys_to_remove = []
    for key, values in dictplots.items():
        if None in values["data"].values():  # Check if there are any None values in the "data"
            keys_to_remove.append(key)  # Add the key to the list for later removal

    # Remove the keys after iteration
    for key in keys_to_remove:
        dictplots.pop(key, None)
    return dictplots


def PlotSmartConvolve(dictplots,test_dictplots = None,
        plot_test_losses = True,
        numwindows1 = 50, numwindows2 = 200):
    if plot_test_losses == True and test_dictplots == None:
        print("Set test losses dictionary to print!")
        return
    
    number_of_plots = len(dictplots)

    fig, axes = plt.subplots(nrows=number_of_plots, ncols=3, figsize=(4*3, number_of_plots*4))
    
    win = [signal.windows.hann(1), signal.windows.hann(numwindows1), signal.windows.hann(numwindows2)]  # convolution window size
    i = 0
    color_iterable = iter(mcolors.TABLEAU_COLORS)
    for plot_name, plot_info in dictplots.items():
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
            # end if
            for j in range(3):
                axes[i,j].semilogy(signal.convolve(curve, win[j], mode='valid') / sum(win[j]), 
                                   color = newcolor,
                                   label = "Train",
                                   ls = linestyle)
                axes[i,j].set_xlabel('Batches')
            # end for
        # end for
        axes[i,0].set_ylabel(plot_info["yname_latex"])
        #axes[i,0].legend(loc="lower left")
        i += 1
    # end for
    i=0
    if plot_test_losses == True:    
        for plot_name, plot_info in test_dictplots.items():
            for legend, test_curve in plot_info["data"].items(): 
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
                # end if
                pace_ratio = len(curve)/len(test_curve)
                test_curve_interpolated = np.interp(np.arange(len(curve)),
                                                    np.arange(len(test_curve))*pace_ratio,
                                                    test_curve) # to match train curve
                for j in range(3):
                    axes[i,j].semilogy(signal.convolve(test_curve_interpolated, win[j], mode='valid') / sum(win[j]), 
                                    color = newcolor,
                                    label = "Test",
                                    ls = linestyle)
            axes[i,0].legend(loc="lower left")
            i+=1
    plt.show()
    return fig,axes


# to be deprecated
def PlotSmartConvolve_old(dictplots, numwindows1 = 50, numwindows2 = 200):
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
                axes[i,j].semilogy(signal.convolve(curve, win[j], mode='valid') / sum(win[j]), color = newcolor,label = legend,ls = linestyle)
                #axes[i,j].semilogy(curve, color = newcolor,label = legend,ls = linestyle)
                axes[i,j].set_xlabel('Batches')
            #end for
        #end for
        axes[i,0].set_ylabel(plot_info["yname_latex"])
        axes[i,0].legend(loc="lower left")
        i += 1
    plt.show()
    return fig,axes

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def point_plot(encoder, data_loader, batch_idx, config,
        show_title=True, colormap='jet', normalize_to_unit_square = False, 
        s=40, draw_grid=False, figsize=(9, 9)):
    # Plotting the latent embedding of data taken from dataloader
    # using the encoder function in "encoder"
    # params of the dataset taken from YAML file "config"
    # Extract labels and data from the dataset
    #labels = data[:][1]
    #data = data[:][0]

    data_tensor_list = []
    labels_list =[]
    for batch_data_labels in data_loader:
        batch_data, batch_labels = batch_data_labels
        data_tensor_list.append(batch_data)
        labels_list.append(batch_labels)
    data_tensor = torch.cat(data_tensor_list)
    labels = torch.cat(labels_list)
    
    D = config["architecture"]["input_dim"]
    dataset_name = config["dataset"]["name"]

    # Perform encoding
    with torch.no_grad():
        data_tensor = data_tensor.view(-1, D)  # reshape the data (flatten)
        encoded_data = encoder(data_tensor).cpu()

    # Convert to numpy for plotting
    encoded_data_to_plot = encoded_data.numpy()
    labels = labels.numpy()
    selected_labels = np.unique(labels)

    #plt.rcParams.update({'font.size': 20})
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # checking if normalization of plotting scale is needed
    if normalize_to_unit_square == True:
        encoded_data = encoded_data/torch.pi
    else:
        ax.set_ylim(-math.pi, math.pi)
        ax.set_xlim(-math.pi, math.pi)
        ax.set_yticks([-3., -2., -1., 0., 1., 2., 3.])
        ax.set_xticks([-3., -2., -1., 0., 1., 2., 3.])
    
    # Create scatter plot
    if dataset_name == "Swissroll":
        sc = ax.scatter(encoded_data_to_plot[:, 0], encoded_data_to_plot[:, 1], s=s, c=labels, alpha=1.0, 
                        marker='o', edgecolor='none', cmap = colormap)
    elif dataset_name in ["MNIST01", "MNIST_subset"]:
        k = len(config["dataset"]["selected_labels"])
        norm = plt.Normalize(vmin=min(labels), vmax=max(labels))
        sc = ax.scatter(encoded_data_to_plot[:, 0], encoded_data_to_plot[:, 1], s=s, c=labels, alpha=1.0, 
                        marker='o', edgecolor='none', 
                        cmap=colormap)
                        #cmap = discrete_cmap_upd(k, colormap))
        # create discrete colorbar
        cmap = plt.get_cmap(colormap)
        discrete_cmap = mcolors.ListedColormap([cmap(norm(label)) for label in selected_labels])
        # Create a ScalarMappable for the discrete colorbar
        sm = plt.cm.ScalarMappable(cmap=discrete_cmap, norm=norm)
        sm.set_array([])  # No need for a full color scale
        tick_positions = np.linspace(selected_labels.min(),selected_labels.max(),k)# fix the positions of top and bottom ticks.
        cbar = fig.colorbar(sm, ax=ax, ticks=tick_positions, shrink=0.7, spacing = "uniform")
        cbar.set_label('Cluster Label')
        cbar.set_ticklabels(selected_labels)  # Show only unique labels

    # Add title if required
    if show_title:
        ax.set_title(f'Latent space for test data in AE at batch {batch_idx}')
    
    # Enable grid if required
    ax.grid(draw_grid)

    # Adjust layout to prevent elements from being cut off
    fig.tight_layout()

    # Return the figure object
    return fig

# this function will be deprecated
def point_plot_old(encoder, data: torch.utils.data.dataset.Subset, dataset_name, 
               batch_idx, config, device, show_title=True, colormap='jet', 
               s=40, draw_grid=False, figsize=(9, 9)):
    # Plotting the latent embedding of "data" using the encoder function in "encoder"
    # params of the dataset taken from YAML file "config"
    # Extract labels and data from the dataset
    #labels = data[:][1]
    #data = data[:][0]
    if dataset_name == "Synthetic":
        labels = data.dataset.tensors[1]
        data = data.dataset.tensors[0]
    elif dataset_name in ["MNIST01", "MNIST_subset"]:
        data = data.data.to(torch.float32)/255.
        labels = data.targets
    else:
        labels = data.targets
        data = data.data
    
    D = config["architecture"]["input_dim"]
    dataset_name = config["dataset"]["name"]

    # figure out the number of classes of data
    if config["dataset"]["name"] == "MNIST":
        k = 10
    elif config["dataset"]["name"] == "MNIST01":
        k = len(config["dataset"]["selected_labels"])
    elif config["dataset"]["name"] == "Synthetic":
        k = config["dataset"]["k"]
    
    # Perform encoding
    with torch.no_grad():
        data = data.view(-1, D)  # reshape the data (flatten)
        data = data.to(device)  # Uncomment this if needed to move data to GPU/CPU
        encoded_data = encoder(data)

    # Convert to numpy for plotting

    encoded_data2plot = ( encoded_data.cpu() / torch.pi ).numpy()
    labels = labels.numpy()

    #plt.rcParams.update({'font.size': 20})
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot
    if dataset_name == "Swissroll":
        sc = ax.scatter(encoded_data2plot[:, 0], encoded_data2plot[:, 1], s=s, c=labels, alpha=1.0, marker='o', edgecolor='none', cmap = colormap)
    else:
        sc = ax.scatter(encoded_data2plot[:, 0], encoded_data2plot[:, 1], s=s, c=labels, alpha=1.0, marker='o', edgecolor='none', cmap = discrete_cmap(k, colormap))
            # a smart way to place ticks nicely on a colorbar
        from matplotlib.colors import  BoundaryNorm
        bounds = np.linspace(-0.5, k - 0.5, k + 1)
        norm = BoundaryNorm(bounds, discrete_cmap(k, colormap).N)
        cbar = plt.colorbar(sc, boundaries=bounds, norm=norm, ticks=np.arange(k))

    # Add title if required
    if show_title:
        ax.set_title(f'Latent space for test data in AE at batch {batch_idx}')
    
    # Enable grid if required
    ax.grid(draw_grid)

    # Adjust layout to prevent elements from being cut off
    fig.tight_layout()

    # Return the figure object
    return fig

# this function is used for plotting while training
def point_plot_fast(encoded_points,labels, 
               batch_idx, config, show_title=True, colormap='jet', 
               s=40, draw_grid=False, figsize=(9, 9)):
    # Plotting the latent embedding of "data" using the encoder function in "encoder"
    # params of the dataset taken from YAML file "config"
    dataset_name = config["dataset"]["name"]

    #plt.rcParams.update({'font.size': 20})
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)    
    # Create scatter plot
    if dataset_name == "Swissroll":
        sc = ax.scatter(encoded_points[:, 0], encoded_points[:, 1], s=s, c=labels, alpha=1.0, marker='o', 
                edgecolor='none', cmap = colormap)
    elif dataset_name in["MNIST_subset", "MNIST01","MNIST","Synthetic"]:
        selected_labels = torch.tensor(config["dataset"]["selected_labels"])
        if config["dataset"]["name"] == "MNIST":
            selected_labels = torch.arange(10)
        elif config["dataset"]["name"] == "Synthetic":
            selected_labels = torch.arange(config["dataset"]["k"])
        k = len(selected_labels)
        mask = torch.isin(labels, torch.tensor(selected_labels))
        # Apply the mask to filter data and labels
        filtered_data = encoded_points[mask]
        filtered_labels = labels[mask]
        sc = ax.scatter(filtered_data[:, 0], filtered_data[:, 1], s=s, c=filtered_labels, 
                        alpha=1.0, marker='o', 
                edgecolor='none', cmap = colormap)
        
        cmap = plt.get_cmap(colormap)
        norm = plt.Normalize(vmin=torch.min(selected_labels).item(),
                              vmax=torch.max(selected_labels).item())
        discrete_cmap = mcolors.ListedColormap([cmap(norm(label)) for label in selected_labels.tolist()])
        # Create a ScalarMappable for the discrete colorbar
        sm = plt.cm.ScalarMappable(cmap=discrete_cmap, norm=norm)
        sm.set_array([])  # No need for a full color scale
        tick_positions = np.linspace(selected_labels.min(),selected_labels.max(),k)# fix the positions of top and bottom ticks.
        cbar = fig.colorbar(sm, ax=ax, ticks=tick_positions, shrink=0.7, spacing = "uniform")
        cbar.set_label('Cluster Label')
        cbar.set_ticklabels(selected_labels.tolist())  # Show only unique labels
    # Add title if required
    if show_title:
        ax.set_title(f'Latent space for test data in AE at batch {batch_idx}')
    
    # Enable grid if required
    ax.grid(draw_grid)

    # Adjust layout to prevent elements from being cut off
    fig.tight_layout()

    # Return the figure object
    return fig