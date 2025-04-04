import torch
import numpy as np
import matplotlib.pyplot as plt
import ricci_regularization, time, os
from tqdm import tqdm

def initialize_centers(encoded_points, num_clusters, num_data_points):
    """
    Initializes the cluster centers and geodesic connections using PyTorch.
    
    Parameters:
        encoded_points (torch.Tensor): Latent encoding of data points, shape (n_samples, n_features).
        num_clusters (int): Number of clusters.
    
    Returns:
        centers (torch.Tensor): Initialized cluster centers, shape (num_clusters, n_features).
        probabilities (torch.Tensor): Sample probabilities, shape (num_clusters, n_samples).
    """
    # Sample num_clusters times Randomly num_data_points probabilities that sum to 1 for each
    # random initialization of a cluster center
    probabilities = torch.distributions.Dirichlet(torch.ones(num_data_points)).sample((num_clusters,))
    centers = probabilities @ encoded_points.reshape(num_data_points,-1) 
    #print(f"Initialized centers:\num_data_points {centers}")
    return centers

def construct_interpolation_points_on_segments_connecting_centers2encoded_data(starting_points, final_points, num_aux_points, cut_off_ends = True):
    """
    Connect every point in `starting_points` to every point in `final_points` with intermediate points.

    Args:
        starting_points (torch.Tensor): Tensor of shape (num_data_points, latent_dim) representing points.
        final_points (torch.Tensor): Tensor of shape (num_clusters, latent_dim) representing centroid points.
        num_aux_points (int): Number of intermediate points (including endpoints) per segment.

    Returns:
        torch.Tensor: Tensor of shape (num_data_points * num_clusters * num_aux_points, latent_dim) containing all intermediate points.
    """
    # Check that the final dimensions of inputs match
    if starting_points.shape[-1] != final_points.shape[-1] or final_points.shape[-1] != starting_points.shape[-1]:
        raise ValueError(
            f"Mismatch in dimensions: 'starting_points' and 'final_points' must have the same final dimension. "
            f"Got starting_points with shape {starting_points.shape}, final_points with shape {final_points.shape}. "
        )

    # Generate interpolation parameters (num_aux_points values between 0 and 1)
    t = torch.linspace(0, 1, steps=num_aux_points).to(starting_points.device).view(1, 1, num_aux_points, 1)  # Shape: (1, 1, num_aux_points, 1)

    # Reshape starting_points and final_points for broadcasting
    starting_points_expanded = starting_points.unsqueeze(1).unsqueeze(2)  # Shape: (num_starting_points, 1, 1, points_dim)
    final_points_expanded = final_points.unsqueeze(0).unsqueeze(2)        # Shape: (1, num_final_points, 1, points_dim)

    # Compute all intermediate points using linear interpolation
    geodesic_curve = starting_points_expanded + t * (final_points_expanded - starting_points_expanded)  # Shape: (num_data_points, num_clusters, num_aux_points, latent_dim)

    if cut_off_ends == True:
        # Select interpolation_points cutting of the starting and the final point for every segment
        interpolation_points = geodesic_curve[:,:,1:-1,:]
    else:
        interpolation_points = geodesic_curve
    return interpolation_points

def geodesics_from_parameters_interpolation_points(parameters_of_geodesics, end_points):
    """
    Constructs geodesics from parameters of the geodesics and end points.
    This function simply adds end points to intermediate points.

    Parameters:
    - parameters_of_geodesics (torch.Tensor): Interpolation parameters with shape 
      (num_starting_points, num_clusters, num_interpolation_points, latent_dim).
    - end_points (list of torch.Tensor): [starting_points, final_points], where:
      - starting_points: Shape (num_starting_points, latent_dim).
      - final_points: Shape (num_clusters, latent_dim).

    Returns:
    - torch.Tensor: Complete geodesics with shape 
      (num_starting_points, num_clusters, num_interpolation_points + 2, latent_dim).
    """
    # reading the shapes of the parameters
    num_starting_points, num_clusters, num_interpolation_points, latent_dim = parameters_of_geodesics.shape
    starting_points, final_points = end_points
    # starting_points are usually encoded data
    # final_points are usually cluster centroids  

    #expand starting_points
    starting_points_expanded = starting_points.unsqueeze(1).unsqueeze(2) # Shape: (num_starting_points, 1, 1, latent_dim)
    starting_points_expanded = starting_points_expanded.expand(num_starting_points, num_clusters , 1, latent_dim)
    #expand final_points
    final_points_expanded = final_points.unsqueeze(0).unsqueeze(2)  # Shape: (1, num_clusters, 1, latent_dim)
    final_points_expanded = final_points_expanded.expand(num_starting_points, num_clusters , 1, latent_dim)
    # concatenate the starting points, the interpolation_points and final_points  along the dimention associated interpolation_points
    geodesic_curve = torch.cat((starting_points_expanded, parameters_of_geodesics, final_points_expanded),dim=2) 
    return geodesic_curve

def construct_shortest_linear_segments_connecting(starting_points, final_points, num_aux_points):
    # Check that the final dimensions of inputs match
    if starting_points.shape[-1] != final_points.shape[-1] or final_points.shape[-1] != starting_points.shape[-1]:
        raise ValueError(
            f"Mismatch in dimensions: 'starting_points' and 'final_points' must have the same final dimension. "
            f"Got starting_points with shape {starting_points.shape}, final_points with shape {final_points.shape}. "
        )

    # Generate interpolation parameters (num_aux_points values between 0 and 1)
    t = torch.linspace(0, 1, steps=num_aux_points).to(starting_points.device).view(1, 1, num_aux_points, 1)  # Shape: (1, 1, num_aux_points, 1)

    # Reshape starting_points and final_points for broadcasting
    starting_points_expanded = starting_points.unsqueeze(1).unsqueeze(2)  # Shape: (num_starting_points, 1, 1, points_dim)
    final_points_expanded = final_points.unsqueeze(0).unsqueeze(2)        # Shape: (1, num_final_points, 1, points_dim)

    # Compute all intermediate points using linear interpolation and recenter all geodesics
    starting_points_recentered = torch.remainder((starting_points_expanded - final_points_expanded + torch.pi), 2*torch.pi ) - torch.pi
    #final_points_recentered = torch.zeros_like(final_points_expanded)

    geodesic_curve_recentered = (1 - t) * starting_points_recentered + final_points_expanded 
    return geodesic_curve_recentered

def geodesics_from_parameters_schauder(geodesic_solver, parameters_of_geodesics, end_points, periodicity_mode = True):
    """
    Constructs geodesic curves using Schauder basis representation.

    Parameters:
    geodesic_solver: NumericalGeodesics
        A solver used to compute geodesics numerically.
    parameters_of_geodesics: torch.Tensor
        Coefficients representing geodesic curves in the Schauder basis.
        Shape: (num_starting_points, num_final_points, num_basis_functions, dim).
    end_points: tuple of [starting_points, final_points]
        the tuple contains:
        - starting_points (torch.Tensor): Initial points of geodesics, shape (num_starting_points, dim).
        - final_points (torch.Tensor): Final points of geodesics, shape (num_final_points, dim).

    Returns:
    geodesic_curve: torch.Tensor
        The computed geodesic curves, shape (num_starting_points, num_final_points, step_count, dim).
    """
    parameters = parameters_of_geodesics
    starting_points, final_points = end_points
    
    basis = geodesic_solver.schauder_bases["zero_boundary"]["basis"]
    step_count = basis.shape[0]

    if periodicity_mode == True:
        linear_curve = construct_shortest_linear_segments_connecting(
            starting_points=starting_points, 
            final_points=final_points,
            num_aux_points=step_count
        )
    else:
        linear_curve = construct_interpolation_points_on_segments_connecting_centers2encoded_data(
            starting_points=starting_points, 
            final_points=final_points,
            num_aux_points=step_count, 
            cut_off_ends=False)
    
    geodesic_curve = linear_curve + torch.einsum("sn,bend->besd", basis, parameters)
    return geodesic_curve

def compute_energy(mode, parameters_of_geodesics, end_points, decoder, geodesic_solver=None, 
                   reduction = "mean", device = "cuda", periodicity_mode = True):
    """
    Computes the energy of geodesic curves using finite differences.

    Parameters:
    mode: str
        Determines the method used to compute geodesics. Options:
        - "Interpolation_points": Uses interpolation-based geodesics.
        - "Schauder": Uses Schauder basis representation for geodesics.
    parameters_of_geodesics: torch.Tensor
        Coefficients or parameters defining the geodesic curves.
        - Shape depends on the chosen mode.
    end_points: tuple
        A tuple containing:
        - starting_points (torch.Tensor): Initial points of geodesics.
        - final_points (torch.Tensor): Final points of geodesics.
    decoder: callable, optional (default=torus_ae.decoder_torus)
        A function that decodes geodesic curves from their latent representation.
    geodesic_solver: NumericalGeodesics
        A solver used to compute geodesics numerically.
    reduction: 'sum', 'none'

    Returns:
    energy: torch.Tensor (scalar)
        The total energy of the geodesic curves, computed as the sum of squared differences
        between consecutive points in the decoded geodesic curves.
    """
    if mode == "Interpolation_points":
        geodesic_curve = geodesics_from_parameters_interpolation_points(parameters_of_geodesics, end_points)
    elif mode == "Schauder":
        geodesic_curve = geodesics_from_parameters_schauder(geodesic_solver, parameters_of_geodesics, end_points, 
                                                            periodicity_mode=periodicity_mode)
    geodesic_curve = geodesic_curve.to(device) # shape (N,K,step_count,d)
    # define step count (number of interpolation points on the geodesic)
    step_count = geodesic_curve.shape[-2]
    # decode the geodesics
    decoded_geodesic_curve = decoder(geodesic_curve)
    # Compute energy (finite differences)
    tangent_vectors = decoded_geodesic_curve[:,:,1:,:] - decoded_geodesic_curve[:,:,:-1,:] # shape (N,K,step_count-1,D)
    if reduction == "none":
        energy = (tangent_vectors**2).sum(dim=(-2,-1)) # comute Euclidean energies of the curves in R^D
    if reduction == "sum":
        energy = (tangent_vectors**2).sum()
    if reduction == "mean":
        energy = (tangent_vectors**2).sum(dim=(-2,-1)).mean() # 
    energy *= (step_count - 1) # correct normalization of finite differences energy computation
    # Warning! by default the outpiut is the single scalar, i.e the mean of all the energies among N*K geodesics
    return energy

def compute_lengths(mode, parameters_of_geodesics, end_points, decoder, geodesic_solver=None, 
                    reduction="none", device = "cuda", periodicity_mode = True, return_geodesic_curve = False):
    """
    Computes the lengths of geodesic curves in Euclidean space.

    Parameters:
    mode: str
        Determines the method used to compute geodesics. Options:
        - "Interpolation_points": Uses interpolation-based geodesics.
        - "Schauder": Uses Schauder basis representation for geodesics.
    parameters_of_geodesics: torch.Tensor
        Coefficients or parameters defining the geodesic curves.
        - Shape depends on the chosen mode.
    end_points: tuple
        A tuple containing:
        - starting_points (torch.Tensor): Initial points of geodesics.
        - final_points (torch.Tensor): Final points of geodesics.
    decoder: callable, optional (default=torus_ae.decoder_torus)
        A function that decodes geodesic curves from their latent representation.
    geodesic_solver: NumericalGeodesics
        A solver used to compute geodesics numerically.

    Returns:
    computed_lengths: torch.Tensor
        A tensor containing the computed Euclidean lengths of all geodesic curves.
        - Shape: (batch_size, step_count - 1).
    """
    if mode == "Interpolation_points":
        geodesic_curve = geodesics_from_parameters_interpolation_points(parameters_of_geodesics, end_points)
    elif mode == "Schauder":
        geodesic_curve = geodesics_from_parameters_schauder(geodesic_solver, parameters_of_geodesics, end_points, periodicity_mode=periodicity_mode)
    geodesic_curve = geodesic_curve.to(device)
    decoded_geodesic_curve = decoder(geodesic_curve)
    tangent_vectors = decoded_geodesic_curve[:,:,1:,:] - decoded_geodesic_curve[:,:,:-1,:]
    if reduction == "none":
        lengths = (tangent_vectors).norm(dim=(-1)).sum(dim=(-1)) # first find all norms of small tangent vectors in the discretization then sum them for each geodesic
    if reduction == "sum":
        lengths = (tangent_vectors).norm(dim=(-1)).sum()
    if reduction == "old":    
        lengths = torch.sqrt((tangent_vectors**2).sum(dim=(-2,-1))) # seems to be a wrong formula
    # Warning! by default the outpiut is the vector of length of all geodesics 
    if return_geodesic_curve == True:
        return geodesic_curve, lengths
    else:
        return lengths

# ---------------------------------------
#plotting
def construct_mapping_np(memberships_np, ground_truth_labels):
    unique_labels = np.unique(memberships_np)
    mapping = {}

    for label in unique_labels:
        indices = np.where(memberships_np == label)[0]
        gt_subset = ground_truth_labels[indices]
        unique_gt, counts = np.unique(gt_subset, return_counts=True)
        most_common_gt = unique_gt[np.argmax(counts)]
        mapping[label] = most_common_gt
    return mapping

def apply_mapping_np(memberships_np, mapping):
    # Replace each color label with its most frequent ground truth match
    mapped_array = np.array([mapping[label] for label in memberships_np])
    return mapped_array
def plot_octopus(geodesic_curve, periodicity_mode=True, memberships = None, ground_truth_labels = None, 
                 saving_folder = None, suffix = None, verbose = True,
                 xlim = torch.pi, ylim = torch.pi, 
                 show_geodesics_in_original_local_charts = False,
                 show_only_geodesics_to_nearest_centroids = False,
                 size_of_points = 1):
    """
    Plots geodesics, centers, and datapoints on a 2D plane with options to visualize memberships 
    and meaningful geodesics, and save the resulting plot.

    Parameters:
    - geodesic_curve (torch.Tensor): Tensor of shape (num_datapoints, num_clusters, num_aux_points_on_geodesics, latent_dimension) 
      representing the geodesic segments.
    - memberships (torch.Tensor, optional): Tensor of shape (num_datapoints,) representing class memberships for each datapoint.
    - saving_folder (str, optional): Folder path to save the plots.
    - suffix (int, optional): Frame index to append to the saved filename.
    - verbose (bool, optional): Whether to display the plot. If False, the plot will be closed after saving.
    - xlim (float or None, optional): Limits for the x-axis. If None, no limit is applied.
    - ylim (float or None, optional): Limits for the y-axis. If None, no limit is applied.

    Returns:
    - None
    """
    def move_to_main_local_chart(tensor):
        return torch.remainder(tensor + torch.pi, 2*torch.pi) - torch.pi
    
    if show_geodesics_in_original_local_charts == False:
        geodesic_curve = move_to_main_local_chart(geodesic_curve)
    # Convert to numpy
    if memberships != None:
        memberships_np = np.array(memberships)

    # uniformizing coloring if the ground truth colors are nown
    if ground_truth_labels!= None:
        mapping = construct_mapping_np(memberships_np, np.array(ground_truth_labels))
        memberships_np = apply_mapping_np(memberships_np,mapping)

    # Detach geodesic_curve from computation graph to avoid accidental gradients.
    geodesic_curve = geodesic_curve.detach() #shape (num_datapoints, num_clusters, num_aux_points_on_geodesics,latent_dimension)
    N = geodesic_curve.shape[0] # num data points
    K = geodesic_curve.shape[1] # num clusters
    if xlim != None:
        plt.xlim(-xlim, xlim)
    if ylim != None:
        plt.ylim(-xlim, xlim)
    # plot allgeodesics in orange if needed
    if show_only_geodesics_to_nearest_centroids == False:
        if periodicity_mode == True:
            plt.scatter(geodesic_curve[:,:,:,0], geodesic_curve[:,:,:,1],marker='o',
                    s=1, c = "orange",zorder=5)
        else:
            for i in range(N):
                for j in range(K):
                    plt.plot(geodesic_curve[i,j,:,0], geodesic_curve[i,j,:,1],'-',marker='o', c = "orange", markersize=3)
        # end if
    # end if

    # plot centers
    centers = geodesic_curve[0,:,-1,:]
    # plot the datapoints (the starting points on all the geodesics, colored by memberships if specified):
    if memberships!= None:
        num_classes = int(memberships_np.max().item()) + 1
        colormap = ricci_regularization.discrete_cmap(num_classes, 'jet')
        # plot cluster centers
        plt.scatter(centers[:,0], centers[:,1], c=apply_mapping_np(np.arange(K),mapping), 
                marker='*', edgecolor='black',  
                cmap=colormap, s = 170,zorder = 12)
        
        # plot initial points of the geodesics (i.e. encoded data points) colored by assigned labels
        plt.scatter(geodesic_curve[:,0,0,0],
                        geodesic_curve[:,0,0,1], 
                        c=memberships_np, marker='o', edgecolor='none', 
                        cmap=colormap, s = 30,zorder = 10)
        # recompute the geodesics to nearest centroids
        batch_indices = torch.arange(N) # this is needed, since   geodesic_curve[:, cluster_index_of_each_point, :, :] will produce a tensor of shape (N,N,step_count,d)
        # pick only geodesics connecting points to cluster relevant centroids where the points are assigned
        geodesics2nearestcentroids = geodesic_curve[batch_indices, memberships, :, :].detach() # shape (N,step_count,d)
        step_count = geodesics2nearestcentroids.shape[1]
        # color the geodesics to nearest centroids
        if periodicity_mode == True:
            plt.scatter(geodesics2nearestcentroids[:,:,0], 
                        geodesics2nearestcentroids[:,:,1], 
                        c=memberships_np.repeat(step_count,axis=0).T, cmap=colormap, s=size_of_points, zorder = 10)
        else:
            for i in range(N):
                colors = colormap(memberships_np) # this gives wrong colors
                plt.plot(geodesics2nearestcentroids[i,:,0], geodesics2nearestcentroids[i,:,1], 
                         c=colors[i], zorder = 10)
    else:
        plt.scatter(centers[:,0], centers[:,1], c="red", label = "centers", marker='*', edgecolor='black', s = 170,zorder = 10)
        plt.scatter(geodesic_curve[:,0,0,0], geodesic_curve[:,0,0,1], c="green", label = "centers", marker='o', s = 30,zorder = 10)
    # save images
    if saving_folder != None:
        if suffix != None:
            saving_path = saving_folder + f"/frame_{suffix:04d}.png"
            plt.savefig(saving_path)
            print(f"Octopus saved at {saving_path}")
    if verbose == False:
        plt.close()
    plt.show()
    return  

def manifold_plot_selected_labels(encoded_points2plot, encoded_points_labels, 
                                  selected_labels, plot_title='', verbose=True, 
                                  save_plot=True, saving_folder=None,
                                  file_saving_name=None):
    # Manifold plot for all points with selected labels
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(encoded_points2plot[:, 0], encoded_points2plot[:, 1], 
                        c=encoded_points_labels, cmap='jet', alpha=0.7)
    # Set plot limits to [-pi, pi] for both dimensions
    plt.xlim(-torch.pi, torch.pi)
    plt.ylim(-torch.pi, torch.pi)
    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', 
                markerfacecolor=scatter.cmap(scatter.norm(label)), markersize=8) 
            for label in selected_labels]
    plt.legend(handles, selected_labels, title="Ground Truth Labels",loc="upper left")
    plt.title(plot_title)
    if save_plot==True:
        plt.savefig(f"{saving_folder}/{file_saving_name}.pdf",bbox_inches='tight', format="pdf")
        print(f"manifold plot saved as{saving_folder}/{file_saving_name}.pdf")
    if verbose==True:
        plt.show()
    else:
        plt.close()

# Decision boundary plot
from scipy.interpolate import griddata
def plot_knn_decision_boundary(encoded_points, labels_for_coloring,
        ground_truth_labels = None, selected_labels = None, 
        grid_resolution=100, contour_levels_count=10, neighbours_number=7, 
        distance_computation_mode = "torus",interpolation_method='nearest',
        background_opacity = 0.5, points_size = 50,
        verbose=True, plot_title=None,save_plot=True, 
        saving_folder=None, file_saving_name=None, 
        cmap_points='coolwarm', cmap_background='coolwarm'):
    """
    Plots a smoother decision boundary of a k-Nearest Neighbors (k-NN) classifier in a 2D latent space.
    
    This version uses `scipy.interpolate.griddata` to create a smoothly interpolated background.
    """
    # Convert tensors to numpy
    encoded_points_np = encoded_points.numpy()
    labels_for_coloring_np = labels_for_coloring.numpy()

    # uniformizing coloring if the ground truth colors are nown
    def construct_mapping_np(labels_for_coloring_np, ground_truth_labels):
        unique_labels = np.unique(labels_for_coloring_np)
        mapping = {}

        for label in unique_labels:
            indices = np.where(labels_for_coloring_np == label)[0]
            gt_subset = ground_truth_labels[indices]
            unique_gt, counts = np.unique(gt_subset, return_counts=True)
            most_common_gt = unique_gt[np.argmax(counts)]
            mapping[label] = most_common_gt
        return mapping

    def apply_mapping_np(labels_for_coloring_np, mapping):
        # Replace each color label with its most frequent ground truth match
        mapped_array = np.array([mapping[label] for label in labels_for_coloring_np])
        return mapped_array
    if ground_truth_labels!= None:
        mapping = construct_mapping_np(labels_for_coloring_np, np.array(ground_truth_labels))
        labels_for_coloring_np = apply_mapping_np(labels_for_coloring_np,mapping)

    # the old way:
    """
    if selected_labels == None:
        # mapping all labels to 0,1,2...,k-1 with preserving the order
        _, labels_for_coloring_np = np.unique(labels_for_coloring_np, return_inverse=True)
    else:
        # Original tensor with values 0, 1, 2
        _, labels_for_coloring_np = np.unique(labels_for_coloring_np, return_inverse=True)
        # Mapping: 0 -> 1, 1 -> 5, 2 -> 8
        mapping = torch.tensor(selected_labels)
        # Apply mapping
        labels_for_coloring_np = mapping[labels_for_coloring_np]
    """
    # Create a dense uniform grid in [-pi, pi]^2
    x_vals = np.linspace(-np.pi, np.pi, grid_resolution)
    y_vals = np.linspace(-np.pi, np.pi, grid_resolution)
    grid_x, grid_y = np.meshgrid(x_vals, y_vals)
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    
    # 2. Compute pairwise Euclidean distances between grid points and encoded data
    # if distance_mode == "torus"
    # (||x - y||^2 = (x1 - y1)^2 + (x2 - y2)^2)
    # or using torus periodicity
    if distance_computation_mode =="plane":
        dists = torch.cdist(torch.from_numpy(grid_points), encoded_points)  # Shape: (grid_size^2, N_encoded)
    elif distance_computation_mode == "torus":
        coordinate_wise_distances = torch.abs(torch.from_numpy(grid_points)[:,None,:] - encoded_points_np[None,:,:]) #shape (grid_size^2, N_encoded, dim)
        coordinate_wise_distances_torus = torch.min(coordinate_wise_distances, 2*torch.pi - coordinate_wise_distances)
        dists = coordinate_wise_distances_torus.norm(dim = 2) # Shape: (grid_size^2, N_encoded)
    dists = dists.numpy()
    # Find the indices of the k=neighbours_number nearest neighbors
    nn_indices = np.argsort(dists, axis=1)[:, :neighbours_number]
    
    # Assign cluster labels by majority vote
    nn_labels = labels_for_coloring_np[nn_indices]
    grid_labels = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=nn_labels)
    
    # Smooth interpolation using griddata
    grid_labels_smooth = griddata(grid_points, grid_labels, (grid_x, grid_y), method=interpolation_method)
    
    # Plot results
    plt.figure(figsize=(8, 6))
    plt.contourf(grid_x, grid_y, grid_labels_smooth, levels=contour_levels_count, cmap=cmap_background, alpha=background_opacity)
    
    # Overlay original encoded points
    plt.scatter(encoded_points_np[:, 0], encoded_points_np[:, 1], 
                c=labels_for_coloring_np, cmap=cmap_points, edgecolors='k', 
                s=points_size, alpha=1.0)
    
    # Set plot limits and labels
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-np.pi, np.pi)
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    if plot_title==None:
        plt.title(f"{file_saving_name.replace('_',' ')} via {neighbours_number}-NN")
    else:
        plt.title(plot_title)
    if save_plot==True:
        plt.savefig(f"{saving_folder}/{file_saving_name}.pdf", bbox_inches='tight', format="pdf")
        print(f"Decision boundary plot saved as {saving_folder}/{file_saving_name}.pdf")
    
    if verbose  == True:
        plt.show()
    else:
        plt.close()
    return

#older version without interpolation. to be deprecated
def plot_knn_decision_boundary_nonsmooth(encoded_points, labels_for_coloring, 
        grid_resolution = 100, 
        neighbours_number = 7, distance_computation_mode = "torus",
        verbose=True, plot_title=None, 
        save_plot=True, saving_folder=None, 
        file_saving_name=None, cmap_points='jet',
        cmap_background='coolwarm'):
    """
    Plots the decision boundary of a k-Nearest Neighbors (k-NN) classifier in a 2D latent space.

    Parameters:
    - encoded_points: Tensor of shape (N, 2), representing data points in a 2D latent space.
    - labels_for_coloring: Tensor of shape (N,), representing class labels of encoded points.
    - grid_resolution: Number of points per axis to create a uniform grid.
    - neighbours_number: Number of neighbors to consider in k-NN classification.
    - verbose: If True, displays the plot.
    - save_plot: If True, saves the plot as a PDF file.
    - saving_folder: Directory where the plot should be saved (if save_plot is True).
    - file_saving_name: Name of the saved plot file.
    - cmap_points: Colormap for original data points.
    - cmap_background: Colormap for decision boundary regions.

    Steps:
    1. Generate a uniform grid of points covering the range [-pi, pi]².
    2. Compute the Euclidean distances between grid points and encoded data points.
    3. Determine the k nearest neighbors for each grid point.
    4. Assign cluster labels to grid points based on majority vote from neighbors.
    5. Plot the decision boundary along with the original data points.
    """
    # 1. Create a dense uniform grid in [-pi, pi]^2
    x_vals = torch.linspace(-torch.pi, torch.pi, grid_resolution)
    y_vals = torch.linspace(-torch.pi, torch.pi, grid_resolution)
    grid_x, grid_y = torch.meshgrid(x_vals, y_vals, indexing="ij")
    grid_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # Shape: (grid_size^2, 2)

    # 2. Compute pairwise Euclidean distances between grid points and encoded data
    # if distance_mode == "torus"
    # (||x - y||^2 = (x1 - y1)^2 + (x2 - y2)^2)
    # or using torus periodicity
    if distance_computation_mode =="plane":
        dists = torch.cdist(grid_points, encoded_points)  # Shape: (grid_size^2, N_encoded)
    elif distance_computation_mode == "torus":
        coordinate_wise_distances = torch.abs(grid_points[:,None,:] - encoded_points[None,:,:]) #shape (grid_size^2, N_encoded, dim)
        coordinate_wise_distances_torus = torch.min(coordinate_wise_distances, 2*torch.pi - coordinate_wise_distances)
        dists = coordinate_wise_distances_torus.norm(dim = 2) # Shape: (grid_size^2, N_encoded)
    # 3. Find the indices of the k=neighbours_number nearest neighbors
    _, nn_indices = torch.topk(dists, k=neighbours_number, largest=False, dim=1)  # Get indices of k=neighbours_number nearest neighbors

    # 4. Assign cluster labels by majority vote
    nn_labels = labels_for_coloring[nn_indices]  # Retrieve labels of nearest neighbors
    grid_labels, _ = torch.mode(nn_labels, dim=1)  # Majority vote

    # 5. Plot results
    plt.figure(figsize=(8, 6))

    # Plot the grid colored by assigned cluster
    plt.scatter(grid_points[:, 0], grid_points[:, 1], c=grid_labels, cmap=cmap_background, alpha=0.3, marker='s', s=10)

    # Overlay original encoded points
    plt.scatter(encoded_points[:, 0], encoded_points[:, 1], c=labels_for_coloring, cmap=cmap_points, edgecolors='k', s=40)

    # Set plot limits and labels
    plt.xlim(-torch.pi, torch.pi)
    plt.ylim(-torch.pi, torch.pi)
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    if plot_title==None:
        plt.title(f"{file_saving_name.replace('_',' ')} via {neighbours_number}-NN")
    else:
        plt.title(plot_title)
    if save_plot==True:
        plt.savefig(f"{saving_folder}/{file_saving_name}.pdf",bbox_inches='tight', format="pdf")
        print(f"Decision boundary plot saved as{saving_folder}/{file_saving_name}.pdf")
    if verbose==True:
        plt.show()



#-----------------
# Clustering evaluation F-measure
# Function to compute the set of pairs in the same cluster
def get_pairs(tensor):
    """
    Computes the set of index pairs where elements in the tensor have the same value.
    
    Parameters:
    tensor (torch.Tensor): A 1D tensor containing elements for which index pairs need to be found.
    
    Returns:
    set: A set of tuples representing index pairs (i, j) where tensor[i] == tensor[j].
    """
    index_map = {}
    
    # Store indices for each unique value
    for idx, val in enumerate(tensor.tolist()):
        if val not in index_map:
            index_map[val] = []
        index_map[val].append(idx)
    
    # Generate unique index pairs
    pairs = set()
    for indices in index_map.values():
        if len(indices) > 1:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    pairs.add((indices[i], indices[j]))
    
    return pairs

def compute_f_measure(P_labels, Q_labels):
    """
    Computes the F-measure between two clusterings P and Q.
    
    Parameters:
        P_labels (list or np.array or torch.tensor): Cluster labels for partition P.
        Q_labels (list or np.array or torch.tensor): Cluster labels for partition Q.
    
    Returns:
        float: The F-measure score in the range [0,1].
    """
    n = len(P_labels)
    assert len(Q_labels) == n, "Partitions must have the same number of samples."
    
    P_pairs = get_pairs(P_labels)
    Q_pairs = get_pairs(Q_labels)
    
    a = len(P_pairs & Q_pairs)  # Intersection
    b = len(Q_pairs - P_pairs)  # Pairs in Q but not in P
    c = len(P_pairs - Q_pairs)  # Pairs in P but not in Q
    
    # Compute F-measure
    if 2 * a + b + c == 0:
        return 1.0  # If both partitions have no pairs, they are identical
    
    F_measure = (2 * a) / (2 * a + b + c)
    return F_measure

#-----------------------------------------------------
# loading data for clustering
def get_validation_dataset(yaml_config):
    # Load data loaders based on YAML configuration
    dict = ricci_regularization.DataLoaders.get_dataloaders(
        dataset_config=yaml_config["dataset"],
        data_loader_config=yaml_config["data_loader_settings"],
        dtype=torch.float32
    )
    print("Experiment results loaded successfully.")
    # Loading data
    validation_dataset = dict.get("test_dataset")  # Assuming 'test_dataset' is a key returned by get_dataloaders

    # Loading the pre-tained AE
    torus_ae, Path_ae_weights = ricci_regularization.DataLoaders.get_tuned_nn(config=yaml_config)
    torus_ae.cpu()
    torus_ae.eval()
    return torus_ae, validation_dataset
def load_points_for_clustering(validation_dataset, random_seed_picking_points, yaml_config, torus_ae, 
                               Path_clustering_setup, verbose = False, N=300):
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=100)
    D = yaml_config["architecture"]["input_dim"]
    list_encoded_data_filtered = []
    list_labels_filtered = []
    torus_ae.cpu()
    for data, label in validation_loader: # in order to respect blind test philosophy 
        mask_batch = torch.isin(label, torch.tensor(yaml_config["dataset"]["selected_labels"])) # mask will be used to chose only labels in selected_labels
        data_filtered = data[mask_batch]
        labels_filtered = label[mask_batch]
        enc_images = torus_ae.encoder_to_lifting(data_filtered.reshape(-1, D)).detach()
        list_encoded_data_filtered.append(enc_images)
        list_labels_filtered.append(labels_filtered)
    all_encoded_data_filtered = torch.cat(list_encoded_data_filtered)
    all_labels_filtered = torch.cat(list_labels_filtered)
    # balanced clusters
    encoded_points = []
    encoded_labels = []
    # Sample 100 points from each label
    torch.manual_seed(random_seed_picking_points) # reproducibility
    for label in yaml_config["dataset"]["selected_labels"]:
        indices = torch.where(all_labels_filtered == label)[0]
        sampled_indices = indices[torch.randperm(len(indices))[:min(N//3, len(indices))]]
        encoded_points.append(all_encoded_data_filtered[sampled_indices])
        encoded_labels.append(all_labels_filtered[sampled_indices])
    # Convert to tensors
    encoded_points = torch.cat(encoded_points, dim=0)
    ground_truth_labels = torch.cat(encoded_labels, dim=0)

    if not os.path.exists(Path_clustering_setup):  # Check if the picture path does not exist
        os.mkdir(Path_clustering_setup)  # Create the directory for plots if not yet created
        print(f"Created directory: {Path_clustering_setup}")  # Print directory creation feedback
    else:
        print(f"Directiry already exists: {Path_clustering_setup}")
    # manifold plot
    manifold_plot_selected_labels(all_encoded_data_filtered,
                all_labels_filtered,yaml_config["dataset"]["selected_labels"],
                saving_folder=Path_clustering_setup, plot_title="Manifold plot for all points with selected labels",
                file_saving_name="Manifold_plot_selected_labels", verbose=verbose)
    return encoded_points, ground_truth_labels

# main function executing Riemannian Kmeans 
def Riemannian_k_means_fit(encoded_points, params):
    """
    Executes the Riemannian K-Means clustering algorithm on data points encoded in a latent space.

    This algorithm clusters data points on a Riemannian manifold metric on which 
    is set by the pullback of Euclidean metric by the decoder.
    The algorithm iteratively minimizes the geodesic energy between each point and its assigned cluster centroid. 
    It supports different modes for geodesic computation: interpolation points or Schauder basis.

    Parameters:
        encoded_points (torch.Tensor): Encoded data points of shape (N, d), where N is the number of points and d is the dimension.
        params (dict): A dictionary of algorithm hyperparameters including:
            - N (int): Number of data points.
            - periodicity_mode (str): Mode to handle periodicity in the latent space.
            - K (int): Number of clusters.
            - torus_ae (nn.Module): Autoencoder model with a decoder for the torus manifold.
            - d (int): Dimensionality of the latent space.
            - beta (float): Step size for centroid updates.
            - learning_rate (float): Learning rate for geodesic optimization.
            - num_iter_outer (int): Number of outer iterations (cluster reassignment and centroid update).
            - num_iter_inner (int): Number of inner iterations (geodesic path optimization).
            - device (torch.device): Computation device (CPU or GPU).
            - mode (str): Geodesic computation mode ('Interpolation_points' or 'Schauder').
            - n_max (int): Maximal order of Schauder basis elements.
            - step_count (int): Number of steps or auxiliary points on each geodesic segment.

    Returns:
        dict: A dictionary containing:
            - "Riemannian_k_means_labels" (List[int]): Cluster index assigned to each data point.
            - "history" (List[dict]): A list of dicts tracking training progress, including:
                - intraclass_variance
                - intraclass_variance_by_cluster
                - norm_Frechet_mean_gradient
                - geodesics_to_nearest_centroids_lengths
                - geodesics_to_nearest_centroids_lengths_by_cluster
            - "geodesic_curve" (torch.Tensor): Tensor of shape (N, step_count, d) containing geodesic paths from each point to its assigned centroid.
            - "time_secs" (float): Total execution time in seconds.
    """
    N = params["N"]
    periodicity_mode = params["periodicity_mode"]
    K = params["K"]
    torus_ae = params["torus_ae"]
    d = params["d"]
    beta = params["beta"]
    learning_rate = params["learning_rate"]
    num_iter_outer = params["num_iter_outer"]
    num_iter_inner = params["num_iter_inner"]
    device = params["device"]
    mode = params["mode"]
    n_max = params["n_max"]
    step_count = params["step_count"]

    #initialization
    initial_centroids = initialize_centers(encoded_points, K, N) 
    current_centroids = torch.clone(initial_centroids) 

    if mode == "Interpolation_points":
        geodesic_solver = None
        # Initialize geodesic segments
        parameters_of_geodesics = construct_interpolation_points_on_segments_connecting_centers2encoded_data(
                encoded_points, 
                initial_centroids, 
                num_aux_points = step_count)
    elif mode == "Schauder":
        geodesic_solver = ricci_regularization.Schauder.NumericalGeodesics(n_max, step_count)
        # Get Schauder basis
        N_max = geodesic_solver.schauder_bases["zero_boundary"]["N_max"]
        basis = geodesic_solver.schauder_bases["zero_boundary"]["basis"]
        # Define parameters (batch_size × N_max × dim)
        parameters_of_geodesics = torch.zeros((N, K, N_max, d), requires_grad=True)
    init_parameters = torch.clone(parameters_of_geodesics) # save initial segments
    # Set optimizer params
    parameters = torch.nn.Parameter(parameters_of_geodesics) # Wrap as a parameter

    optimizer = torch.optim.SGD([parameters], lr=learning_rate)

    cluster_index_of_each_point = None
    geodesics_to_nearest_centroids = None

    #losses
    history = []

    # timing
    start_time = time.time()
    # sending the nn to selected device (usually it should be cuda)
    torus_ae.to(device)
    
    # ----------------------------
    # Riemannian K-means Algorithm
    # ----------------------------
    # Outer loop 
    t = tqdm(range(num_iter_outer), desc="Outer Loop iteration: 0")
    for iter_outer in t:    
        # Inner loop (refining geodesics)
        for iter_inner in range(num_iter_inner):
    #for iter_outer in range(num_iter_outer):
        # Inner loop (refining geodesics)
    #    for iter_inner in range(num_iter_inner):
            optimizer.zero_grad()  # Zero gradients
            # Compute the loss
            energies_of_geodesics = compute_energy(
                    mode = mode, 
                    parameters_of_geodesics=parameters, 
                    end_points = [encoded_points, current_centroids],
                    decoder = torus_ae.decoder_torus,
                    geodesic_solver = geodesic_solver,
                    reduction="none", device=device, 
                    periodicity_mode=periodicity_mode)
            loss_geodesics = energies_of_geodesics.sum()
            # Backpropagation: compute gradients
            loss_geodesics.backward()
            # Update parameters
            optimizer.step()
            # Store the loss value
        # end inner loop
        energies_of_geodesics = energies_of_geodesics.cpu()
        # compute geodesic_curve of shape (N,K,step_count,d)
        # compute a vector of length of all geodesics shape (N,K)
        with torch.no_grad():
            geodesic_curve, lengths_of_geodesics = compute_lengths(
                    mode = mode,
                    parameters_of_geodesics=parameters,
                    end_points = [encoded_points, current_centroids],
                    decoder = torus_ae.decoder_torus,
                    geodesic_solver = geodesic_solver,
                    reduction="none", device=device, 
                    periodicity_mode=periodicity_mode,
                    return_geodesic_curve=True) 
        lengths_of_geodesics = lengths_of_geodesics.cpu() # shape (N,K)
        geodesic_curve = geodesic_curve.cpu()

        # retrieve the class membership of each point by finding the closest cluster centroid 
        cluster_index_of_each_point = torch.argmin(lengths_of_geodesics, dim=1) # shape (N)
        batch_indices = torch.arange(N) # this is needed, since   geodesic_curve[:, cluster_index_of_each_point, :, :] will produce a tensor of shape (N,N,step_count,d)
        # pick only geodesics connecting points to cluster relevant centroids where the points are assigned
        geodesics_to_nearest_centroids = geodesic_curve[batch_indices, cluster_index_of_each_point, :, :].detach() # shape (N,step_count,d)

        # v is the direction to move the cluster centroids # shape (N,d)
        v = geodesics_to_nearest_centroids[:,-1,:] - geodesics_to_nearest_centroids[:,-2,:]
        v = v / v.norm(dim=1).unsqueeze(-1) # find the last segments of the geod shape (N,d)
        
        # Compute weighted Frechet mean gradient for each cluster
        weighted_v = lengths_of_geodesics[:, 0].unsqueeze(-1) * v  # Shape: (N, d)
        # Create a one-hot encoding of the cluster indices
        one_hot_clusters = torch.nn.functional.one_hot(cluster_index_of_each_point, num_classes=K).float()  # Shape: (N, K)
        # Compute the gradients for each cluster
        Frechet_mean_gradient = one_hot_clusters.T @ weighted_v  # Shape: (K, d)
        # Update cluster centroids
        with torch.no_grad():
            current_centroids += - beta * Frechet_mean_gradient  # Update all centroids simultaneously

        # Compute average Frechet mean gradient norm among the K clusters on step iter_outer 
        average_Frechet_mean_gradient_norm = (Frechet_mean_gradient.norm(dim=1).mean()).item()

        # saving the lengths of geodesics_to_nearest_centroids
        geodesics_to_nearest_centroids_lengths = lengths_of_geodesics[batch_indices, cluster_index_of_each_point]
        
        # save intra-class variance
        intraclass_variance = (1/N) * energies_of_geodesics[batch_indices, cluster_index_of_each_point]
        
        #compute the sum of geodesic length for each cluster
        #scatter_add_ is the reverse of torch.gather
        length_of_geodesics_to_nearest_centroids_by_cluster = torch.zeros(K, dtype=geodesics_to_nearest_centroids_lengths.dtype)
        length_of_geodesics_to_nearest_centroids_by_cluster.scatter_add_(0, cluster_index_of_each_point, geodesics_to_nearest_centroids_lengths)    
        
        #compute the Intra-class variance, i.e. sum of geodesic energy for each cluster
        #scatter_add_ is the reverse of torch.gather
        intraclass_variance_by_cluster = torch.zeros(K, dtype=geodesics_to_nearest_centroids_lengths.dtype)
        intraclass_variance_by_cluster.scatter_add_(0, cluster_index_of_each_point, intraclass_variance)    
        
        history_item = {
            "intraclass_variance"                              : intraclass_variance.detach().sum().numpy(),
            "intraclass_variance_by_cluster"                   : intraclass_variance_by_cluster.unsqueeze(0).detach().numpy(), 
            "norm_Frechet_mean_gradient"                       : average_Frechet_mean_gradient_norm,
            "geodesics_to_nearest_centroids_lengths"           : geodesics_to_nearest_centroids_lengths.detach().sum().numpy(),
            "geodesics_to_nearest_centroids_lengths_by_cluster": length_of_geodesics_to_nearest_centroids_by_cluster.unsqueeze(0).detach().numpy()
        }
        history.append( history_item )
        t.set_description(f"Outer Loop iteration: {iter_outer+1}, Centroid gradient norm:{average_Frechet_mean_gradient_norm:.4f}, Total geodesic energy:{loss_geodesics:.4f}")  # Update description dynamically
    #timing
    end_time = time.time()
    algorithm_execution_time = end_time - start_time
    results = {
        "Riemannian_k_means_labels": cluster_index_of_each_point.tolist(), 
        "history": history, 
        "geodesic_curve": geodesic_curve, 
        "time_secs": algorithm_execution_time
    }
    return results


def Riemannian_k_means_losses_plot(history, Path_pictures, verbose = False):
    norm_Frechet_mean_gradient_history = []
    geodesics_to_nearest_centroids_lengths_by_cluster_history = []
    geodesics_to_nearest_centroids_lengths_history = []
    intraclass_variance_by_cluster_history = []
    intraclass_variance_history = []
    for i in range(len(history)):
        norm_Frechet_mean_gradient_history.append(history[i]["norm_Frechet_mean_gradient"])
        geodesics_to_nearest_centroids_lengths_by_cluster_history.append(history[i]["geodesics_to_nearest_centroids_lengths_by_cluster"])
        geodesics_to_nearest_centroids_lengths_history.append(history[i]["geodesics_to_nearest_centroids_lengths"])
        intraclass_variance_by_cluster_history.append(history[i]["intraclass_variance_by_cluster"])
        intraclass_variance_history.append(history[i]["intraclass_variance"])

    #plotting 
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Create a figure with 1 row and 3 columns

    # Plot norm_Frechet_mean_gradient_history
    axes[0].plot(norm_Frechet_mean_gradient_history, marker='o', markersize=3) 
    axes[0].set_title('Average norm of gradients of centroids')
    axes[0].set_xlabel('Outer loop iterations')
    axes[0].set_ylabel('Loss')

    # Plot geodesics_to_nearest_centroids lengths by cluster
    # Generate a color palette with distinct colors
    K = len(intraclass_variance_by_cluster_history[0])
    colors = plt.cm.jet(torch.linspace(0, 1, K))  # Use a colormap (e.g., 'viridis')

    #lengths_of_geodesics_to_nearest_centroids_concatenated = torch.cat((geodesics_to_nearest_centroids_lengths_by_cluster_history), dim=0).detach()
    lengths_of_geodesics_to_nearest_centroids_concatenated = np.concatenate(geodesics_to_nearest_centroids_lengths_by_cluster_history)
    for i in range(K):
        axes[1].plot(lengths_of_geodesics_to_nearest_centroids_concatenated[:, i],marker='o',markersize=3,
                        label=f'Lengths of geodesics in cluster {i}', color=colors[i])
    axes[1].set_xlabel('Outer Loop Iterations')
    axes[1].set_ylabel('Loss')
    axes[1].legend()

    # Plot geodesics_to_nearest_centroids_lengths_history
    axes[1].plot(geodesics_to_nearest_centroids_lengths_history, marker='o', markersize=3, 
                label='Lengths of geodesics in all clusters', color='green')
    axes[1].set_title('Lengths of geodesics to nearest centroids')
    axes[1].set_xlabel('Outer loop iterations')
    axes[1].legend(loc= 'upper right')

    intraclass_variance_concatenated = np.concatenate(intraclass_variance_by_cluster_history)
    #torch.cat((intraclass_variance_by_cluster_history), dim=0).detach()
    for i in range(K):
        axes[2].plot(intraclass_variance_concatenated[:, i],marker='o',markersize=3,
                        label=f'Variance of geodesics of cluster {i} ', color=colors[i])
    axes[2].set_xlabel('Outer Loop Iterations')
    axes[2].set_ylabel('Loss')
    axes[2].legend()

    # Plot geodesics_to_nearest_centroids_lengths_history
    axes[2].plot(intraclass_variance_history, marker='o', markersize=3,
                label='Intra-class variance', color='green')
    axes[2].set_title('Intra-class variances')
    axes[2].set_xlabel('Outer loop iterations')
    axes[2].legend()

    # Adjust layout
    plt.tight_layout()
    if Path_pictures!=None:
        plt.savefig(f"{Path_pictures}/kmeans_losses.pdf",bbox_inches='tight', format="pdf")
    if verbose == True:
        plt.show()
    else:
        plt.close()
    return