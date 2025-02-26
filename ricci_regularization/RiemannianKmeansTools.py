import torch
import numpy as np
import matplotlib.pyplot as plt
import ricci_regularization

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

def geodesics_from_parameters_schauder(geodesic_solver, parameters_of_geodesics, end_points):
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

    linear_curve = construct_interpolation_points_on_segments_connecting_centers2encoded_data(starting_points=starting_points, final_points=final_points,num_aux_points=step_count, cut_off_ends=False)
    
    geodesic_curve = linear_curve + torch.einsum("sn,bend->besd", basis, parameters)
    return geodesic_curve

def compute_energy(mode, parameters_of_geodesics, end_points, decoder, geodesic_solver=None, reduction = "sum"):
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
        geodesic_curve = geodesics_from_parameters_schauder(geodesic_solver, parameters_of_geodesics, end_points)
    # decode the geodesics
    decoded_geodesic_curve = decoder(geodesic_curve)
    # Compute energy (finite differences)
    tangent_vectors = decoded_geodesic_curve[:,:,1:,:] - decoded_geodesic_curve[:,:,:-1,:]
    if reduction == "none":
        energy = (tangent_vectors**2).sum(dim=(-2,-1)) # comute Euclidean compute_lengths of the curves in R^D
    if reduction == "sum":
        energy = (tangent_vectors**2).sum()
    # Warning! by default the outpiut is the single scalar, i.e the sum of all the energies
    return energy

def compute_lengths(mode, parameters_of_geodesics, end_points, decoder, geodesic_solver=None, reduction="none"):
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
        geodesic_curve = geodesics_from_parameters_schauder(geodesic_solver, parameters_of_geodesics, end_points)
    decoded_geodesic_curve = decoder(geodesic_curve)
    tangent_vectors = decoded_geodesic_curve[:,:,1:,:] - decoded_geodesic_curve[:,:,:-1,:]
    if reduction == "none":
        lengths = (tangent_vectors).norm(dim=(-1)).sum(dim=(-1)) # first find all norms of small tangent vectors in the discretization then sum them for each geodesic
    if reduction == "sum":
        lengths = (tangent_vectors).norm(dim=(-1)).sum()
    if reduction == "old":    
        lengths = torch.sqrt((tangent_vectors**2).sum(dim=(-2,-1))) # seems to be a wrong formula
    # Warning! by default the outpiut is the vector of length of all geodesics 
    return lengths

# ---------------------------------------
#plotting

def plot_octopus(geodesic_curve, memberships = None, 
                 saving_folder = None, suffix = None, verbose = True,
                 xlim = torch.pi, ylim = torch.pi):
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
    # Detach geodesic_curve from computation graph to avoid accidental gradients.
    geodesic_curve = geodesic_curve.detach() #shape (num_datapoints, num_clusters, num_aux_points_on_geodesics,latent_dimension)
    N = geodesic_curve.shape[0] # num data points
    K = geodesic_curve.shape[1] # num clusters
    if xlim != None:
        plt.xlim(-xlim, xlim)
    if ylim != None:
        plt.ylim(-xlim, xlim)
    for i in range(N):
        for j in range(K):
            #if j == 0:
            #    color = "blue"
            #else:
            color = "orange"
            plt.plot(geodesic_curve[i,j,:,0], geodesic_curve[i,j,:,1],'-',marker='o', c = color, markersize=3)
    # plot centers
    centers = geodesic_curve[0,:,-1,:]
    # plot the datapoints (the starting points on all the geodesics, colored by memberships if specified):
    if memberships!= None:
        num_classes = int(memberships.max().item()) + 1
        plt.scatter(centers[:,0], centers[:,1], c=torch.arange(num_classes), marker='*', edgecolor='black',  cmap=ricci_regularization.discrete_cmap(num_classes, 'jet'), s = 170,zorder = 12)
        plt.scatter(geodesic_curve[:,0,0,0], geodesic_curve[:,0,0,1], c=memberships, marker='o', edgecolor='none', cmap=ricci_regularization.discrete_cmap(num_classes, 'jet'), s = 30,zorder = 10)
        # recompute the geodesics to nearest centroids
        batch_indices = torch.arange(N) # this is needed, since   geodesic_curve[:, cluster_index_of_each_point, :, :] will produce a tensor of shape (N,N,step_count,d)
        # pick only geodesics connecting points to cluster relevant centroids where the points are assigned
        geodesics2nearestcentroids = geodesic_curve[batch_indices, memberships, :, :].detach() # shape (N,step_count,d)
        # color the geodesics
        cmap = ricci_regularization.discrete_cmap(num_classes, 'jet')  # Define your colormap
        colors = cmap(memberships.cpu().numpy())  # Map memberships to colors
        for i in range(N):
            plt.plot(geodesics2nearestcentroids[i,:,0], geodesics2nearestcentroids[i,:,1], c=colors[i], zorder = 10)
    else:
        plt.scatter(centers[:,0], centers[:,1], c="red", label = "centers", marker='*', edgecolor='black', s = 170,zorder = 10)
        plt.scatter(geodesic_curve[:,0,0,0], geodesic_curve[:,0,0,1], c="green", label = "centers", marker='o', s = 30,zorder = 10)
    # save images
    if saving_folder != None:
        if suffix != None:
            plt.savefig(saving_folder + f"/frame_{suffix:04d}.png")
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
    plt.legend(handles, selected_labels, title="Ground Truth Labels")
    plt.title(plot_title)
    if save_plot==True:
        plt.savefig(f"{saving_folder}/{file_saving_name}.pdf",bbox_inches='tight', format="pdf")
        print(f"manifold plot saved as{saving_folder}/{file_saving_name}.pdf")
    if verbose==True:
        plt.show()

# Decision boundary plot
from scipy.interpolate import griddata
def plot_knn_decision_boundary(encoded_points, labels_for_coloring, grid_resolution=100, 
                               neighbours_number=7, interpolation_method='linear', verbose=True, save_plot=True, 
                               saving_folder=None, file_saving_name=None, 
                               cmap_points='jet', cmap_background='coolwarm'):
    """
    Plots a smoother decision boundary of a k-Nearest Neighbors (k-NN) classifier in a 2D latent space.
    
    This version uses `scipy.interpolate.griddata` to create a smoothly interpolated background.
    """
    # Convert tensors to numpy
    encoded_points_np = encoded_points.numpy()
    labels_for_coloring_np = labels_for_coloring.numpy()
    
    # Create a dense uniform grid in [-pi, pi]^2
    x_vals = np.linspace(-np.pi, np.pi, grid_resolution)
    y_vals = np.linspace(-np.pi, np.pi, grid_resolution)
    grid_x, grid_y = np.meshgrid(x_vals, y_vals)
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    
    # Compute pairwise Euclidean distances between grid points and encoded data
    dists = np.linalg.norm(grid_points[:, None, :] - encoded_points_np[None, :, :], axis=2)
    
    # Find the indices of the k=neighbours_number nearest neighbors
    nn_indices = np.argsort(dists, axis=1)[:, :neighbours_number]
    
    # Assign cluster labels by majority vote
    nn_labels = labels_for_coloring_np[nn_indices]
    grid_labels = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=nn_labels)
    
    # Smooth interpolation using griddata
    grid_labels_smooth = griddata(grid_points, grid_labels, (grid_x, grid_y), method=interpolation_method)
    
    # Plot results
    plt.figure(figsize=(8, 6))
    plt.contourf(grid_x, grid_y, grid_labels_smooth, levels=10, cmap=cmap_background, alpha=0.6)
    
    # Overlay original encoded points
    plt.scatter(encoded_points_np[:, 0], encoded_points_np[:, 1], c=labels_for_coloring_np, cmap=cmap_points, edgecolors='k', s=40)
    
    # Set plot limits and labels
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-np.pi, np.pi)
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title(f"{file_saving_name.replace('_',' ')} via {neighbours_number}-NN")
    
    if save_plot:
        plt.savefig(f"{saving_folder}/{file_saving_name}.pdf", bbox_inches='tight', format="pdf")
        print(f"Decision boundary plot saved as {saving_folder}/{file_saving_name}.pdf")
    
    if verbose:
        plt.show()

#older version to be deprecated
def plot_knn_decision_boundary_nonsmooth(encoded_points, labels_for_coloring, grid_resolution = 100, 
        neighbours_number = 7,verbose=True, save_plot=True, saving_folder=None, 
        file_saving_name=None, cmap_points='jet',cmap_background='coolwarm'):
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
    # (||x - y||^2 = (x1 - y1)^2 + (x2 - y2)^2)
    dists = torch.cdist(grid_points, encoded_points)  # Shape: (grid_size^2, N_encoded)

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
    plt.title(f"{file_saving_name.replace('_',' ')} via {neighbours_number}-NN")
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