import torch
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
    all_points_on_geodesics = starting_points_expanded + t * (final_points_expanded - starting_points_expanded)  # Shape: (num_data_points, num_clusters, num_aux_points, latent_dim)

    if cut_off_ends == True:
        # Select interpolation_points cutting of the starting and the final point for every segment
        interpolation_points = all_points_on_geodesics[:,:,1:-1,:]
    else:
        interpolation_points = all_points_on_geodesics
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
    all_points_on_geodesics = torch.cat((starting_points_expanded, parameters_of_geodesics, final_points_expanded),dim=2) 
    return all_points_on_geodesics

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

def compute_energy(mode, parameters_of_geodesics, end_points, decoder, geodesic_solver=None):
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
    energy = ( (decoded_geodesic_curve[:,:, 1:, :] - decoded_geodesic_curve[:,:, :-1, :]) ** 2 ).sum()
    # Warning! the outpiut is the single scalar, i.e the sum of all the energies
    return energy

def compute_lengths(mode, parameters_of_geodesics, end_points, decoder, geodesic_solver=None):
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
    computed_lengths = torch.sqrt((tangent_vectors**2).sum(dim=(-2,-1))) # comute Euclidean compute_lengths of the curves in R^D
    # Warning! the outpiut is the vector of length of all geodesics 
    return computed_lengths

# ---------------------------------------
#plotting

def plot_octopus(all_points_on_geodesics, memberships = None, meaningful_geodesics = None, 
                 saving_folder = None, suffix = None, verbose = True,
                 xlim = torch.pi, ylim = torch.pi):
    """
    Plots geodesics, centers, and datapoints on a 2D plane with options to visualize memberships 
    and meaningful geodesics, and save the resulting plot.

    Parameters:
    - all_points_on_geodesics (torch.Tensor): Tensor of shape (num_datapoints, num_clusters, num_aux_points_on_geodesics, latent_dimension) 
      representing the geodesic segments.
    - memberships (torch.Tensor, optional): Tensor of shape (num_datapoints,) representing class memberships for each datapoint.
    - meaningful_geodesics (torch.Tensor, optional): Tensor of geodesics to be overlaid for visualization.
    - saving_folder (str, optional): Folder path to save the plots.
    - suffix (int, optional): Frame index to append to the saved filename.
    - verbose (bool, optional): Whether to display the plot. If False, the plot will be closed after saving.
    - xlim (float or None, optional): Limits for the x-axis. If None, no limit is applied.
    - ylim (float or None, optional): Limits for the y-axis. If None, no limit is applied.

    Returns:
    - None
    """
    # Detach all_points_on_geodesics from computation graph to avoid accidental gradients.
    all_points_on_geodesics = all_points_on_geodesics.detach() #shape (num_datapoints, num_clusters, num_aux_points_on_geodesics,latent_dimension)
    N = all_points_on_geodesics.shape[0] # num data points
    K = all_points_on_geodesics.shape[1] # num clusters
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
            plt.plot(all_points_on_geodesics[i,j,:,0], all_points_on_geodesics[i,j,:,1],'-',marker='o', c = color, markersize=3)
    # plot centers
    centers = all_points_on_geodesics[0,:,-1,:]
    # plot the datapoints (the starting points on all the geodesics, colored by memberships if specified):
    if memberships!= None:
        num_classes = int(memberships.max().item()) + 1
        plt.scatter(centers[:,0], centers[:,1], c=torch.arange(num_classes), marker='*', edgecolor='black',  cmap=ricci_regularization.discrete_cmap(num_classes, 'jet'), s = 170,zorder = 12)
        plt.scatter(all_points_on_geodesics[:,0,0,0], all_points_on_geodesics[:,0,0,1], c=memberships, marker='o', edgecolor='none', cmap=ricci_regularization.discrete_cmap(num_classes, 'jet'), s = 30,zorder = 10)
    else:
        plt.scatter(centers[:,0], centers[:,1], c="red", label = "centers", marker='*', edgecolor='black', s = 170,zorder = 10)
        plt.scatter(all_points_on_geodesics[:,0,0,0], all_points_on_geodesics[:,0,0,1], c="green", label = "centers", marker='o', s = 30,zorder = 10)
    if meaningful_geodesics != None:
        num_classes = int(memberships.max().item()) + 1
        cmap = ricci_regularization.discrete_cmap(num_classes, 'jet')  # Define your colormap
        colors = cmap(memberships.cpu().numpy())  # Map memberships to colors
        for i in range(N):
            plt.plot(meaningful_geodesics[i,:,0], meaningful_geodesics[i,:,1], c=colors[i], zorder = 10)
    # save images
    if saving_folder != None:
        if suffix != None:
            plt.savefig(saving_folder + f"/frame_{suffix:04d}.png")
    if verbose == False:
        plt.close()
    plt.show()
    return  