import torch
import math
import ricci_regularization
def curv_loss_on_OOD_samples(extreme_curv_points_tensor, decoder, OOD_params_dict, latent_space_dim):
    """
    Computes the curvature loss for out-of-distribution (OOD) samples generated around extreme curvature points.

    This function generates OOD samples by perturbing extreme curvature points, and then calculates the curvature loss 
    on those samples.

    Parameters:
    - extreme_curv_points_tensor: Tensor containing extreme curvature points from the model.
    - decoder: The decoder function used to generate data from latent space.
    - OOD_params_dict: Dictionary containing parameters for OOD generation, including the number of samples and variance.
    - latent_space_dim: The dimension of the latent space in which the decoder operates.

    Returns:
    - curvature_loss_on_OOD_samples: The computed curvature loss for the generated OOD samples.
    """
    device = extreme_curv_points_tensor.device
    with torch.no_grad():
        centers = extreme_curv_points_tensor.repeat_interleave(OOD_params_dict["n_ood"],dim=0)
        centers = centers.to(device)
        samples_centered_at_zero = (OOD_params_dict["sigma_ood"]**2)*torch.randn(OOD_params_dict["N_extr"]*OOD_params_dict["n_ood"], latent_space_dim, device=device)
        OOD_batch = centers + samples_centered_at_zero
    OOD_batch.requires_grad_()
    # OOD loss function
    curvature_loss_on_OOD_samples = ricci_regularization.Ricci.curvature_loss_jacfwd(points=OOD_batch, function=decoder)
    return curvature_loss_on_OOD_samples

def find_extreme_curvature_points(data_batch, extreme_curv_points_tensor,
                 extreme_curv_value_tensor,batch_idx,encoder, decoder,
                 OOD_params_dict, output_dim, verbose = False):
    """
    Finds and updates extreme curvature points by computing the curvature values for a given batch,
    selecting the most extreme curvature points (both positive and negative), and keeping track of them.

    This function maintains a dynamic set of extreme curvature points and their corresponding curvature values,
    which are used for Out-Of-Distribution (OOD) sampling.

    Parameters:
    - data_batch: The current batch of data for which curvature values are calculated.
    - extreme_curv_points_tensor: The current set of extreme curvature points.
    - extreme_curv_value_tensor: The curvature values corresponding to the extreme curvature points.
    - batch_idx: The current batch index (for printing purposes if verbose is enabled).
    - encoder: The encoder function used to map the data to the latent space.
    - decoder: The decoder function used to evaluate the curvature.
    - OOD_params_dict: Dictionary containing OOD sampling parameters (e.g., number of extreme points, decay factor).
    - output_dim: The dimensionality of the output space (latent space).
    - verbose: If True, prints additional information about the extreme curvature points and their volumes.

    Returns:
    - extreme_curv_points_tensor: Updated tensor of extreme curvature points.
    - extreme_curv_value_tensor: Updated tensor of curvature values corresponding to the extreme curvature points.
    """
    # Exrteme curvature batch
    new_curv_points_tensor = encoder(data_batch.view(-1,output_dim))
    new_curv_points_tensor = new_curv_points_tensor.detach()
    new_curv_value_tensor,_ = ricci_regularization.Sc_g_jacfwd_vmap(new_curv_points_tensor, function = decoder)
    new_curv_value_tensor = new_curv_value_tensor.detach()
    
    # merge extreme points and new batch 
    extreme_curv_points_tensor = torch.cat((extreme_curv_points_tensor,
                                            new_curv_points_tensor),dim=0)
    extreme_curv_value_tensor = torch.cat((extreme_curv_value_tensor,
                                            new_curv_value_tensor),dim=0)
    
    # sort by curvature value points and curvature values. 
    indices = torch.argsort(extreme_curv_value_tensor)
    extreme_curv_points_tensor = torch.index_select(extreme_curv_points_tensor,dim = 0, index= indices)
    extreme_curv_value_tensor = torch.index_select(extreme_curv_value_tensor,dim = 0, index= indices)
    
    # take most N_extr//2 negative and N_extr//2 most positive
    extreme_curv_points_tensor = torch.cat((extreme_curv_points_tensor[:OOD_params_dict["N_extr"]//2],extreme_curv_points_tensor[-OOD_params_dict["N_extr"]//2:]),dim=0)
    extreme_curv_value_tensor = torch.cat((extreme_curv_value_tensor[:OOD_params_dict["N_extr"]//2],extreme_curv_value_tensor[-OOD_params_dict["N_extr"]//2:]),dim=0)
    extreme_curv_points_tensor = extreme_curv_points_tensor.detach()
    if verbose == True:
        metric_on_OOD_samples = ricci_regularization.metric_jacfwd_vmap(extreme_curv_points_tensor,
                                           function=decoder)
        det_on_OOD_samples = torch.det(metric_on_OOD_samples)
        volume_on_OOD_samples = torch.sqrt(det_on_OOD_samples).detach()
        print(f"\nafter {batch_idx} batches we keep",extreme_curv_value_tensor.shape[0],
                "points with extreme curvature values: \n",extreme_curv_value_tensor,
                "\nvolume form values:\n", volume_on_OOD_samples)
    #            "\n at points:\n", extreme_curv_points_tensor)
    # end if
       
    # multiply curv values by decay factor
    extreme_curv_value_tensor = math.exp(-OOD_params_dict["r_ood"])*extreme_curv_value_tensor
    
    # if not enough points, keep 16 of each (min and max) anyway      
    # but when OOD sampling, sample around min negative
    # and max positive. Print how many of each are used!
    # OOD sampling
    return extreme_curv_points_tensor, extreme_curv_value_tensor
