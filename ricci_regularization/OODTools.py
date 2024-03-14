import torch
import math
import ricci_regularization
def curv_loss_on_OOD_samples(extreme_curv_points_tensor, decoder, sigma_ood,n_ood, N_extr,latent_space_dim):
    with torch.no_grad():
        centers = extreme_curv_points_tensor.repeat_interleave(n_ood,dim=0)
        samples_centered_at_zero = (sigma_ood**2)*torch.randn(N_extr*n_ood, latent_space_dim)
        OOD_batch = centers + samples_centered_at_zero
    OOD_batch.requires_grad_()
    
    # OOD loss function
    metric_on_OOD = ricci_regularization.metric_jacfwd_vmap(OOD_batch,
                                           function=decoder)
    det_on_OOD = torch.det(metric_on_OOD)
    volume_on_OOD = torch.sqrt(det_on_OOD).detach()
    Scalar_curvature_on_OOD = ricci_regularization.Sc_jacfwd_vmap(OOD_batch,
                                           function=decoder)
    
    loss = (torch.square(Scalar_curvature_on_OOD)*volume_on_OOD).mean()
    return loss

def find_extreme_curvature_points(data_batch, extreme_curv_points_tensor,
                 extreme_curv_value_tensor,batch_idx,encoder, decoder,
                 r_ood, N_extr, output_dim, print_values = False):
    # Exrteme curvature batch
    new_curv_points_tensor = encoder(data_batch.view(-1,output_dim))
    new_curv_points_tensor = new_curv_points_tensor.detach()
    new_curv_value_tensor = ricci_regularization.Sc_jacfwd_vmap(new_curv_points_tensor, function = decoder)
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
    extreme_curv_points_tensor = torch.cat((extreme_curv_points_tensor[:N_extr//2],extreme_curv_points_tensor[-N_extr//2:]),dim=0)
    extreme_curv_value_tensor = torch.cat((extreme_curv_value_tensor[:N_extr//2],extreme_curv_value_tensor[-N_extr//2:]),dim=0)
    extreme_curv_points_tensor = extreme_curv_points_tensor.detach()
    metric_on_OOD = ricci_regularization.metric_jacfwd_vmap(extreme_curv_points_tensor,
                                           function=decoder)
    det_on_OOD = torch.det(metric_on_OOD)
    volume_on_OOD = torch.sqrt(det_on_OOD).detach()
    if print_values == True:
        print(f"\nafter {batch_idx} batches we keep",extreme_curv_value_tensor.shape[0],
                "points with extreme curvature values: \n",extreme_curv_value_tensor,
                "\nvolume form values:\n", volume_on_OOD)
    #            "\n at points:\n", extreme_curv_points_tensor)
        
    # multiply curv values by decay factor
    extreme_curv_value_tensor = math.exp(-r_ood)*extreme_curv_value_tensor
    
    # if not enough points, keep 16 of each (min and max) anyway      
    # but when OOD sampling, sample around min negative
    # and max positive. Print how many of each are used!
    # OOD sampling
    return extreme_curv_points_tensor, extreme_curv_value_tensor