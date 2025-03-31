import torch
import ricci_regularization
from tqdm.notebook import tqdm

def compute_equidistribution_loss(z, latent_dim, num_moments):
    """
    Computes an equidistribution loss for a latent space representation using moment matching with uniform distribution.

    This loss encourages the latent vectors to follow a uniform distribution by minimizing 
    deviations from the expected statistical moments of a uniform distribution. It is based 
    on Chebyshev polynomials to match different orders of moments.

    Parameters:
    - z (torch.Tensor): Latent representation of shape (batch_size, 2 * latent_dim).
    - latent_dim (int): Dimension of the latent space.
    - num_moments (int): Number of moments to consider (up to 4).

    Returns:
    - torch.Tensor: Scalar loss value encouraging uniform distribution in latent space.
    """
    z_sin = z[:, 0:latent_dim]
    z_cos = z[:, latent_dim:2*latent_dim]

    mode1 = torch.mean(z, dim=0)
    mode1 = torch.sum(mode1 * mode1)

    mode2_1 = torch.mean(2 * z_cos * z_cos - 1, dim=0)
    mode2_1 = torch.sum(mode2_1 * mode2_1)
    mode2_2 = torch.mean(2 * z_sin * z_cos, dim=0)
    mode2_2 = torch.sum(mode2_2 * mode2_2)
    mode2 = mode2_1 + mode2_2

    equidistribution_loss = mode1 + mode2

    if num_moments > 2:
        mode3_1 = torch.mean(4 * z_cos**3 - 3 * z_cos, dim=0)
        mode3_1 = torch.sum(mode3_1 * mode3_1)
        mode3_2 = torch.mean(z_sin * (8 * z_cos**3 - 4 * z_cos), dim=0)
        mode3_2 = torch.sum(mode3_2 * mode3_2)
        mode3 = mode3_1 + mode3_2
        equidistribution_loss += mode3

    if num_moments > 3:
        mode4_1 = torch.mean(8 * z_cos**4 - 8 * z_cos**2 + 1, dim=0)
        mode4_1 = torch.sum(mode4_1 * mode4_1)
        mode4_2 = torch.mean(z_sin * (16 * z_cos**4 - 12 * z_cos**2 + 1), dim=0)
        mode4_2 = torch.sum(mode4_2 * mode4_2)
        mode4 = mode4_1 + mode4_2
        equidistribution_loss += mode4

    return equidistribution_loss


# this function is not used yet. to be called from loss_funtion, now it is inside it
def compute_contractive_loss(data,torus_ae, training_config):
    """
    Computes the contractive loss for the given data based on the Jacobian norm of the encoder.

    The contractive loss encourages the encoder's output to have small variations, thereby regularizing the model
    by penalizing large gradients with respect to the input.

    Parameters:
    - data: Input data batch for which the loss is computed.
    - torus_ae: The trained torus autoencoder model.
    - training_config: Configuration dictionary containing architecture and loss settings.

    Returns:
    - max_loss: The maximum contractive loss for the batch, based on the Jacobian norm of the encoder.
    """
    encoder_jac_norm = ricci_regularization.Jacobian_norm_jacrev_vmap( data, 
                                function = torus_ae.encoder_torus,
                                input_dim = training_config["architecture"]["input_dim"] )
    outlyers_encoder_jac_norm = encoder_jac_norm - training_config["loss_settings"]["delta_encoder"]
    return torch.nn.ReLU()( outlyers_encoder_jac_norm ).max()

def compute_curvature_loss(data, torus_ae, training_config):
    try:
        uniform_OOD_sampling_flag  = training_config["training_mode"]["uniform_OOD_sampling_flag"]
    except KeyError:
        uniform_OOD_sampling_flag = False
        print("OOD uniform samling flag not set, prenalizing curvature on encoded data")
    if uniform_OOD_sampling_flag == True:
        num_sampled_points = training_config["OOD_settings"]["n_OOD"]
        d = training_config["architecture"]["latent_dim"]
        sampled_points = (torch.rand((num_sampled_points, d)) * (2 * torch.pi)) - torch.pi
        encoded_points_to_compute_curvature = sampled_points.to(data.device)
    else:
        encoded_points_to_compute_curvature = torus_ae.encoder_to_lifting(data).detach()

    if training_config["training_mode"]["curvature_computation_mode"] == "jacfwd":
        curvature_loss = ricci_regularization.Ricci.curvature_loss_jacfwd(
            encoded_points_to_compute_curvature, 
            function=torus_ae.decoder_torus, 
            eps=training_config["loss_settings"]["eps"]
        )
    elif training_config["training_mode"]["curvature_computation_mode"] == "fd":
        curvature_loss["Curvature"] = ricci_regularization.FiniteDifferences.curvature_loss_fd(
            points=encoded_points_to_compute_curvature,
            function=torus_ae.decoder_torus,
            h=0.01,
            eps=training_config["loss_settings"]["eps"]
        )
    return curvature_loss

def compute_all_losses_for_diagnostic(data, torus_ae, training_config):
    dict_losses_diagnostic = {}
    
    # choose points for curvature penalization
    if training_config["training_mode"]["uniform_OOD_sampling_flag"]:
        num_sampled_points = training_config["OOD_settings"]["n_OOD"]
        d = training_config["architecture"]["latent_dim"]
        sampled_points = (torch.rand((num_sampled_points, d)) * (2 * torch.pi)) - torch.pi
        encoded_points_to_compute_curvature = sampled_points.to(data.device)
    else:
        encoded_points_to_compute_curvature = torus_ae.encoder_to_lifting(data).detach()

    # compute curvature loss if needed: Sc is scalar curvature, g is metric
    if training_config["training_mode"]["compute_curvature"] == True:
        Sc_on_data, g_on_data = ricci_regularization.Sc_g_jacfwd_vmap(
                encoded_points_to_compute_curvature,
                function=torus_ae.decoder_torus,
                eps=training_config["loss_settings"]["eps"]
            )
        dict_losses_diagnostic["curv_squared_mean"] = torch.square(Sc_on_data.detach()).mean()
        dict_losses_diagnostic["curv_squared_max"] = torch.square(Sc_on_data.detach()).max()
    else:
        encoded_points_no_grad = torus_ae.encoder_to_lifting(data).detach()
        g_on_data = ricci_regularization.metric_jacfwd_vmap(
            encoded_points_no_grad,
            function=torus_ae.decoder_torus
        )
    
    # compute metric determinant and other losses
    det_g_on_data = torch.det(g_on_data)
    dict_losses_diagnostic["Curvature"] = ( ( Sc_on_data**2 ) * torch.sqrt( det_g_on_data ) ).mean()    
    g_inv_train_batch = torch.linalg.inv(
        g_on_data + training_config["loss_settings"]["eps"] * torch.eye(
            training_config["architecture"]["latent_dim"]
        ).to(data.device)
    )
    g_inv_norm_train_batch = torch.linalg.matrix_norm(g_inv_train_batch)

    dict_losses_diagnostic["g_inv_norm_mean"] = torch.mean(g_inv_norm_train_batch)
    dict_losses_diagnostic["g_inv_norm_max"] = torch.max(g_inv_norm_train_batch)
    dict_losses_diagnostic["g_det_mean"] = det_g_on_data.mean()
    dict_losses_diagnostic["g_det_max"] = det_g_on_data.max()
    dict_losses_diagnostic["g_det_min"] = det_g_on_data.min()

    decoder_jac_norm = torch.func.vmap(torch.trace)(g_on_data)
    dict_losses_diagnostic["decoder_jac_norm_mean"] = decoder_jac_norm.mean()
    dict_losses_diagnostic["decoder_jac_norm_max"] = decoder_jac_norm.max()
    dict_losses_diagnostic["decoder_contractive_loss"] = torch.nn.ReLU()(decoder_jac_norm).max()

    # contractive loss (of the encoder)
    encoder_jac_norm = ricci_regularization.Jacobian_norm_jacrev_vmap(
        data, 
        function=torus_ae.encoder_torus,
        input_dim=training_config["architecture"]["input_dim"]
    )
    outlyers_encoder_jac_norm = encoder_jac_norm - training_config["loss_settings"]["delta_encoder"]
    dict_losses_diagnostic["Contractive"] = torch.nn.ReLU()( outlyers_encoder_jac_norm ).max()
    dict_losses_diagnostic["encoder_jac_norm_mean"] = encoder_jac_norm.mean()
    dict_losses_diagnostic["encoder_jac_norm_max"] = encoder_jac_norm.max()
    return dict_losses_diagnostic

# Loss = MSE + equidistribution + contractive + curvature 
# add loss computation mode, that is training_config["training_mode"]
def loss_function(recon_data, data, z, torus_ae, training_config):
    """
    Parameters:
    - recon_data (torch.Tensor): Reconstructed data.
    - data (torch.Tensor): Original input data.
    - z (torch.Tensor): Latent representation.
    - torus_ae (object): The toroidal autoencoder model.
    - training_config (dict): Configuration dictionary containing training settings.

    Returns:
    - dict: A dictionary containing different loss components.
    """
    MSE = torch.nn.functional.mse_loss(recon_data, data, reduction='mean')

    equidistribution_loss = compute_equidistribution_loss(z,
        latent_dim=training_config["architecture"]["latent_dim"],
        num_moments=training_config["loss_settings"]["num_moments"])
    
    dict_losses = {
        "MSE": MSE,
        "Equidistribution": equidistribution_loss,
    }

    # check if full diagnostic loss computation is required
    if training_config["training_mode"]["diagnostic_mode"]:
        dict_diagnostic = compute_all_losses_for_diagnostic(data, torus_ae, training_config)
        dict_losses.update(dict_diagnostic)
    
    else:
        if training_config["training_mode"]["compute_contractive_loss"]:
            dict_losses["Contractive"] = compute_contractive_loss(data=data, torus_ae=torus_ae,training_config=training_config)
        # end if
        if training_config["training_mode"]["compute_curvature"]:
            dict_losses["Curvature"] = compute_curvature_loss(data, torus_ae, training_config)
        # end if
    # end if
    return dict_losses



def train(torus_ae, training_config, train_loader, optimizer, 
          epoch=1, batch_idx = 0, dict_loss_arrays={},
          device = "cuda"):
    """
    Train function for the torus autoencoder.

    Parameters:
    - torus_ae: The torus autoencoder model.
    - training_config: Dictionary containing training settings such as optimizer, loss weights, and architecture details.
    - train_loader: DataLoader for training batches.
    - optimizer: Optimizer used for training.
    - epoch: Current epoch number (default is 1).
    - batch_idx: Current batch index, useful when continuing training across multiple calls.
    - dict_loss_arrays: Dictionary to store loss values across batches (default is None, initializes as an empty dictionary).
    - device: Device to use for training ("cuda" or "cpu").

    Returns:
    - batch_idx: Updated batch index after processing all batches.
    - dict_loss_arrays: Dictionary containing loss history for different loss components.
    """

    #  creating a dict for losses shown in progress bar
    dict_loss2print = {}
    if batch_idx == 0:
        dict_loss_arrays = {}
    torus_ae.train()
    torus_ae.to(device)
    print(f"Epoch {epoch}/",training_config["optimizer_settings"]["num_epochs"])
    t = tqdm( train_loader, desc="Train", position=0 )

    # OOD points initialization
    if training_config["training_mode"]["OOD_regime"] == True:
        first_batch,_ = next(iter(train_loader))
        first_batch = first_batch.to(device)
        extreme_curv_points_tensor = torus_ae.encoder_to_lifting(first_batch.view(-1,training_config["architecture"]["input_dim"])[:training_config["OOD_settings"]["N_extr"]]).detach()
        extreme_curv_points_tensor.to(device)
        extreme_curv_value_tensor,_ = ricci_regularization.Sc_g_jacfwd_vmap(extreme_curv_points_tensor, 
                function=torus_ae.decoder_torus,eps=training_config["loss_settings"]["eps"])
    
    for (data, labels) in t:
        data = data.to(device)
        data = data.reshape(-1, training_config["architecture"]["input_dim"])
        optimizer.zero_grad()
        # Forward
        recon_batch, z = torus_ae( data )
        # Computing necessary losses on the current batch
        dict_losses = loss_function( recon_batch, data, z,
                                     torus_ae = torus_ae,
                                     training_config = training_config )
        # appending current losses to loss history
        for key in dict_losses.keys():
            if batch_idx == 0:
                dict_loss_arrays[key] = []
            # losses to keep in memory:
            dict_loss_arrays[key].append(dict_losses[key].item())
            # losses to show on progress bar: 
            dict_loss2print[key] = f"{dict_losses[key].item():.4f}"
            # moving average (per epoch)
            #dict_loss2print[key] = f"{np.array(dict_loss_arrays[key])[-batches_per_epoch:].mean():.4f}"
        # end for 
        loss = training_config["loss_settings"]["lambda_recon"] * dict_losses["MSE"] 
        loss += training_config["loss_settings"]["lambda_unif"] * dict_losses["Equidistribution"] 

        if training_config["training_mode"]["compute_contractive_loss"] == True:
            # adding the contractive loss on the currenct batch to the loss function
            loss += training_config["loss_settings"]["lambda_contractive"] * dict_losses["Contractive"]
            
        if (training_config["training_mode"]["compute_curvature"] == True): 
            # adding the curvature loss on the currenct batch to the loss function
            loss += training_config["loss_settings"]["lambda_curv"] * dict_losses["Curvature"]
            
        # OOD regime (optional)
        if training_config["training_mode"]["OOD_regime"] == True:
            OOD_params_dict = training_config["OOD_settings"]
            extreme_curv_points_tensor, extreme_curv_value_tensor = ricci_regularization.find_extreme_curvature_points(
                data_batch = data,
                extreme_curv_points_tensor = extreme_curv_points_tensor,
                extreme_curv_value_tensor = extreme_curv_value_tensor,
                batch_idx = batch_idx,
                encoder=torus_ae.encoder_to_lifting,decoder = torus_ae.decoder_torus,
                OOD_params_dict = training_config["OOD_settings"],
                output_dim=training_config["architecture"]["input_dim"])
            
            if (batch_idx % OOD_params_dict["T_ood"] == 0) & (batch_idx >= training_config["OOD_settings"]["start_ood"]):
                OOD_curvature_loss = ricci_regularization.OODTools.curv_loss_on_OOD_samples(
                    extreme_curv_points_tensor = extreme_curv_points_tensor,
                    decoder = torus_ae.decoder_torus,
                    OOD_params_dict = training_config["OOD_settings"],
                    latent_space_dim = training_config["architecture"]["latent_dim"] )
                if training_config["training_mode"]["diagnostic_mode"] == True:
                    print("Curvature functional at OOD points", OOD_curvature_loss)
                loss = training_config["OOD_settings"]["OOD_w"] * OOD_curvature_loss
            #end if
        #end if
        
        # Backpropagate
        loss.backward()
        optimizer.step()

        # Progress bar plotting
        t.set_postfix(dict_loss2print)
        # Switching batch index
        batch_idx += 1
    #end for
    return batch_idx, dict_loss_arrays

def test(torus_ae, test_loader, training_config, dict_loss_arrays = {}, batch_idx = 0, device ="cuda"):
    """
    Test function for evaluating the torus autoencoder model.

    Parameters:
    - torus_ae: The trained torus autoencoder model.
    - test_loader: DataLoader for the test dataset.
    - training_config: Dictionary containing configuration settings, such as architecture details.
    - dict_loss_arrays: Dictionary to store loss values for each batch (default is None, initializes as empty).
    - batch_idx: The current batch index (default is 0).
    - device: The device to run the model on ("cuda" for GPU or "cpu" for CPU, default is "cuda").

    Returns:
    - batch_idx: The updated batch index after processing all batches.
    - dict_loss_arrays: Dictionary containing loss history for different loss components.
    """
    dict_loss2print = {}
    torus_ae.to(device)
    t = tqdm( test_loader, desc="Test", position=1 )
    for data, _ in t:
        data = data.to(device)
        data = data.reshape(-1, training_config["architecture"]["input_dim"])
        recon_batch, z = torus_ae(data)
        dict_losses = loss_function(recon_batch, data, z, 
                torus_ae = torus_ae, training_config=training_config)
        # appending current losses to loss history    
        for key in dict_losses.keys():
            if batch_idx == 0:
                dict_loss_arrays[key] = []
            dict_loss_arrays[key].append(dict_losses[key].item())
            # mean losses to print
            dict_loss2print[key] = f"{dict_losses[key]:.4f}"
            # dict_loss2print[key] = f"{np.array(dict_loss_arrays[key]).mean():.4f}"
        t.set_postfix(dict_loss2print)
        # switch batch index
        batch_idx+=1
    #end for
    return batch_idx, dict_loss_arrays