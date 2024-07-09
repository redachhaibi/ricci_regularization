import torch
def uniform_loss(z, latent_dim, num_moments):
    z_sin = z[:, 0:latent_dim]
    z_cos = z[:, latent_dim:2*latent_dim]
    mode1 = torch.mean( z, dim = 0)
    mode1 = torch.sum( mode1*mode1 )
    mode2_1 = torch.mean( 2*z_cos*z_cos-1, dim = 0)
    mode2_1 = torch.sum( mode2_1*mode2_1)
    mode2_2 = torch.mean( 2*z_sin*z_cos, dim = 0)
    mode2_2 = torch.sum( mode2_2*mode2_2 )
    mode2 = mode2_1 + mode2_2
    unif_loss = mode1 + mode2
    if num_moments > 2:
        mode3_1 = torch.mean( 4*z_cos**3-3*z_cos, dim = 0)
        mode3_1 = torch.sum( mode3_1*mode3_1)
        mode3_2 = torch.mean( z_sin*(8*z_cos**3-4*z_cos), dim = 0)
        mode3_2 = torch.sum( mode3_2*mode3_2 )
        mode3 = mode3_1 + mode3_2
        unif_loss += mode3
    if num_moments > 3:
        mode4_1 = torch.mean( 8*z_cos**4-8*z_cos**2+1, dim = 0)
        mode4_1 = torch.sum( mode4_1*mode4_1)
        mode4_2 = torch.mean( z_sin*(16*z_cos**4-12*z_cos**2+1), dim = 0)
        mode4_2 = torch.sum( mode4_2*mode4_2 )
        mode4 = mode4_1 + mode4_2
        unif_loss += mode4
    return unif_loss