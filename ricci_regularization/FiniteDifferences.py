import torch
import math
import ricci_regularization

# f.d. on batches of rhombus minigrids

def build_mini_grid_batch(centers: torch.Tensor, h: float) -> torch.Tensor:
    """
    Builds a batch of mini-grids centered at the given batch of points.
    
    Args:
        centers (torch.Tensor): A 2D tensor with shape (N, 2) representing N centers.
        grid_size (int): The size of the mini-grid (grid_size x grid_size).
        h (float): The step size for the grid.
        
    Returns:
        torch.Tensor: A batch of mini-grids of shape (N, grid_size * grid_size, 2).
    """
    dtype = centers.dtype
    offset = torch.arange(-3, 4, dtype=dtype) * h  # Relative offsets from the center (-3h, -2h, ..., 3h)
    grid_x, grid_y = torch.meshgrid(offset, offset, indexing='ij')  # 7x7 grid for x and y

    # Stack the coordinates (x, y) together and add to the center
    mini_grid = torch.stack([grid_x, grid_y], dim=-1).float()  # Shape: (7, 7, 2)
    mini_grid = mini_grid.reshape(49,2) # Shape: (49, 2)
    # Expand dimensions to match the number of centers
    mini_grid = mini_grid.unsqueeze(0)  # shape: (1, grid_size * grid_size, 2)

    device = centers.device
    mini_grid = mini_grid.to(device)
    # Broadcast the centers to create the batch
    centers = centers.unsqueeze(1)  # shape: (N, 1, 2)

    # Add the centers to the mini-grid points
    batch_minigrids = mini_grid + centers  # shape: (N, grid_size * grid_size, 2)

    d = centers.shape[-1]
    batch_size = centers.shape[0]
    batch_minigrids = batch_minigrids.reshape(batch_size, 7, 7, d) # shape batch_size * 7 * 7 * d
    return batch_minigrids

def indices(minigrid_size=7):
    # Step 0: Initialize a tensor of size minigrid_size x minigrid_size filled with -1.
    # This will be the grid where we'll fill in the rhombus shape.
    rhombus_tensor = -torch.ones((minigrid_size, minigrid_size), dtype=torch.int)

    # Step 1: Calculate Manhattan distances from the center to define a rhombus.
    # Determine the center of the grid.
    center = minigrid_size // 2
    
    # Create grid coordinates for x and y using meshgrid.
    grid_x, grid_y = torch.meshgrid(torch.arange(minigrid_size), torch.arange(minigrid_size), indexing="ij")

    # Compute Manhattan distance from the center point (center, center) in the grid.
    manhattan_distance = torch.abs(grid_x - center) + torch.abs(grid_y - center)

    # Create a mask that defines the rhombus by checking if the Manhattan distance 
    # is less than or equal to the center distance. This mask selects the rhombus region.
    mask = manhattan_distance <= center

    # Step 2: Fill the masked rhombus area with sequential indices.
    # Compute the number of points in the rhombus (approximately half the grid size).
    num_rhombus_points = (minigrid_size * minigrid_size) // 2 + 1
    
    # Create a tensor of sequential indices to fill the rhombus (0, 1, 2, ..., num_rhombus_points-1).
    indices = torch.arange(num_rhombus_points, dtype=torch.int)
    
    # Fill the rhombus-shaped region in the tensor with these indices.
    rhombus_tensor[mask] = indices

    # Step 3: Compute Manhattan distances for steps in x and y directions relative to the center.

    # Compute Manhattan distance from the point (center + 1, center) - this shifts right on the grid.
    manhattan_distance_x_next = torch.abs(grid_x - (center + 1)) + torch.abs(grid_y - center)
    mask_x_next = manhattan_distance_x_next <= center - 1  # Define mask for next step in x direction.

    # Compute Manhattan distance from the point (center - 1, center) - this shifts left on the grid.
    manhattan_distance_x_prev = torch.abs(grid_x - (center - 1)) + torch.abs(grid_y - center)
    mask_x_prev = manhattan_distance_x_prev <= center - 1  # Define mask for previous step in x direction.

    # Compute Manhattan distance from the point (center, center + 1) - this shifts up on the grid.
    manhattan_distance_y_next = torch.abs(grid_x - center) + torch.abs(grid_y - (center + 1))
    mask_y_next = manhattan_distance_y_next <= center - 1  # Define mask for next step in y direction.

    # Compute Manhattan distance from the point (center, center - 1) - this shifts down on the grid.
    manhattan_distance_y_prev = torch.abs(grid_x - center) + torch.abs(grid_y - (center - 1))
    mask_y_prev = manhattan_distance_y_prev <= center - 1  # Define mask for previous step in y direction.

    # Compute Manhattan distance from the point (center, center) - this is the central position.
    manhattan_distance_central = torch.abs(grid_x - center) + torch.abs(grid_y - center)
    mask_central = manhattan_distance_central <= center - 1  # Define mask for the center position.

    # Step 4: Extract the rhombus indices for the different shifts and central position.
    
    # Extract indices for rhombus points shifted by +1 in the x direction.
    indices_x_next = rhombus_tensor[mask_x_next]
    
    # Extract indices for rhombus points shifted by -1 in the x direction.
    indices_x_prev = rhombus_tensor[mask_x_prev]
    
    # Extract indices for rhombus points shifted by +1 in the y direction.
    indices_y_next = rhombus_tensor[mask_y_next]
    
    # Extract indices for rhombus points shifted by -1 in the y direction.
    indices_y_prev = rhombus_tensor[mask_y_prev]
    
    # Extract indices for the central rhombus points.
    indices_central = rhombus_tensor[mask_central]

    # Return the mask and the indices for the next, previous, and central steps.
    return mask, indices_x_next, indices_x_prev, indices_y_next, indices_y_prev, indices_central


def Sc_g_fd_batch_minigrids_rhombus (centers, function, h=0.01, eps = 0.0):
    # input: 
    # centers: torch.tensor are the batch of points in dimension d
    # they are centers of rhombus minigrids inside 7 x 7 square grids 
    # function: is the function through which we pullbac the euclidean metric, 
    # typically a decoder
    # h: float the step of the grid
    # eps: float regularization parameter for computaion of the inverse of metric
    
    d = centers.shape[-1]
    batch_size = centers.shape[0]
    # create a batch of minigrids with given centers and step h
    batch_minigrids = build_mini_grid_batch(centers, h) # shape batch_size * 7 * 7 * d

    # Create the rhombus mask and indices for differentiation 
    mask, indices_x_next, indices_x_prev, indices_y_next, indices_y_prev,_ = indices( minigrid_size = 7)
    # expand the mask to shape [batch_size, 7, 7, d] 
    batch_mask = mask.unsqueeze(0).unsqueeze(-1).expand(batch_size, 7, 7, d)

    # Extract rhombus values for the batch (use the mask on the batch of minigrids)
    rhombus_minigrids_batch = batch_minigrids[batch_mask].view(batch_size, -1, d)

    # Evaluate the decoder psi only on the rhombus
    psi = function( rhombus_minigrids_batch ) # shape batch_size * 25 * D

    # compute dpsi
    dpsi_dx_fast = ( psi[:, indices_x_next] - psi[:, indices_x_prev] ) / ( 2 * h ) # shape batch_size * 13 * D
    dpsi_dy_fast = ( psi[:, indices_y_next] - psi[:, indices_y_prev] ) / ( 2 * h ) # shape batch_size * 13 * D
    dpsi = torch.cat(( dpsi_dx_fast.unsqueeze(-1), dpsi_dy_fast.unsqueeze(-1) ), -1) # shape batch_size * 13 * D * d
    
    # compute metric
    # b is batch_size, g,h are coordinates on the minigrid, D is output of psi dimension, i,j are local coordinates
    g = torch.einsum('bgDi,bgDj->bgij', dpsi,dpsi) # shape batch_size * 13 * d * d

    # compute metric derivatives

    # Get new indices for differentiation 
    _, indices_x_next, indices_x_prev, indices_y_next, indices_y_prev, indices_central  = indices(minigrid_size = 5)

    dg_dx = ( g[:, indices_x_next] - g[:, indices_x_prev] ) / ( 2 * h ) # shape batch_size * 5 * D
    dg_dy = ( g[:, indices_y_next] - g[:, indices_y_prev] ) / ( 2 * h ) # shape batch_size * 5 * D
    dg = torch.cat((dg_dx.unsqueeze(-1), dg_dy.unsqueeze(-1)), dim = -1) # shape batch_size * 5 * d * d * d
    del dg_dx, dg_dy

    # compute inverse of g
    device = g.device
    
    # cutting the shape of g to compute g_inv
    g = g[:, indices_central] # new shape: batch_size * 5 * d * d
    g_inv = torch.inverse(g + eps*torch.eye(d,device=device)) # shape batch_size * 5 * d * d
    
    # leaving only values of g at centers
    g = g[:, 2]

    # compute Christoffel symbols
    # b is batch_size, g,h are coordinates on the minigrid, i, m, k, l are local coordinates
    Christoffel = 0.5*(torch.einsum('bgim,bgmkl->bgikl',g_inv,dg)+
              torch.einsum('bgim,bgmlk->bgikl',g_inv,dg)-
              torch.einsum('bgim,bgklm->bgikl',g_inv,dg)
              ) # shape batch_size * 5 * d * d * d
    del dg

    # compute Christoffel symbols' derivatives
    # Get new indices for differentiation 
    _, indices_x_next, indices_x_prev, indices_y_next, indices_y_prev, indices_central  = indices(minigrid_size = 3)

    dChristoffel_dx = ( Christoffel[:, indices_x_next] - Christoffel[:, indices_x_prev] ) / ( 2 * h ) # shape batch_size * 1 * D
    dChristoffel_dy = ( Christoffel[:, indices_y_next] - Christoffel[:, indices_y_prev] ) / ( 2 * h ) # shape batch_size * 1 * D

    dChristoffel = torch.cat((dChristoffel_dx.unsqueeze(-1),
                              dChristoffel_dy.unsqueeze(-1)), dim = -1) # shape batch_size * 1 * d * d * d * d
    del dChristoffel_dx, dChristoffel_dy
    # squeezing since we only have values at centers of minigrids (one point)
    dChristoffel = dChristoffel.squeeze() # shape batch_size * d * d * d * d
    
    # compute Riemann tensor

    # cutting the shape of Christoffels to compute Riemann
    assert indices_central[0] == 2 # the central index should be 2 indeed
    Christoffel = Christoffel[:, indices_central].squeeze() # new shape: batch_size * d * d * d
    # b is batch_size i, j, k, l, p are local coordinates
    Riemann = torch.einsum("biljk->bijkl",dChristoffel) - torch.einsum("bikjl->bijkl",dChristoffel)
    Riemann += torch.einsum("bikp,bplj->bijkl", Christoffel, Christoffel) - torch.einsum("bilp,bpkj->bijkl", Christoffel, Christoffel)
    # Riemann shape: batch_size * d * d * d
    del dChristoffel, Christoffel

    # compute Ricci
    # b is batch_size c, s, r are local coordinates
    Ricci = torch.einsum("bcscr->bsr",Riemann)
    del Riemann

    # compute scalar curvature
    # cutting the shape of the inverse of the metric. Only needed at one central point:
    g_inv = g_inv[:,indices_central].squeeze() # shape batch_size * d * d
    # b is batch_size s, r are local coordinates
    Sc = torch.einsum('bsr,bsr->b',g_inv,Ricci)
    del Ricci, g_inv
    return Sc, g

def curvature_loss(points, function, h, eps):
    # returns the curvature loss 
    R, g = Sc_g_fd_batch_minigrids_rhombus(centers = points, function = function, h = h, eps=eps)
    return ( ( R**2 ) * torch.sqrt( torch.det(g) ) ).mean()


# Uniform grids ( suboptimal with recurrence)
# Grid covering the whole latent space
def make_grid_fd(numsteps, xlim_left = -torch.pi, xlim_right = torch.pi,
              ylim_bottom = -torch.pi, ylim_top = torch.pi):
    xs = torch.linspace(xlim_left, xlim_right, steps = numsteps)
    ys = torch.linspace(ylim_bottom, ylim_top, steps = numsteps)

    tgrid = torch.cartesian_prod(ys, xs)
    tgrid = tgrid.roll(1,1)
    return tgrid


#metric on a grid which is the pull-back of a Euclidean metric by 'function'
def metric_fd_grid(grid, function):
    numsteps = int(math.sqrt(grid.shape[0]))
    
    hx = float(abs((grid[numsteps**2 - 1] - grid[0])[0]))/(numsteps - 1)
    hy = float(abs((grid[numsteps**2 - 1] - grid[0])[1]))/(numsteps - 1)
    
    latent = grid
    #latent = latent.to(device)
    psi = function(latent)
    psi_next_x =  psi.roll(-1,0)
    psi_prev_x =  psi.roll(1,0)
    psi_next_y =  psi.roll(-numsteps,0)
    psi_prev_y =  psi.roll(numsteps,0)
    
    dpsidx = (psi_next_x - psi_prev_x)/(2*hx)
    dpsidy = (psi_next_y - psi_prev_y)/(2*hy)

    E = torch.func.vmap(torch.dot)(dpsidx, dpsidx)
    F = torch.func.vmap(torch.dot)(dpsidx, dpsidy)
    G = torch.func.vmap(torch.dot)(dpsidy, dpsidy)
    
    metric = torch.cat((E.unsqueeze(0), F.unsqueeze(0), F.unsqueeze(0), G.unsqueeze(0)),0)
    metric = metric.view(4, numsteps*numsteps)
    metric = metric.transpose(0, 1)
    metric = metric.view(numsteps*numsteps, 2, 2)
    return metric

def diff_by_x(tensor_on_grid, h):
    tensor_next_x =  tensor_on_grid.roll(-1,0)
    tensor_prev_x =  tensor_on_grid.roll(1,0)
    tensor_dx = (tensor_next_x - tensor_prev_x)/(2*h)
    return tensor_dx

def diff_by_y(tensor, numsteps, h):
    psi = tensor
    psi_next_y =  psi.roll(-numsteps,0)
    psi_prev_y =  psi.roll(numsteps,0)
    dpsidy = (psi_next_y - psi_prev_y)/(2*h)
    return dpsidy

def metric_der_fd_grid(grid, function):
    h = (grid[1] - grid[0]).norm()
    numsteps = int(math.sqrt(grid.shape[0]))
    metric = ricci_regularization.metric_fd_grid(grid, 
                    function = function)
    dg_dx_fd = diff_by_x(metric, h=h)
    dg_dy_fd = diff_by_y(metric, numsteps=numsteps, h = h)
    dg = torch.cat((dg_dx_fd.unsqueeze(-1), dg_dy_fd.unsqueeze(-1)), dim = -1)
    return dg

def metric_inv_fd(u, function, eps=0.0):
    g = metric_fd_grid(u,function)
    d = g.shape[-1]
    device = g.device
    g_inv = torch.inverse(g + eps*torch.eye(d,device=device))
    return g_inv

#metric_inv_jacfd_vmap = torch.func.vmap(metric_inv_fd)

def Ch_fd (u, function, eps = 0.0):
    g_inv = metric_inv_fd(u,function,eps=eps)
    dg = metric_der_fd_grid(u,function)
    Ch = 0.5*(torch.einsum('bim,bmkl->bikl',g_inv,dg)+
              torch.einsum('bim,bmlk->bikl',g_inv,dg)-
              torch.einsum('bim,bklm->bikl',g_inv,dg)
              )
    return Ch
#Ch_fd_vmap = torch.func.vmap(Ch_fd)

def Ch_der_fd (grid, function, eps=0.0):
    h = (grid[1] - grid[0]).norm() # =size/numsteps
    numsteps = int(math.sqrt(grid.shape[0]))
    Ch = Ch_fd(grid, function=function, eps=eps)
    dChdx = diff_by_x(Ch, h)
    dChdy = diff_by_y(Ch, numsteps=numsteps, h = h)
    dCh = torch.cat((dChdx.unsqueeze(-1), dChdy.unsqueeze(-1)), dim = -1)
    return dCh


# Riemann curvature tensor (3,1)
def Riem_fd(u, function,eps=0.0):
    Ch = Ch_fd(u, function, eps=eps)
    Ch_der = Ch_der_fd(u, function, eps=eps)

    Riem = torch.einsum("biljk->bijkl",Ch_der) - torch.einsum("bikjl->bijkl",Ch_der)
    Riem += torch.einsum("bikp,bplj->bijkl", Ch, Ch) - torch.einsum("bilp,bpkj->bijkl", Ch, Ch)
    return Riem

def Ric_fd(u, function, eps=0.0):
    Riemann = Riem_fd(u, function, eps=eps)
    Ric = torch.einsum("bcscr->bsr",Riemann)
    return Ric

def Sc_fd (u, function, eps = 0.0):
    Ricci = Ric_fd(u, function=function,eps=eps)
    metric_inv = metric_inv_fd(u,function=function, eps=eps)
    Sc = torch.einsum('bsr,bsr->b',metric_inv,Ricci)
    return Sc

def error_fd_jacfwd_on_grid(tensor_fd, tensor_jacfwd, cut = 1):
    
    numsteps = int( math.sqrt(tensor_jacfwd.shape[0]) )
    #finite differences
    tensor_fd = tensor_fd.reshape(numsteps, numsteps, -1)
    tensor_fd_on_grid_no_borders = tensor_fd[cut:-cut,cut:-cut]

    # Jacfwd
    tensor_jacfwd = tensor_jacfwd.reshape(numsteps, numsteps, - 1)
    tensor_jacfwd_on_grid_no_borders = tensor_jacfwd[cut:-cut,cut:-cut]

    error = torch.functional.F.mse_loss(tensor_fd_on_grid_no_borders, tensor_jacfwd_on_grid_no_borders)
    return error

# old style
# compute the grid of metric
# 'cut' defines number of grid layers cut from the border 
def compute_error_metric_on_grid(numsteps, function, cut = 1):
    tgrid = make_grid_fd(numsteps)

    #finite differences
    with torch.no_grad():
        metric = metric_fd_grid(tgrid, function)
    metric = metric.reshape(numsteps, numsteps, - 1)
    metric_fd_on_grid_no_borders = metric[cut:-cut,cut:-cut]

    # Jacfwd
    metric_jacfwd_on_grid = ricci_regularization.metric_jacfwd_vmap(tgrid, function = function)
    metric_jacfwd_on_grid_no_borders = metric_jacfwd_on_grid.reshape(numsteps, numsteps, - 1)[1:-1,1:-1]

    #error = (metric_fd_on_grid_no_borders - metric_jacfwd_on_grid_no_borders).norm()**2 /(4 * numsteps**2)
    error = torch.functional.F.mse_loss(metric_jacfwd_on_grid_no_borders, metric_fd_on_grid_no_borders)
    #print("L2 norm of error:", error)
    return error