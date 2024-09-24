import torch
import math
import ricci_regularization

# Grid covering the whole latent space
def make_grid(numsteps, xlim_left = -torch.pi, xlim_right = torch.pi,
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
    tgrid = make_grid(numsteps)

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