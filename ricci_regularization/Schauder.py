# imported from wavelets.py
from tqdm                  import tqdm
import torch, math
import numpy as np

# Mathematical source for both Haar and Schauder(-Faber) bases: https://en.wikipedia.org/wiki/Haar_wavelet
# Numpy implementation of Haar wavelet generation on [0,1]
def haar_wavelet(time_grid):
    index_set1 = np.where( (0   <= time_grid) & (time_grid < 0.5) )
    index_set2 = np.where( (0.5 <= time_grid) & (time_grid < 1  ) )
    wavelet = np.zeros_like( time_grid )
    wavelet[ index_set1 ] = 1
    wavelet[ index_set2 ] = -1
    return wavelet

def haar_basis_element(n, k, time_grid):
    return (2.0**(0.5*n)) * haar_wavelet( (2**n)*time_grid- k)


# Returns a total of 2**n_max basis functions corresponding to
#   the Haar wavelets (n,k) corresponding to
#   n=0, ..., n_max-1 and 
#   k=0, ..., 2**n - 1
def haar_basis( n_max, time_grid):
    basis = []
    # Constant function
    basis.append( np.ones_like(time_grid) )
    # The next ones (n>0)
    for n in range(n_max):
        basis = basis + [ haar_basis_element(n,k, time_grid) for k in range(2**n)]
    return basis
    

# Numpy implementation of Schauder basis generation on [0,1]

# Schauder elements are integrals of the Haar elements
"""
def schauder_basis_element(n, k, time_grid):
    haar_element =  haar_basis_element(n, k, time_grid)
    element = np.zeros_like( haar_element )
    element[1:] = (2**(1+0.5*n)) * 0.5* ( haar_element[:-1] + haar_element[1:] )* (time_grid[1:] - time_grid[:-1])
    return element.cumsum()
"""

# Fixed version of previous functon
def schauder_basis_element(n, k, time_grid):
    haar_element =  haar_basis_element(n, k, time_grid)
    element = np.zeros_like( haar_element )
    #for k=0 the following line yields a numerical error of size (2**(1+0.5*n)) * 0.5* (step of time_grid) 
    #element[1:] = (2**(1+0.5*n)) * 0.5* ( haar_element[:-1] + haar_element[1:] )* (time_grid[1:] - time_grid[:-1])
    #return element.cumsum()
    element[1:] = (2**(1+0.5*n)) * 0.5* ( haar_element[:-1] + haar_element[1:] )* (time_grid[1:] - time_grid[:-1])
    element = element.cumsum()
    # a fix:
    step_count = len(time_grid)
    start_index = math.floor(((k + 1) / 2**n) * step_count - 1)
    mask = np.arange(step_count) >= start_index
    # Apply the mask. Make all elements with indices >= start_index equal to zero.
    element[mask] = 0
    return element

def schauder_basis( n_max, time_grid):
    basis = []
    # Constant function
    basis.append( np.ones_like(time_grid) )
    # Identity map
    basis.append( np.copy(time_grid) )
    # The next ones (n>0)
    for n in range(n_max):
        basis = basis + [ schauder_basis_element(n,k, time_grid) for k in range(2**n)]
    return basis


class NumericalGeodesics():
    """docstring for Numerical"""
    def __init__(self, n_max, step_count):
        super(NumericalGeodesics, self).__init__()

        self.n_max      = n_max
        self.step_count = step_count
        self.time_grid  = torch.linspace( 0, 1, step_count)
        
        # Precompute Schauder bases
        self.schauder_bases = {
            "zero_boundary"      : None, 
            "shooting" : None
        }

        # Mode1 = Interpolation mode = Zero boundary functions
        basis = schauder_basis( self.n_max, self.time_grid.numpy() )
        basis = torch.t( torch.tensor( basis ) )
        basis = basis[:,1:] # Throw away the first basis vector, the constant function
        basis = basis[:,1:] # Throw away the second basis vector, the linear function
        N_max = 2**n_max-1  # Number of basis elements in Schauder basis is N_max = 2**n_max + 1 ; After throwing the two first basis vectors N_max = 2**n_max-1
        self.schauder_bases["zero_boundary"] = { "basis": basis, "N_max": N_max}

        # Mode2 = Shooting mode = Free endpoint
        basis = schauder_basis( self.n_max, self.time_grid.numpy() )
        basis = torch.t( torch.tensor( basis ) )
        basis = basis[:,1:] # Throw away the first basis vector, the constant function
        N_max = 2**n_max    # Number of basis elements in Schauder basis is N_max = 2**n_max + 1 ; After throwing the first basis vectors N_max = 2**n_max
        self.schauder_bases["shooting"] = { "basis": basis, "N_max": N_max}


    def computeGeodesicInterpolation(self, generator, m1, m2, epochs, optimizer_info, display_info) :
        """
            generator     : Function taking latent variables and generating a sample. Its gradient encodes the metric tensor.
            m1, m2        : Initial and destination point
            optimizer_info: dict containing
                            "name": Name of torch optimizer
                            "args": Learning rate, Momentum and Nesterov acceleration
            display_info  : String to add to progress bar, useful for identifying running task
        """
        N_max = self.schauder_bases["zero_boundary"]["N_max"]
        basis = self.schauder_bases["zero_boundary"]["basis"]
        # Dimension
        dim   = m1.shape[0]
        # parameters = Coefficients of (base+fiber) curve in Schauder basis
        parameters   = torch.zeros( (N_max, dim) , requires_grad=True)
        # Define linear interpolating curve == naive geodesic
        linear_curve = torch.ones( self.step_count, 1)*m1 + self.time_grid.view( self.step_count, 1)*(m2-m1)

        # Initialization
        curve  = linear_curve
        # Optimizer
        energy = 0
        optimizer = getattr( torch.optim, optimizer_info["name"] )
        optimizer = optimizer( [{ 'params': parameters }], **optimizer_info["args"] )
        # Loop
        with tqdm( range(epochs) ) as t:
            for i in t:
                # Compute curve of latent variables with suboptimal parameters
                curve             = linear_curve + torch.mm( basis, parameters )
                # Output
                #generated_images  = generator( encoding = curve )
                generated_images  = generator( curve )
                # Finite difference computation of energy
                energy = (generated_images[1:,:]-generated_images[:-1,:]).pow(2).sum()
                # Optimize
                optimizer.zero_grad()
                energy.backward(retain_graph=True)
                grad_norm = parameters.grad.norm()
                t.set_description( "%s. Energy %f, Grad %f"%(display_info, energy, grad_norm) )
                optimizer.step()
            # End for
        # End with
        #
        # Recompute curve with optimal parameters
        geodesic_curve = linear_curve + torch.mm( basis, parameters )
        return linear_curve.detach().numpy(), geodesic_curve.detach().numpy()