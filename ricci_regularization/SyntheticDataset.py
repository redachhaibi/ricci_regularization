import torch
import torchvision
import random

torch.manual_seed(0)

# Here is an example of parameters
# D = 784 #dimension
# k = 3 # num of 2d planes in dim D
# n = 6*(10**3) # num of points in each plane

# old style 
def generate_dataset(D, k, n, shift_class=0, intercl_var = 0, var_class = 1):
    phi = [] #list of k ontonormal bases in k planes
    for j in range(k):
        # creating random planes
        rand_vectors = torch.rand(D, 2)
        q, r = torch.qr(rand_vectors)
        phi.append(q)
    #phi

    #creating samples from normal distributions via torch distributions
    data = []
    for i in range(k):
        m = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2) + shift_class*(i+1), var_class*torch.eye(2))
        #m = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), torch.eye(2))
        samples = m.sample(sample_shape=(n,)).T
        #samples = normalize(samples, p = 1, dim = 0)
        #data.append(normalize(torch.matmul(phi[i], samples)))
        samples_transformed = torch.matmul(phi[i], samples) + torch.randn(D,1) * intercl_var
        data.append(samples_transformed)
        #data.append(torch.matmul(phi[i], samples))
    data_tensor = torch.cat(data, dim=1)

    data_tensor = data_tensor.T
    data_tensor = data_tensor.reshape(k*n, 1, 28, 28)

    labels_list = []
    for i in range(k):
        labels_list.append(i*(torch.ones(n)))
    labels = torch.cat(labels_list)

    train_dataset = torch.utils.data.TensorDataset(data_tensor,labels)

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    train_dataset.transform = train_transform

    return train_dataset


# new style
# definition using class

class SyntheticDataset:
    def __init__(self, k = 3, n = 6000, d=2, D=784, shift_class=0, var_class = 1, intercl_var = 0):
        phi = [] #list of k ontonormal bases in k planes
        for j in range(k):
            # creating random planes
            rand_vectors = torch.rand(D, d)
            q, r = torch.qr(rand_vectors)
            phi.append(q)
        #phi

        #creating samples from normal distributions via torch distributions
        data = []
        shifts = []
        for i in range(k):
            m = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(d) + shift_class*(i+1), var_class*torch.eye(d))
            #m = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), torch.eye(2))
            samples = m.sample(sample_shape=(n,)).T
            
            shift = torch.randn(D,1) * intercl_var
            shifts.append(shift) 

            samples_transformed = torch.matmul(phi[i], samples) + shift
            data.append(samples_transformed)
            #data.append(torch.matmul(phi[i], samples))
        data_tensor = torch.cat(data, dim=1)

        data_tensor = data_tensor.T
        data_tensor = data_tensor.reshape(k*n, 1, D)

        labels_list = []
        for i in range(k):
            labels_list.append(i*(torch.ones(n)))
        labels = torch.cat(labels_list)

        train_dataset = torch.utils.data.TensorDataset(data_tensor,labels)

        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

        train_dataset.transform = train_transform

        self.create = train_dataset
        self.rotations = phi
        self.shifts = shifts