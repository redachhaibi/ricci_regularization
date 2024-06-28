import json
from sklearn import datasets
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms
import sklearn

def get_dataloaders_tuned_nn(Path_experiment_json:str, additional_path = ''):
    """
    Loads dataset, splits it into training and testing sets, creates DataLoaders, sets up a VAE architecture, 
    loads pre-trained weights, and returns a dictionary containing the DataLoaders, the tuned VAE, and the configuration.

    Parameters:
    Path_experiment_json (str): Path to the JSON configuration file.
    additional_path (str, optional): Additional path to the pre-trained weights.

    Returns:
    dict: A dictionary with the training DataLoader, testing DataLoader, the tuned AE model, and the configuration.
    """

    with open(Path_experiment_json) as json_file:
        json_config = json.load(json_file)

    Path_experiments = json_config["Path_experiments"]
    experiment_name = json_config["experiment_name"]
    experiment_number = json_config["experiment_number"]
    Path_pictures = json_config["Path_pictures"]
    set_name    = json_config["dataset"]["name"]
    split_ratio = json_config["optimization_parameters"]["split_ratio"]
    batch_size  = json_config["optimization_parameters"]["batch_size"]
    datasets_root = '../../datasets/'
    # Dataset uploading 
    # import sys
    # sys.path.append('../') # have to go 1 level up
    import ricci_regularization
    if set_name == "MNIST":
        #MNIST_SIZE = 28
        # MNIST Dataset
        D = 784
        train_dataset = datasets.MNIST(root=datasets_root, train=True, transform=transforms.ToTensor(), download=True)
        test_dataset  = datasets.MNIST(root=datasets_root, train=False, transform=transforms.ToTensor(), download=False)
    elif set_name == "MNIST01":
        D = 784
        full_mnist_dataset = datasets.MNIST(root=datasets_root, train=True, transform=transforms.ToTensor(), download=True)
        test_dataset  = datasets.MNIST(root=datasets_root, train=False, transform=transforms.ToTensor(), download=False)
        #indices01 = torch.where((full_mnist_dataset.targets == 0) | (full_mnist_dataset.targets == 1))[0]
        mask = (full_mnist_dataset.targets == -1) 
        selected_labels = json_config["dataset"]["selected_labels"]
        for label in selected_labels:
            mask = mask | (full_mnist_dataset.targets == label)
        indices01 = torch.where(mask)[0]
        train_dataset = Subset(full_mnist_dataset, indices01) # MNIST only with 0,1 indices

    elif set_name == "Synthetic":
        k = json_config["dataset"]["parameters"]["k"]
        n = json_config["dataset"]["parameters"]["n"]
        d = json_config["architecture"]["latent_dim"]
        D = json_config["architecture"]["input_dim"]
        shift_class = json_config["dataset"]["parameters"]["shift_class"]
        intercl_var = json_config["dataset"]["parameters"]["intercl_var"]
        var_class = json_config["dataset"]["parameters"]["var_class"]
        # Generate dataset
        # via classes
        torch.manual_seed(0) # reproducibility
        my_dataset = ricci_regularization.SyntheticDataset(k=k,n=n,d=d,D=D,
                                            shift_class=shift_class, intercl_var=intercl_var, var_class=var_class)

        train_dataset = my_dataset.create
    elif set_name == "Swissroll":
        sr_noise = json_config["dataset"]["parameters"]["sr_noise"]
        sr_numpoints = json_config["dataset"]["parameters"]["sr_numpoints"]

        D = 3
        train_dataset =  sklearn.datasets.make_swiss_roll(n_samples=sr_numpoints, noise=sr_noise)
        sr_points = torch.from_numpy(train_dataset[0]).to(torch.float32)
        #sr_points = torch.cat((sr_points,torch.zeros(sr_numpoints,D-3)),dim=1)
        sr_colors = torch.from_numpy(train_dataset[1]).to(torch.float32)
        from torch.utils.data import TensorDataset
        train_dataset = TensorDataset(sr_points,sr_colors)

    m = len(train_dataset)
    train_data, test_data = torch.utils.data.random_split(train_dataset, [m-int(m*split_ratio), int(m*split_ratio)])

    test_loader  = torch.utils.data.DataLoader(test_data , batch_size=batch_size)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # VAE structure
    latent_dim = json_config["architecture"]["latent_dim"]
    input_dim  = json_config["architecture"]["input_dim"]
    try:
        architecture_type = json_config["architecture"]["name"]
    except KeyError:
        architecture_type = "TorusAE"
    if architecture_type== "TorusAE":
        torus_ae   = ricci_regularization.Architectures.TorusAE(x_dim=input_dim, h_dim1= 512, h_dim2=256, z_dim=latent_dim)
    elif architecture_type =="TorusConvAE":
        torus_ae   = ricci_regularization.Architectures.TorusConvAE(x_dim=input_dim, h_dim1= 512, h_dim2=256, z_dim=latent_dim,pixels=28)
    if torch.cuda.is_available():
        torus_ae.cuda()

    #PATH_ae_wights = "../" + json_config["weights_saved_at"]
    try:
        PATH_ae_wights = additional_path + json_config["weights_saved_at"]
    except KeyError:
        PATH_ae_wights = additional_path + json_config["weights_file"]
    torus_ae.load_state_dict(torch.load(PATH_ae_wights))
    torus_ae.eval()
    
    dict = {
        "train_loader" : train_loader,
        "test_loader" : test_loader,
        "tuned_neural_network": torus_ae,
        "json_config" : json_config
    }
    return dict