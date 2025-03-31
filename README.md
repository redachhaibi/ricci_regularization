# Metric Regularization of Latent Spaces via Ricci-type gradient-flows

This package is work in progress in the context of Alexey Lazarev's PhD thesis.
The goal is to study numerical methods for metric regularization.

## Content

The repository is structured as follows. We only describe the most important files for a new user.
```bash
./
|-- ricci_regularization: Core of the package. 
|  |-- Architectures.py : Choice of AE architectures: TorusAE, TorusConvAE, etc.
|  |-- DataLoaders.py : Dataset loading and neural net weights loading.
|  |-- FiniteDifferences.py : Computing all useful Riemannian geometry tensors via Finite differences.
|  |-- Losscomputation.py : Train,test and loss comutation functions.
|  |-- OODTools.py : functions for OOD sampling
|  |-- PlottingTools.py : Plotting in latent space: Manifold plots, heatmaps, etc..
|  |-- Ricci.py : Computing all useful Riemannian geometry tensors via Autograd
|  |-- RiemannianKmeansTools.py : Ingredients of K-means and its plotting.
|  |-- Schauder.py : Schauder basis.
|  |-- SyntheticGaussians.py : Creating the Synthetic Gaussians dataset


|-- ipynb: Contains Python notebooks which demonstrate how the code works. Most important files:
|  |-- AE_torus_training.ipynb: training of the AE 
|  |-- AE_torus_report.ipynb: building the report
|  |-- Geodesic_grids_via_Stochman.ipynb: the Grid of geodesics and length ratio benchmark 
|  |-- torus3dembedding.ipynb: 3D Torus embedding
|  |-- Riemannian_K-means.ipynb: Benchmark of K-means clustering performance

|-- tests: TODO
|-- README.md: This file
```

## A quick start:

Step 1:
Train the neural net with curvature regularization by launching ipynb/AE_torus_training.ipynb. Save weights.

Step 2:
Generate and visualize the report of the training. Launch ipynb/AE_torus_report.ipynb.

Step 3:
Check the geodesics in the latent space. Launch ipynb/Geodesic_grids_via_Stochman.ipynb.

Step 4:
Check the quasi-isometric embedding of the Torus latent space. Launch ipynb/torus3dembedding.ipynb.

Step 5:
Perform clustering with Riemannian K-means. Launch ipynb/Riemannian_K-means.ipynb.

## Installation

1. Create new virtual environment

```bash
python3 -m venv .venv_ricci
```

(Do
sudo apt install python3-venv
if needed)

3. Activate virtual environment

```bash
source .venv_ricci/bin/activate
```

4. Upgrade pip, wheel and setuptools 

```bash
pip install --upgrade pip
pip install --upgrade setuptools
pip install wheel
```

5. Install the `ricci_regularization` package.

```bash
python setup.py develop
```

6. (Optional) In order to use Jupyter with this virtual environment .venv
```bash
pip install ipykernel
python -m ipykernel install --user --name=.venv_ricci
```
(see https://janakiev.com/blog/jupyter-virtual-envs/ for details)

7. Additionnal packages which could be removed in ulterior versions
```bash
pip install numpy torch torchvision umap-learn pypdf pdfkit jupyter stochman geomstats
```

## Configuration
Nothing to do

## Credits
Later
