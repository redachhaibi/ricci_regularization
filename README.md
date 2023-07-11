# Metric Regularization of Latent Spaces via Ricci-type gradient-flows

This package is work in progress in the context of Alexei Lazarev's phD thesis.
The goal is to study numerical methods for metric regularization.

## Content

The repository is structured as follows. We only describe the most important files for a new user.
```bash
./
|-- ricci_regularization_ Core of package. 
|  |-- ???.py        : TODO
|-- ipynb: Contains Python notebooks which demonstrate how the code works
|  |-- ???.ipynb: TODO
|-- tests: Unit tests
|-- README.md: This file
```

## Installation

1. Create new virtual environment

```bash
$ python3 -m venv .venv_ricci
```

(Do
sudo apt install python3-venv
if needed)

3. Activate virtual environment

```bash
$ source .venv_ricci/bin/activate
```

4. Upgrade pip, wheel and setuptools 

```bash
$ pip install --upgrade pip
$ pip install --upgrade setuptools
$ pip install wheel
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

7. (Not needed if step 5 is used) Packages
```bash
pip install numpy matplotlib scipy torch torchvision
```

## Configuration
Nothing to do

## Credits
Later