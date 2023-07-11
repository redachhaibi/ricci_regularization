from setuptools import Extension, setup, find_packages
from os import path

local_path = path.abspath(path.dirname(__file__))

print("Local path: ", local_path)
print("")

print("Launching setup...")
# Setup
setup(
    name='ricci_regularization',

    version='0.01',

    description='Metric regularization of latent spaces using Ricci-type flows',
    long_description=""" TODO
    """,
    url='',

    author='Anonymous',
    author_email='Anonymous',

    license='MIT License',

    install_requires=["numpy", "matplotlib", "scipy", "scikit-learn", "torch", "torchvision", "tqdm", "pandas"],

    keywords='',

    packages=find_packages(),

    entry_points={
        'console_scripts': [
            'sample=sample:main',
        ],
    },

    ext_modules=[],
)