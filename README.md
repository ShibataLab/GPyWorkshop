# GPyWorkshop

The aim of this workshop is to provide an introduction to GP, BGPLVM and MRD. It also includes a code review to implement these models in Python. Contents of the workshop:

* Theory
  * Gaussian Processes
  * Gaussian Process Latent Variable Model
  * Bayesian Gaussian Process Latent Variable Model
  * Manifold Relevance Determination
* Practical
  * GPy Library
  * GPflow Library

## Materials

Please clone this repository to access the materials for the workshop. This repository contains the presentation PDFs and Ipython notebooks with exercises on GPy.

## Requirements

Please install the following in your PC to run the experiments provided in the repository:
* [Anaconda](https://www.continuum.io/downloads): Necessary for Windows, optional for Linux.
* [Scipy Stack](https://www.scipy.org/index.html): This includes numpy, matplotlib and Ipython. Installation can be done using `pip`:
```
(sudo) pip install numpy --upgrade
(sudo) pip install jupyter --upgrade
(sudo) pip install matplotlib --upgrade
(sudo) pip install ipython[all] --upgrade
```
The `sudo` is optional if you want to have installation in the root folder when working in the Linux operating system. It should not be used for Anaconda or when working in the Windows operating system.
* [Tensorflow](https://www.tensorflow.org/install/): Dependency for the GPflow package. Installation instructions available on the Tensorflow website.
* [GPy](https://github.com/SheffieldML/GPy): Installation instructions available on the homepage.
* [GPflow](http://gpflow.readthedocs.io/en/latest/index.html): Installation instructions available on the homepage.

## PC setup for Ubuntu (14.04-16.04)

* GPy has dependencies for Fortran, LibBlas and LibAtlas. Please run the following command before installing GPy:
```
sudo apt-get install libblas-dev liblapack-dev libatlas-base-dev gfortran
```

## PC setup for Windows

For Windows Operating System:
* Install Anaconda by downloading from this [link](https://www.continuum.io/downloads#windows).
* Install GPy by opening the Command Prompt window and typing the following command:
```
pip install GPy
```
* All the dependencies required for running the code are either available in Anaconda or installed using GPy.
