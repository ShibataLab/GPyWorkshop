# GPyWorkshop

The aim of this workshop is to provide an introduction to Gaussian Processes and get the participants started with GPy Library. Contents of the workshop:

* Part 1
  * Gaussian Processes
  * Gaussian Process Latent Variable Model
* Part 2
  * Bayesian Gaussian Process Latent Variable Model
  * Manifold Relevance Determination

## Materials

Please clone this repository to access the materials for the workshop. This repository contains the presentation PDFs and Ipython notebooks with exercises on GPy.

## Requirements

Please install the following in your PC before coming to the workshop:
* [GPy](https://github.com/SheffieldML/GPy): Installation instructions available on the homepage.
* [Scipy Stack](https://www.scipy.org/index.html): This includes numpy, matplotlib and Ipython. Installation can be done using `pip`:
```
(sudo) pip install numpy --upgrade
(sudo) pip install jupyter --upgrade
(sudo) pip install matplotlib --upgrade
(sudo) pip install ipython[all] --upgrade
```
The `sudo` is optional if you want to have installation in the root folder when working in the Linux operating system. It should not be used for Anaconda or when working in the Windows operating system.

* [Anaconda](https://www.continuum.io/downloads): Necessary for Windows, optional for Linux.

## PC setup for Windows

For Windows Operating System:
* Install Anaconda by downloading from this [link](https://www.continuum.io/downloads#windows).
* Install GPy by opening the Command Prompt window and typing the following command:
```
pip install GPy
```
* All the dependencies required for running the code are either available in Anaconda or installed using GPy.
