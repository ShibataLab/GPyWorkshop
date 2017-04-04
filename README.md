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

This repository contains the presentation PDFs and Ipython notebooks with exercises on GPy.

## PC setup for Ubuntu (14.04-16.04)

* Install Docker by running the following command:
```
sudo apt-get install -y docker.io
```
* Clone the docker image for the workshop
```
sudo docker pull buntyke/gp-workshop
```
* Run the docker image for the workshop
```
sudo docker run -it -p 8888:8888 buntyke/gp-workshop
```
* The command outputs text that includes an URL like this:
```
Copy/paste this URL into your browser when you connect for the first time, to login with a token:
http://localhost:8888/?token=b8dbd6e58f68195b150bfcc69751bd97ddc20c097767d100
```
Copy the url and paste in the web browser to start the Ipython session.

## PC setup for Windows, Mac OSX

* Download docker-toolbox from this [link](https://www.docker.com/products/docker-toolbox).
* Install docker-toolbox and agree with all options. This will install Docker-Quickstart-Terminal.
* Open the Docker-Quickstart-Terminal application which opens a terminal.
* Download the docker image with this command:
```
docker pull buntyke/gp-workshop
```
* Run the docker image with this command:
```
docker run -it -p 8888:8888 buntyke/gp-workshop
```
The command outputs text that includes an URL like this:
```
Copy/paste this URL into your browser when you connect for the first time, to login with a token:
http://localhost:8888/?token=b8dbd6e58f68195b150bfcc69751bd97ddc20c097767d100
```
* Open a powershell and run the following command:
```
docker-machine.exe ip default
```
The command outputs an IP like this:
```
192.168.99.100
```
* Replace localhost with the IP and paste URL above in the web browser to start Ipython session:
```
http://192.168.99.100:8888/?token=b8dbd6e58f68195b150bfcc69751bd97ddc20c097767d100
```
