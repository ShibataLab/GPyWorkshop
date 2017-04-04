# interactive.py: demo to interact with latent space learnt by BGPLVM
# Author: Nishanth Koganti
# Date: 20117/4/1
# Source: Gaussian Process Summer School, 2015

# import libraries
import matplotlib

import GPy
import numpy as np
import cPickle as pickle

from matplotlib import pyplot as plt

# choose subset of digits to work on
which = [0,1,2,6,7,9] 
data = np.load('digits.npy')
data = data[which,:,:,:]
num_classes, num_samples, height, width = data.shape

# get the digits data and corresponding labels
Y = data.reshape((data.shape[0]*data.shape[1],data.shape[2]*data.shape[3]))
lbls = np.array([[l]*num_samples for l in which]).reshape(Y.shape[0], 1)
labels = np.array([[str(l)]*num_samples for l in which])

# load the pickle file
with open('digits.p', 'rb') as f:
    m = pickle.load(f)

# create interactive visualizer
fig = plt.figure('Latent Space', figsize=(16,6))
ax_latent = fig.add_subplot(121)
ax_scales = fig.add_subplot(122)

fig_out = plt.figure('Output', figsize=(1,1))
ax_image  = fig_out.add_subplot(111)
fig_out.tight_layout(pad=0)

data_show = GPy.plotting.matplot_dep.visualize.image_show(m.Y[0:1, :], dimensions=(16, 16), transpose=0, invert=0, scale=False, axes=ax_image)

lvm_visualizer = GPy.plotting.matplot_dep.visualize.lvm_dimselect(m.X.mean.copy(), m, data_show, ax_latent, ax_scales, labels=labels.flatten())
plt.show()
