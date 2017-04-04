# gpdm.py: Implementation of GPDM for single sequence
# This can be used as a starting point for further implementations
# Author: Nishanth
# Date: 2017/01/17
# Source: GPflow source code

import numpy as np
import tensorflow as tf
from GPflow.tf_wraps import eye
from GPflow.model import GPModel
from GPflow._settings import settings
from GPflow.mean_functions import Zero
from GPflow.param import Param, DataHolder
from GPflow.densities import multivariate_normal
from GPflow import kernels, transforms, likelihoods

float_type = settings.dtypes.float_type

def PCA_reduce(X, Q):
    """
    A helpful function for linearly reducing the dimensionality of the data X
    to Q.
    :param X: data array of size N (number of points) x D (dimensions)
    :param Q: Number of latent dimensions, Q < D
    :return: PCA projection array of size N x Q.
    """
    assert Q <= X.shape[1], 'Cannot have more latent dimensions than observed'
    evecs, evals = np.linalg.eigh(np.cov(X.T))
    i = np.argsort(evecs)[::-1]
    W = evals[:, i]
    W = W[:, :Q]
    return (X - X.mean(0)).dot(W)

class GPDM(GPModel):
    """
    Gaussian Process Dynamical Model.
    This is a vanilla implementation of GPDM with uniformly sampled single sequence
    and mean prediction implementation.
    """

    def __init__(self, Y, latent_dim, X_mean=None, map_kern=None, dyn_kern=None):
        """
        Initialise GPDM object. This method only works with a Gaussian likelihood.
        :param Y: data matrix (N x D)
        :param X_mean: latent positions (N x Q), by default initialized using PCA.
        :param kern: kernel specification, by default RBF
        :param mean_function: mean function, by default None.
        """
        # initialize latent_positions
        if X_mean is None:
            X_mean = PCA_reduce(Y, latent_dim)

        # define kernel functions
        if map_kern is None:
            map_kern = kernels.RBF(latent_dim)

        if dyn_kern is None:
            dyn_kern = kernels.RBF(latent_dim) + kernels.Linear(latent_dim)

        # initialize variables
        self.num_latent = X_mean.shape[1]

        # initialize parent GPModel
        mean_function = Zero()
        likelihood = likelihoods.Gaussian()
        Y = DataHolder(Y, on_shape_change='pass')
        X = DataHolder(X_mean, on_shape_change='pass')
        GPModel.__init__(self, X, Y, map_kern, likelihood, mean_function)

        # initialize dynamics parameters
        self.dyn_kern = dyn_kern
        self.dyn_mean_function = Zero()
        self.dyn_likelihood = likelihoods.Gaussian()
        # set latent positions as model param
        del self.X
        self.X = Param(X_mean)

    def build_likelihood(self):
        """
        Construct a tensorflow function to compute the likelihood.
        \log p(Y | theta).
        """

        # dynamics log likelihood
        K_dyn = self.dyn_kern.K(self.X[:-1,:]) + eye(tf.shape(self.X[:-1,:])[0])*self.dyn_likelihood.variance
        L_dyn = tf.cholesky(K_dyn)

        # log likelihood is defined using multivariate_normal function
        diff_dyn = self.X[1:,:] - self.dyn_mean_function(self.X[:-1,:])
        alpha_dyn = tf.matrix_triangular_solve(L_dyn, diff_dyn, lower=True)

        # initialize model parameters
        num_dims_dyn = 1 if tf.rank(self.X[1:,:]) == 1 else tf.shape(self.X[1:,:])[1]
        num_dims_dyn = tf.cast(num_dims_dyn, float_type)
        num_points_dyn = tf.cast(tf.shape(self.X[1:,:])[0], float_type)

        # compute log likelihood
        llh_dyn = - 0.5 * num_dims_dyn * num_points_dyn * np.log(2 * np.pi)
        llh_dyn += - num_dims_dyn * tf.reduce_sum(tf.log(tf.diag_part(L_dyn)))
        llh_dyn += - 0.5 * tf.reduce_sum(tf.square(alpha_dyn))

        # mapping log likelihood
        K_map = self.kern.K(self.X) + eye(tf.shape(self.X)[0])*self.likelihood.variance
        L_map = tf.cholesky(K_map)

        # log likelihood is defined using multivariate_normal function
        diff_map = self.Y - self.mean_function(self.X)
        alpha_map = tf.matrix_triangular_solve(L_map, diff_map, lower=True)

        # initialize model parameters
        num_dims_map = 1 if tf.rank(self.Y) == 1 else tf.shape(self.Y)[1]
        num_dims_map = tf.cast(num_dims_map, float_type)
        num_points_map = tf.cast(tf.shape(self.Y)[0], float_type)

        # compute log likelihood
        llh_map = - 0.5 * num_dims_map * num_points_map * np.log(2 * np.pi)
        llh_map += - num_dims_map * tf.reduce_sum(tf.log(tf.diag_part(L_map)))
        llh_map += - 0.5 * tf.reduce_sum(tf.square(alpha_map))

        return llh_dyn+llh_map

    def build_predict(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, point at which we want to predict.
        This method computes, p(F* | Y ), where F* are points on the GP at Xnew.
        This will be similar to GP Regression.
        """

        # compute kernel for test points
        Kx = self.kern.K(self.X, Xnew)

        # compute kernel matrix and cholesky decomp.
        K = self.kern.K(self.X) + eye(tf.shape(self.X)[0]) * self.likelihood.variance
        L = tf.cholesky(K)

        # compute L^-1kx
        A = tf.matrix_triangular_solve(L, Kx, lower=True)
        # compute L^-1(y-mu(x))
        V = tf.matrix_triangular_solve(L, self.Y - self.mean_function(self.X))
        # compute fmean = kx^TK^-1(y-mu(x))
        fmean = tf.matmul(tf.transpose(A), V) + self.mean_function(Xnew)

        # diag var or full variance
        if full_cov:
            # compute kxx - kxTK^-1kx
            fvar = self.kern.K(Xnew) - tf.matmul(tf.transpose(A), A)
            shape = tf.stack([1, 1, tf.shape(self.Y)[1]])
            fvar = tf.tile(tf.expand_dims(fvar, 2), shape)
        else:
            # compute single value for variance
            fvar = self.kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(A), 0)
            fvar = tf.tile(tf.reshape(fvar, (-1, 1)), [1, tf.shape(self.Y)[1]])
        return fmean, fvar
