# bcgplvm.py: Implementation of back constrained GPLVM with minimal inheritance
# Author: Nishanth
# Date: 2017/01/18
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

class BCGPLVM(GPModel):
    """
    Gaussian Process Latent Variable Model.
    This is a vanilla implementation of GPLVM with Gaussian likelihood.
    """

    def __init__(self, Y, latent_dim, X_mean=None, kern=None, back_kern=None, mean_function=Zero()):
        """
        Initialise GPLVM object. This method only works with a Gaussian likelihood.
        :param Y: data matrix (N x D)
        :param X_mean: latent positions (N x Q), by default initialized using PCA.
        :param kern: kernel specification, by default RBF
        :param mean_function: mean function, by default None.
        """

        # define kernel function
        if kern is None:
            kern = kernels.RBF(latent_dim)
            back_kern = kernels.RBF(latent_dim)

        # initialize latent_positions
        if X_mean is None:
            X_mean = PCA_reduce(Y, latent_dim)

        # initialize variables
        self.num_latent = X_mean.shape[1]

        # initialize variables
        likelihood = likelihoods.Gaussian()
        Y = DataHolder(Y, on_shape_change='pass')
        X = DataHolder(X_mean, on_shape_change='pass')

        # initialize parent GPModel
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function)

        # initialize back constraint model
        self.back_kern = back_kern
        self.back_mean_function = Zero()
        self.back_likelihood = likelihoods.Gaussian()

        # set latent positions as model param
        del self.X
        self.X = Param(X_mean)

    def build_likelihood(self):
        """
        Construct a tensorflow function to compute the likelihood.
        \log p(Y | theta).
        """

        # forward mapping
        K_forward = self.kern.K(self.X) + eye(tf.shape(self.X)[0]) * self.likelihood.variance
        L_forward = tf.cholesky(K_forward)

        # log likelihood is defined using multivariate_normal function
        diff_forward = self.Y - self.mean_function(self.X)
        alpha_forward = tf.matrix_triangular_solve(L_forward, diff_forward, lower=True)

        # initialize model parameters
        num_dims_forward = 1 if tf.rank(self.Y) == 1 else tf.shape(self.Y)[1]
        num_dims_forward = tf.cast(num_dims_forward, float_type)
        num_points_forward = tf.cast(tf.shape(self.Y)[0], float_type)

        # compute log likelihood
        llh_forward = - 0.5 * num_dims_forward * num_points_forward * np.log(2 * np.pi)
        llh_forward += - num_dims_forward * tf.reduce_sum(tf.log(tf.diag_part(L_forward)))
        llh_forward += - 0.5 * tf.reduce_sum(tf.square(alpha_forward))

        # backward mapping
        K_backward = self.back_kern.K(self.Y) + eye(tf.shape(self.Y)[0]) * self.back_likelihood.variance
        L_backward = tf.cholesky(K_backward)

        # log likelihood is defined using multivariate_normal function
        diff_backward = self.X - self.mean_function(self.Y)
        alpha_backward = tf.matrix_triangular_solve(L_backward, diff_backward, lower=True)

        # initialize model parameters
        num_dims_backward = 1 if tf.rank(self.X) == 1 else tf.shape(self.X)[1]
        num_dims_backward = tf.cast(num_dims_backward, float_type)
        num_points_backward = tf.cast(tf.shape(self.X)[0], float_type)

        # compute log likelihood
        llh_backward = - 0.5 * num_dims_backward * num_points_backward * np.log(2 * np.pi)
        llh_backward += - num_dims_backward * tf.reduce_sum(tf.log(tf.diag_part(L_backward)))
        llh_backward += - 0.5 * tf.reduce_sum(tf.square(alpha_backward))

        return llh_forward+llh_backward

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
