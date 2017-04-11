## Questions

There were several questions related to the contents of the workshop and these are the follow-up responses:

### Theoretical Questions

* How to initialize model parameters as the complexity increases such as MRD, VGPDM, Deep GPs?

* How to compare various latent variable models?

### Model design

* How to design the kernel function for real-world datasets?
There is some intuition behind this modeling. If our application has periodic signals with some abrupt changes, it makes sense to use a periodic kernel along with a matern kernel. Here the periodicity is modeled by the periodic kernel and the global trend of the signal is modeled by the matern kernel. Further details are available in the [GPML book](http://www.gaussianprocess.org/gpml/chapters/RW4.pdf) by Carl Rasmussen. Additional information can obtained through this [video lecture](https://www.youtube.com/watch?v=rTlognaq4h0), [slides](http://ml.dcs.shef.ac.uk/gpss/gpws14/KernelDesign.pdf).

* How to set the dimensionality of the latent space prior for BGPLVM?
BGPLVM can automatically infer the dimensionality using ARD kernel parameters. However, it is important to set the initial dimensionality large enough to ensure that the data variance is captured. This needs to be done heuristically. If the dimensionality is too small, then all the ARD parameters will be non-zero and implies increasing the dimensionality. The dimensionality for training can be large as long as there are some ARD parameters that converge to zero.

### Model optimization

* How to use the optimize function in GPy?

* What is the intuition behind the constrained optimization of BGPLVM models?

### Latent space initialization

* Isn't an IID initialization of the variational distribution sufficient for BGPLVM?
In theory, the IID initialization should also work well for BGPLVM. However, practically it can be seen that the performance is much less in comparison to initializations based on PCA or kernel PCA. This problem is also partly due to the current optimizers that are used. With further research in optimization, the random initialization could work well and maybe even outperform linear initializations such as PCA.

* Is PCA initialization the only option for BGPLVM? How will random initialization perform on real-world datasets?
The examples in GPy usually use PCA initialization. In my practical experience, the initialization plays a crucial role in improving the performance of the trained model. However, GPy also provides the random initialization option and for small datasets the IID random initialization could be sufficient. NEED TO CHECK!

* Can tSNE be a good initialization for BGPLVM?
tSNE can be suitable as it is trained through a non-linear transformation through an optimization based approach and hence could be a better initialization for non-linear datasets. However, this further complicates an algorithm as an additional preplexity parameter for tSNE needs to be optimized to get a good latent space for BGPLVM. NEED TO CHECK!

* What is the prior used on the latent space for BGPLVM? Is the default option an IID prior?
The latent space prior for BGPLVM is an IID prior. This is used by default in GPy models.

### Hyper parameter initialization

* How to initialize the variance of RBF kernel and variance for Gaussian likelihood?
The RBF kernel variance is initialized using the observed data variance. A heuristic approach followed in the vargplvm toolbox is to define a signal-to-noise ratio (SNR) parameter which is our initial belief of the RBF kernel variance to noise variance ratio. This parameter is used to scale the RBF variance and initialize the variance for Gaussian likelihood.

* How to set the initial SNR value for the model?
For small datasets that are not to noisy. It is sufficient to set an initial SNR of 10. For large datasets that are high dimensional and noisy, it is required to use larger initial SNR values. As a rule of thumb, I set it large enough so that the SNR after training is atleast 10. In my study, I initialized SNR to between 100,1000 and the SNR converges to between 10,100. NEED TO CHECK THIS!

* How to set the number of inducing points for BGPLVM?
The inducing points approximate the latent space in the variational approximation and should be large enough to improve the performance. However, the computational efficiency also reduces with increasing inducing points. As a rule of thumb, I reduce the dataset size by a factor of 10 to set the number of inducing points. Here a more involved search can also be used to set the number.

* Does increasing the number of inducing points increase the accuracy of the model?
Theoretically, increasing the number of inducing points increases the accuracy. However, the number of optimization parameters also increases which could lead to decrease in the accuracy due to bad convergence during optimization and it is difficult to say. NEED TO CHECK THIS!

* How to initialize the length scale for RBF kernel?
For ARD kernel, the length scale parameter is the ARD weight parameter for each latent dimension which is optmized. There can be two strategies to initialize the length scale. All the length scales could be set to 1 given equal relevance to each dimension or length scale can be initialized by the normalized eigen values obtained from the PCA initialization. I usually found that using the PCA eigen values has a better performance as it already provides an estimate of the relevance of each dimension.

* Is it helpful to tune the ratio of the RBF kernel length scale and variance for the latent space prior?
I have not experimented with tuning this ratio and can not comment on the effect it might have on the model. NEED TO CHECK THIS!

### Software Questions

* Is it possible to make modifications to the models in GPy? Is GPflow more suitable for this?
It is possible to design new models and provide various objective functions in GPy. However, the gradients w.r.t each model parameter also needs to be implemented and this could be challenging. Alternately it is easier to implement new models in GPflow library as it relies on tensorflow which has automatic differentiation. In GPflow, only implementing the objective function/ likelihood function is sufficient and the gradients are automatically evaluated.

* How does GPy compare with Gaussian Processes in scikit-learn?
GP on scikit-learn includes GR regression, GP classification and does not include latent variable models. Further, it supports fewer kernel functions and only Gaussian likelihood. For these reasons, GPy is more suitable for development.

* What is the maximum size of inducing variables feasible for GPy?
I have experimented with a maximum number of inducing points 150 on a dataset of 1500 samples. The library worked without any overflow or divide by zero errors. However, the training time was between 3-4 hrs. This training time could be drastically improved on GPflow as it is implemented on tensorflow providing GPU or GPU cluster support as well.

* Are there numerical problems for large datasets?
For my practical application, I did not find any problems with using GPy. However, I can not comment on very large datasets of size greater than 100,000. In these scenarios using MCMC inference techniques could be more effective.
