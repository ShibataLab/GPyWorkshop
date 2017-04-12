## Additional Questions

There were several questions related to the contents of the workshop and these are the follow-up responses:

* Why are we using Python 2?

Both GPy and GPflow are supported in Python 2, 3. There is no specific reason for choosing Python 2 and either can be used. However, the Python 2 versions are still widely used and could have lesser bugs for advanced features of the library.

* How can GPLVM be trained without any input variable X?

GPLVM is a latent variable model and is a form of unsupervised machine learning. It tries to learn a latent manifold that can efficiently reconstruct high-dimensional observations.

* What is the domain of input X in the figure? Is it [0,10]?

In this particular example, we are inferring the GP mapping over input values [0,10]. However, given our observations the GP mapping can be evaluated over any sub domain of real numbers.

* What is the significance of estimating eigen values of covariance functions?

  For valid kernel functions the matrix needs to be positive semi-definite (PSD) (det(C) >= 0). For PSD matrix all eigen values are positive. To check this, we compute eigen values and see if any are negative. If they are then the kernel function is invalid and should not be used. If a kernel fuction generates PSD matrix then it has many useful properties. One property is called the [kernel trick](https://en.wikipedia.org/wiki/Kernel_method). This proves that for a PSD kernel there exists a feature space where a linear algorithm will also work thereby making optimization easier. Further details are available in the [GPML book](http://www.gaussianprocess.org/gpml/chapters/RW4.pdf) by Carl Rasmussen.

* What is the significance of kernel function in sampling functions from a GP? Does it always generate smooth functions?

  The RBF kernel generates smooth functions that are infinite differentiable. However, this need not always be the case. For example the Matern 3/2 kernel is only once differentiable and can generate a different class of functions. This can be visualized by sampling functions with different kernel functions. Further details are available in the [GPML book](http://www.gaussianprocess.org/gpml/chapters/RW4.pdf) by Carl Rasmussen.

* Can we guess how a composite kernel will work?

  The functions sampled from a composite kernel will roughly inherit the properties of each individual kernel. It is difficult to evaluate its functionality. However, the kernel design can be guided some intuition. For example, periodic signals need to be modeled using periodic kernel.

* Does 500 samples mean 500 Gaussian distributions with each mean centered at each sample.

  This modeling is similar to using radial basis functions as input parameters for a linear model. However for a GP, each sample is a dimension of a multivariate Gaussian. The correlation/covariance between any pair of dimensions/samples is given by estimating the kernel function for the samples.

* How to look for outliers in high-dimensional datasets?

  Visualizing each dimension and visually detecting outliers could be troublesome. Usually dimensionality reduction is applied and the most significant dimensions are visualized to detect outliers. This is a good starting point as the significant dimensions from models such as PCA indicate directions with maximum variance. Outliers in this space usually tend to be outliers in the high dimensional dataset as well.

* Will there always be a shared manifold between any set of high dimensional observation spaces?

  Usually, we are interested in learning a shared manifold in applications where we know a shared manifold exists. For example, we intuitively know that a shared manifold exists between EMG signals and finger kinematics. This manifold is the physiological process by which muscle activations causes the motion of the bones in our fingers. Furthermore, this manifold has nonlinear relationship to the signals measured by the senors. Models such as Manifold Relevance Determinance try to model the latent manifold as well as the mappings to the observation spaces.

* How to decide the ideal kernel function to use for an application?

  There is some intuition behind this modeling. If our application has periodic signals with some abrupt changes, it makes sense to use a periodic kernel along with a matern kernel. Here the periodicity is modeled by the periodic kernel and the global trend of the signal is modeled by the matern kernel. Further details are available in the [GPML book](http://www.gaussianprocess.org/gpml/chapters/RW4.pdf) by Carl Rasmussen.
