# Truncated 2D Gaussian distribution

Truncated 2D Gaussian (covariate) distribution class with a method to efficiently calculate the probability density function and a rejection sampler. The 2D Gaussian distribution is assumed to be truncated over a rectangular box.

Performs some calculations in Cython. The PDF calculation follows "Genz, Alan. “Numerical Computation of Multivariate Normal Probabilities.” Journal of Computational and Graphical Statistics 1, no. 2 (1992): 141–49. https://doi.org/10.2307/1390838".


## TODO
- [ ] Prepare a proper setup file.
- [ ] Add example code in readme
- [ ] Make sure it can be pip installable.