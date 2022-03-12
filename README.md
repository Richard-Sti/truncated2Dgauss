# Truncated 2D Gaussian distribution

A tool to calculate the probability density function of a 2-dimensional truncated Gaussian distribution along with rejection sampling. Relies on Cython to integrate the unconstrained Gaussian distribution over the specified 2-dimensional box.

## TODO

- [ ] Reference the paper where the calculation is done.
- [x] Clean up the documentation
- [ ] Write a proper Readme
- [x] Clean up the imports

## Notes
- In the rejection sampling assumed infinite loop, maybe add some maximum iteration?