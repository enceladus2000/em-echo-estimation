# Acoustic Echo Estimation using Expectation-Maximization

This repository contains implementations for expectation-maximization (EM) based algorithms for estimating the delay and direction of acoustic echoes. The algorithm is based on Jensen et al.'s work in [1](readme.md#references). 

## Improvements/Changes over original

The original work[1] outlines a mathematical description of the algorithm. This repository implements it in Python for single and multi channel audio. Additionally:

- Simple modification to the algorithm itself that theoretically guarantees globally optimal solution.
- Sub-sample level estimates using sinc-interpolation and nested optimization step.

## References
[1] J. R. Jensen, U. Saqib, and S. Gannot, “An Em Method for Multichannel Toa and Doa Estimation of Acoustic Echoes,” Research Portal Denmark, pp. 120–124, Oct. 2019, doi: https://doi.org/10.1109/waspaa.2019.8937252. Available: https://ieeexplore.ieee.org/abstract/document/8937252. [Accessed: Aug. 31, 2025]