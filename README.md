# Statistical Calibration

Package for the kinematic calibration of mechanisms based on the covariance of sensor measurements and calibrated dimension.

## Usage

The only exported method `calibrate(f,Cp,Cq,p̄,Q̄,tol;stopval,ftol_rel,...)` takes as postional arguments

1. residual function of constraint equations `f(p,q)=0` where `p` is a vector of calibrated dimesions and `q` a vector of sensor measurements
2. `np x np` and `nq x nq` covariance matrices for `p` and `q` respectively
3. vector of designed dimensions `p̄` and a matrix `Q̄` where each vector represents a measurement of `q`
4. vector (or number) `tol` determining the tolerance of individual (or all) constraint equation

and keyword arguments in the form of [stopping criteria](https://github.com/JuliaOpt/NLopt.jl#stopping-criteria) for [`NLopt`](https://github.com/JuliaOpt/NLopt.jl) problems which are formulated internaly. It's output are corrections `p̂` and `Q̂` the optimal value of `f` and the return code of the optimization `ret`.

## Example
here is an [example](examples/Arm.jl) on a single arm measuring it's endpoint in 2D Cartesian coordinates

## Installation
To install this package on your system simply paste
```
] add https://github.com/lieskjur/StatisticalCalibration.jl
```
into your julia repl

## Algorithm and Approach information

The approach is based around the *density function* of a [*multivariate normal distribution*](https://en.wikipedia.org/wiki/Multivariate_normal_distribution). The problem itself is then defined by a quadratic objective function corresponding to the probability of the corrections and equality constraints in the form of the mechanism's kinematic constaints.

The objective function being quadratic the *COBYLA* algorithm is used for finding the optimization's solution.