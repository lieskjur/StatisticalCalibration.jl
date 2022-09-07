include("StatisticalCalibration.jl")
using .StatisticalCalibration
using LinearAlgebra

q̃(p̄,ϕ,Δl,Δψ) = [ ϕ, (p̄[1]+Δl)*cos(ϕ+p̄[2]+Δψ), (p̄[1]+Δl)*sin(ϕ+p̄[2]+Δψ) ]

# Problem
## Constraint function
f(p,q) = [ p[1]*cos(q[1]+p[2]) - q[2],
           p[1]*sin(q[1]+p[2]) - q[3] ]

## Weights
iCp = diagm([1,1])
iCq = diagm([1e4,1e4,1e4])

## theoretical parameters & Generated test measurements
p̄ = [1,0]
Q̄ = mapreduce(ϕ->q̃(p̄,ϕ,1e-3,1e-3), hcat, [pi/2,pi/3,pi/4,pi/6,pi/8])

# Solution
optp̂,optQ̂,optf,ret = calibrate(f,iCp,iCq,p̄,Q̄)
display(ret)
display(optp̂)
display(optQ̂)