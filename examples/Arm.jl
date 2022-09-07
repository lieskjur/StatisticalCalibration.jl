using StatisticalCalibration
using LinearAlgebra

q̃(p̄,ϕ,Δl,Δψ) = [ ϕ, (p̄[1]+Δl)*cos(ϕ+p̄[2]+Δψ), (p̄[1]+Δl)*sin(ϕ+p̄[2]+Δψ) ]

# Problem
## Constraint function
f(p,q) = [ p[1]*cos(q[1]+p[2]) - q[2],
           p[1]*sin(q[1]+p[2]) - q[3] ]

## Weights
Cp = diagm([1,1])
Cq = diagm([1e-4,1e-4,1e-4])

## theoretical parameters & Generated test measurements
p̄ = [1,0]
Q̄ = mapreduce(ϕ->q̃(p̄,ϕ,1e-3,1e-3), hcat, [pi/2,pi/3,pi/4,pi/6,pi/8])

# Solution
p̂,Q̂,optf,ret = calibrate(f,Cp,Cq,p̄,Q̄,1e-12;xtol_rel=1e-12)
display(ret)
display(p̂)
display(Q̂)