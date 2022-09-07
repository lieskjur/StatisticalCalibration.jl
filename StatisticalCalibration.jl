using NLopt
using LinearAlgebra

# Problem
## Constraint function
f(p,q) = [ p[1]*cos(q[1]+p[2]) - q[2],
           p[1]*sin(q[1]+p[2]) - q[3] ]

## theoretical parameters
p̄ = [1,0]

## Weights
iCp = diagm([1,1])
iCq = diagm([1e4,1e4,1e4])

## Generated test measurements
F(p̄,ϕ,Δl,Δψ) = [ ϕ, (p̄[1]+Δl)*cos(ϕ+p̄[2]+Δψ), (p̄[1]+Δl)*sin(ϕ+p̄[2]+Δψ) ]
Q̄ = mapreduce(ϕ->F(p̄,ϕ,1e-3,1e-3), hcat, [pi/2,pi/3,pi/4,pi/6,pi/8])

# Solver
## dimensions
np = length(p̄) # number of parameters
nq,nm = size(Q̄) # number of coordinates, number of measurements
nc = length(f(p̄,Q̄[1,:])) # number of constraint equations

n = np+nq*nm # dimension of x
m = nc*nm # dimension of c

## Assert weight matrix sizes
@assert size(iCp) == (np,np)
@assert size(iCq) == (nq,nq)

## Helper function to unpack the vector of optimized parameters
function unpack(x::Vector)
    @assert length(x) == n

    p = x[1:np]
    Q = reshape(x[np+1:end],nq,nm)
    return p,Q
end

## Quadratic objective function
function objective_function(x::Vector, grad::Vector)
    if length(grad) > 0
        error("use a derivative-free solver")
    end
    p̂,Q̂ = unpack(x)
    return p̂'*iCp*p̂ + mapreduce(q̂ -> q̂'*iCq*q̂, +, eachcol(Q̂))
end

## Non-linear constraints
function constraint_function(res::Vector, x::Vector, grad::Matrix)
    if length(grad) > 0
        error("use a derivative-free solver")
    end
    p̂,Q̂ = unpack(x)
    res .= mapreduce( q -> f(p̄+p̂,q), vcat, eachcol(Q̂+Q̄))
    return nothing
end

## Optimization problem
opt = Opt(:LN_COBYLA, n)
opt.xtol_rel = 1e-12

opt.min_objective = objective_function
equality_constraint!(opt, constraint_function, 1e-12*ones(m))

# Solution
optf,optx,ret = optimize(opt,zeros(n))
optp̂,optQ̂ = unpack(optx)

display(ret)
display(optp̂)
display(optQ̂)