module StatisticalCalibration
export calibrate
using NLopt

# Solver
struct ProblemDims
    np::Int # number of parameters
    nq::Int # number of coordinates
    nm::Int # number of measurements
    nx::Int # dimension of x
    nc::Int # dimension of c

    function ProblemDims(
        f::Function,
        p::AbstractVector{<:Real},
        Q::AbstractMatrix{<:Real}
        )
        np = length(p) 
        nq,nm = size(Q) 
        nf = length(f(p,Q[1,:]))
        nx = np+nq*nm
        nc = nf*nm
        return new(np,nq,nm,nx,nc)
    end
end

function unpack(dims::ProblemDims, x::AbstractVector{<:Real})
    @assert length(x) == dims.nx
    p = x[1:dims.np]
    Q = reshape(x[dims.np+1:end],dims.nq,dims.nm)
    return p,Q
end

## Quadratic objective function
function objective_function(
    dims::ProblemDims,
    iCp::AbstractMatrix{<:Real},
    iCq::AbstractMatrix{<:Real},
    x::AbstractVector{<:Real},
    grad::AbstractVector{<:Real}
    )
    length(grad) == 0 || error("use a derivative-free solver")
    p̂,Q̂ = unpack(dims,x)
    return p̂'*iCp*p̂ + mapreduce(q̂ -> q̂'*iCq*q̂, +, eachcol(Q̂))
end

## Non-linear constraints
function constraint_function(
    dims::ProblemDims,
    f::Function,
    p̄::AbstractVector{<:Real},
    Q̄::AbstractMatrix{<:Real},
    res::AbstractVector{<:Real},
    x::AbstractVector{<:Real},
    grad::AbstractMatrix{<:Real}
    )
    length(grad) == 0 || error("use a derivative-free solver")
    p̂,Q̂ = unpack(dims,x)
    res .= mapreduce( q -> f(p̄+p̂,q), vcat, eachcol(Q̂+Q̄))
    return nothing
end

## Assert weight matrix sizes
function calibrate(
    f::Function,
    Cp::AbstractMatrix{<:Real},
    Cq::AbstractMatrix{<:Real},
    p̄::AbstractVector{<:Real},
    Q̄::AbstractMatrix{<:Real},
    tol::Union{Real,AbstractVector{<:Real}};
    kwargs...
    )
    
    dims = ProblemDims(f,p̄,Q̄)

    @assert size(Cp) == (dims.np,dims.np)
    @assert size(Cq) == (dims.nq,dims.nq)
    iCp = inv(Cp)
    iCq = inv(Cq)
    
    ## Optimization problem
    opt = Opt(:LN_COBYLA, dims.nx)

    for arg in (:stopval,:ftol_rel,:ftol_abs,:xtol_rel,:xtol_abs,:maxeval,:maxtime)
        !haskey(kwargs,arg) || setproperty!(opt,arg,kwargs[arg])
    end

    opt.min_objective = (x,grad)->objective_function(dims,iCp,iCq,x,grad)
    con_func = (res,x,grad)->constraint_function(dims,f,p̄,Q̄,res,x,grad)
    if isa(tol,Real)
        equality_constraint!(opt, con_func, ones(dims.nc))
    else
        @assert length(tol) == dims.nc
        equality_constraint!(opt, con_func, tol)
    end
    
    # Solution
    optf,optx,ret = optimize(opt,zeros(dims.nx))
    optp̂,optQ̂ = unpack(dims,optx)

    return optp̂,optQ̂,optf,ret
end

end