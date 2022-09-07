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
    iCp::AbstractMatrix{<:Real},
    iCq::AbstractMatrix{<:Real},
    p̄::AbstractVector{<:Real},
    Q̄::AbstractMatrix{<:Real},
    tol::Union{Real,AbstractVector{<:Real}};
    stopval=nothing,
    ftol_rel=nothing,
    ftol_abs=nothing,
    xtol_rel=nothing,
    xtol_abs=nothing,
    maxeval=nothing,
    maxtime=nothing
    )
    
    dims = ProblemDims(f,p̄,Q̄)

    @assert size(iCp) == (dims.np,dims.np)
    @assert size(iCq) == (dims.nq,dims.nq)

    ## Optimization problem
    opt = Opt(:LN_COBYLA, dims.nx)

    stopval != nothing ? opt.stopval = stopval : nothing
    ftol_rel != nothing ? opt.ftol_rel = ftol_rel : nothing
    ftol_abs != nothing ? opt.ftol_abs = ftol_abs : nothing
    xtol_rel != nothing ? opt.xtol_rel = xtol_rel : nothing
    xtol_abs != nothing ? opt.xtol_abs = xtol_abs : nothing
    maxeval != nothing ? opt.maxeval = maxeval : nothing
    maxtime != nothing ? opt.maxtime = maxtime : nothing

    opt.min_objective = (x,grad)->objective_function(dims,iCp,iCq,x,grad)
    con_func = (res,x,grad)->constraint_function(dims,f,p̄,Q̄,res,x,grad)
    if isa(tol,Real)
        equality_constraint!(opt, con_func, ones(dims.nc))
    else
        @assert length(tol) = dims.nc
        equality_constraint!(opt, con_func, tol)
    end
    
    # Solution
    optf,optx,ret = optimize(opt,zeros(dims.nx))
    optp̂,optQ̂ = unpack(dims,optx)

    return optp̂,optQ̂,optf,ret
end

end