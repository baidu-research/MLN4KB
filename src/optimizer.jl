# This file includes various optimizers used to learning the weights of different rules

abstract type AbstractOptimizer end

########################################################################
#                 SGD
########################################################################
mutable struct SGDOptimizer <: AbstractOptimizer
    lr::Float64
    function SGDOptimizer(lr)
        new( lr )
    end
end

function update!( 
    optimizer::SGDOptimizer,
    iterate::Array{Float64},
    gradient::SparseVector{Float64, Int64}
)
    indices, vals = findnz( gradient )
    for i = 1:length(indices)
        idx = indices[i]
        val = vals[i]
        iterate[idx] -= optimizer.lr*val
        # Projection
        iterate[idx] = max(0., iterate[idx])
    end
    return nothing
end

########################################################################
#                 Adagrad
########################################################################
mutable struct AdaGradOptimizer <: AbstractOptimizer
    lr::Float64
    G::Array{Float64}   # diagnoal
    function AdaGradOptimizer(lr, iterate)
        d = length(iterate)
        G = 1e-6*ones( Float64, d )
        new( lr, G )
    end
end

function update!( 
    optimizer::AdaGradOptimizer,
    iterate::Array{Float64},
    gradient::SparseVector{Float64, Int64}
)
    indices, vals = findnz( gradient )
    for i = 1:length(indices)
        idx = indices[i]
        val = vals[i]
        optimizer.G[idx] += (val^2)    # Update G
        iterate[idx] -= optimizer.lr*val/sqrt( optimizer.G[idx] )
        # Projection
        iterate[idx] = max(0., iterate[idx])
    end
    return nothing
end

########################################################################
#                 Adam
########################################################################
mutable struct AdamOptimizer <: AbstractOptimizer
    lr::Float64
    β1::Float64
    β2::Float64
    curβ1::Float64      # curβ1 = β1^t ...
    curβ2::Float64
    m::Array{Float64}   # 1st momentum
    v::Array{Float64}   # 2nd momentum
    function AdamOptimizer(lr, iterate, β1=0.9, β2=0.999)
        d = length(iterate)
        m = 1e-6*ones( Float64, d )
        v = 1e-6*ones( Float64, d )
        curβ1 = β1
        curβ2 = β2
        new( lr, β1, β2, curβ1, curβ2, m, v )
    end
end

function update!( 
    optimizer::AdamOptimizer,
    iterate::Array{Float64},
    gradient::SparseVector{Float64, Int64}
)
    indices, vals = findnz( gradient )
    # Update m and v
    optimizer.m .= optimizer.β1*optimizer.m    
    optimizer.v .= optimizer.β2*optimizer.v
    for i = 1:length(indices)
        idx = indices[i]
        val = vals[i]
        optimizer.m[idx] += (1 - optimizer.β1)*val
        optimizer.v[idx] += (1 - optimizer.β2)*val*val
    end

    optimizer.curβ1 = optimizer.curβ1 * optimizer.β1
    optimizer.curβ2 = optimizer.curβ2 * optimizer.β2

    mhat = optimizer.m / ( 1 - optimizer.curβ1 )
    vhat = optimizer.v / ( 1 - optimizer.curβ2 )

    iterate .= iterate - optimizer.lr * (mhat./( sqrt.( vhat ) .+ 1e-8 )  )

    # Projection
    iterate .= max.(0., iterate)

    return nothing
end






