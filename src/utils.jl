# Helper functions for the package

# Read facts.
function readFacts(
    filename::String
)
    facts = Array{Tuple{String, String, String, Int64}}([])
    io = open(filename, "r")
    while !eof(io)
        line = readline(io)
        line = split(line, '\t')
        e1, r, e2 = [ String(x) for x in line ]
        val = 1
        if r[1] == '!'
            val = 0
        end
        push!(facts, (e1, r, e2, val) )
    end
    close(io)
    return facts
end

# Read relation information.
function readRelationInfo(
    filename::String
)
    ret = Array{ Tuple{String,String} }([])
    io = open(filename, "r")
    while !eof(io)
        line = readline(io)
        line = split(line, r"(\t| )")
        @assert( length(line) == 2 )
        push!( ret, (String(line[1]), String(line[2]) ) )
    end
    close(io)
    return ret
end



# io = open("./examples/UMLS/info.txt", "w")
# for k in keys( mln.kb.relationMap )
#     println( io, k*"\tclosed" )
# end
# close(io)

# ListDict: a data structure that support efficient add, remove and sample.
# See https://stackoverflow.com/questions/15993447/python-data-structure-for-efficient-add-remove-and-random-choice
abstract type AbstractListDict{T<:Any} end

mutable struct ListDict{T} <: AbstractListDict{T}
    items::Array{T}
    itemToPosition::Dict{T, Int64}
    function ListDict{T}() where T
        items = Array{T}([])
        itemToPosition = Dict{T, Int64}()
        new{T}(items, itemToPosition)
    end
end

# # Define the length of ListDict.
function Base.length(ld::ListDict{T}) where T
    return length(ld.items)
end

# Adding item to ListDict.
function add!(ld::ListDict{T}, item::T) where T
    if haskey(ld.itemToPosition, item)
        # @warn("ListDict: item already exist.")
        return nothing
    end
    push!( ld.items, item )
    ld.itemToPosition[item] = length( ld.items )
    return nothing
end

# Removing item in ListDict.
function remove!(ld::ListDict{T}, item::T) where T
    pos = pop!(ld.itemToPosition, item)
    lastItem = pop!(ld.items)
    if pos != length(ld.items)+1
        ld.items[pos] = lastItem
        ld.itemToPosition[lastItem] = pos
    end
    return nothing
end

# Random sampling a item in ListDict.
function sample(ld::ListDict{T}) where T
    if length(ld.items) == 0
        return nothing
    end
    idx = rand(1:length(ld.items))
    return ld.items[idx]
end

# Replace item i with item j.
function update!(ld::ListDict{T}, originalItem::T, newItem::T)  where T
    @assert( haskey( ld.itemToPosition, originalItem ) )
    pos = ld.itemToPosition[ originalItem ]
    ld.itemToPosition[ newItem ] = pos
    ld.items[pos] = newItem
    if originalItem != newItem
        pop!( ld.itemToPosition, originalItem )
    end
    return nothing
end


# Randomly sample an item not in ListDict.
# function sampleNotIn(ld::ListDict, n::Int64)
#     maxIter = 100
#     while(maxIter)
#         idx = rand(1:n)
#         if !haskey(ld.itemToPosition, idx)
#             return idx
#         end
#     end
#     return nothing
# end

# Deep copy of ListDict
function Base.copy(ld::ListDict{T}) where T
    ldCpy = ListDict{T}()
    ldCpy.items = copy( ld.items )
    ldCpy.itemToPosition = copy( ld.itemToPosition )
    return ldCpy
end

# Some linear algebra subroutines
# y = a*x + y
function axpy!(
    a::Float64,
    x::Array{Float64},
    y::Array{Float64}
)
    @assert( length(x) == length(y) )
    n = length(x)
    @inbounds for i = 1:n
        y[i] = y[i] + a*x[i]
    end
    return nothing
end

# Calculate the inner product <x, y>
function sparse_dot(
    x::Array{Float64},
    y::SparseVector{Float64, Int64}
)
    n = length(x)
    ret = 0.0
    @inbounds for i = 1:y.n
        idx = y.nzind[i]
        val = y.nzval[i]
        ret += x[idx]*val
    end
    return ret
end

# -0.0088
# -0.6685
