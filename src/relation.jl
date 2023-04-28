# Struct for relation used in rule
mutable struct RelationInRule
    firstArg::Int64
    secondArg::Int64
    idx::Int64
    negation::Bool
    function RelationInRule(
        var1::Int64,
        var2::Int64,
        idx::Int64,
        negation::Bool
    )
        new(var1, var2, idx, negation)        
    end
end


# Struct for relation
mutable struct Relation
    numActive::Int64
    activeFirstArgs::ListDict{Int64}
    activeSecondArgs::ListDict{Int64}
    firstArgsNeighbours::Dict{Int64, ListDict{Int64}}
    secondArgsNeighbours::Dict{Int64, ListDict{Int64}}
    evidences::Set{Tuple{Int64, Int64}}   # r( a, b )
    idx::Int64
    singleEntity::Bool
    closed::Bool
    function Relation(idx::Int64)
        numActive = 0
        activeFirstArgs = ListDict{Int64}()
        activeSecondArgs = ListDict{Int64}()
        firstArgsNeighbours = Dict{Int64, ListDict{Int64}}()
        secondArgsNeighbours = Dict{Int64, ListDict{Int64}}()
        evidences = Set{Tuple{Int64, Int64}}([])
        singleEntity = false
        closed = false
        new(0, activeFirstArgs, activeSecondArgs, firstArgsNeighbours, secondArgsNeighbours, evidences, idx, singleEntity, closed)
    end
end

# Adding a triplet to relation, r must correspond to this Relation
function add!(
    rel::Relation, 
    triplet::Tuple{Int64, Int64, Int64},
    val::Int64,
    isEvidence::Bool
)
    (e1, r, e2) = triplet
    @assert(r == rel.idx)
    # Add to firstArgsNeighbours and secondArgsNeighbours only if val == 1
    if val == 1 
        add!(rel.activeFirstArgs, e1)
        if !haskey(rel.firstArgsNeighbours, e1)
            # Rel.numActive += 1
            rel.firstArgsNeighbours[e1] = ListDict{Int64}()    
        end
        add!(rel.firstArgsNeighbours[e1], e2)

        add!(rel.activeSecondArgs, e2)
        if !haskey(rel.secondArgsNeighbours, e2)
            # Rel.numActive += 1
            rel.secondArgsNeighbours[e2] = ListDict{Int64}()
        end
        add!(rel.secondArgsNeighbours[e2], e1)
    end

    # Check if triplet is evidence, add if it is.
    if isEvidence
        push!(rel.evidences, (e1, e2) )
    end

    return nothing
end

# Remove a triplet in relation, r must correspond to this Relation
function remove!(
    rel::Relation, 
    triplet::Tuple{Int64, Int64, Int64}
)
    (e1, r, e2) = triplet
    # Checks.
    @assert(r == rel.idx)
    @assert( haskey(rel.firstArgsNeighbours, e1) )
    @assert( haskey(rel.secondArgsNeighbours, e2) )
    @assert( !((e1, e2) in rel.evidences ) )

    # Real operations.
    delete!( rel.firstArgsNeighbours[e1], e2 )
    delete!( rel.firstArgsNeighbours[e2], e1 )

    return nothing
end

function GetValue(
    rel::Relation,
    triplet::Tuple{Int64, Int64, Int64}
)
    (e1, r, e2) = triplet
    @assert( r == rel.idx )
    ret = 0
    if haskey( rel.firstArgsNeighbours, e1 ) && haskey( rel.firstArgsNeighbours[e1].itemToPosition, e2 )
        ret = 1
    end
    return ret
end

function check!(
    rel::Relation
)
    for e1 in keys(rel.firstArgsNeighbours)
        for e2 in rel.firstArgsNeighbours[e1].items
            @assert( haskey( rel.secondArgsNeighbours, e2 ) )
            @assert( haskey( rel.secondArgsNeighbours[e2].itemToPosition, e1 ) )
            # @assert( (e1, e2) in rel.evidences )
        end
    end

    for e2 in keys(rel.secondArgsNeighbours)
        for e1 in rel.secondArgsNeighbours[e2].items
            @assert( haskey( rel.firstArgsNeighbours, e1 ) )
            @assert( haskey( rel.firstArgsNeighbours[e1].itemToPosition, e2 ) )
        end
    end
    return nothing
end

function UpdateNumActive!(
    rel::Relation
)
    rel.numActive = 0
    for e1 in keys(rel.firstArgsNeighbours)
        for e2 in rel.firstArgsNeighbours[e1].items
            rel.numActive += 1
        end
    end
    return nothing
end

function Base.show(
    rel::Relation
)
    entitySet1 = Set( keys(rel.firstArgsNeighbours) )
    entitySet2 = Set( keys(rel.secondArgsNeighbours) )
    allActiveEntitis = union( entitySet1, entitySet2 )
    activeFacts = Set{Tuple{Int64,Int64}}()
    for e1 in keys(rel.firstArgsNeighbours)
        for e2 in rel.firstArgsNeighbours[e1].items
            push!( activeFacts, (e1, e2) )
        end
    end
    for e2 in keys(rel.secondArgsNeighbours)
        for e1 in rel.secondArgsNeighbours[e2].items
            push!( activeFacts, (e1, e2) )
        end
    end
    @printf("\n\tRelation idx: %i\n", rel.idx)
    @printf("\t# entities involved: %i\n", length(allActiveEntitis))
    @printf("\t# active facts: %i\n", length(activeFacts) )
    if rel.singleEntity
        @printf("\tSingle entity\n")
    else
        @printf("\tDoulbe entities\n")
    end
    if rel.closed
        @printf("\tClosed relation\n")
    else
        @printf("\tOpen relation\n")
    end
end

function Base.copy(
    rel::Relation
)
    relCpy = Relation(-1)

    relCpy.numActive = copy(rel.numActive)
    relCpy.activeFirstArgs = deepcopy(rel.activeFirstArgs)
    relCpy.activeSecondArgs = deepcopy(rel.activeSecondArgs)
    relCpy.firstArgsNeighbours = deepcopy(rel.firstArgsNeighbours)
    relCpy.secondArgsNeighbours = deepcopy(rel.secondArgsNeighbours)
    relCpy.evidences = deepcopy(rel.evidences)
    relCpy.idx = copy(rel.idx)
    relCpy.singleEntity = copy(rel.singleEntity)
    relCpy.closed = copy(rel.closed)

    return relCpy
end

# # Reset relations
# function reset!(
#     rel::Relation
# )
#     rel.activeFirstArgs = ListDict{Int64}()
#     rel.activeSecondArgs = ListDict{Int64}()
#     rel.firstArgsNeighbours = Dict{Int64, ListDict{Int64}}()
#     rel.secondArgsNeighbours = Dict{Int64, ListDict{Int64}}()
#     for (e1, e2) in rel.evidences
#         if !haskey( rel.activeFirstArgs.itemToPosition, e1 )
#             add!( rel.activeFirstArgs, e1 )
#         end
#         if !haskey( rel.activeSecondArgs.itemToPosition, e2 )
#             add!( rel.activeSecondArgs, e2 )
#         end
#         # Add to 'firstArgsNeighbours' and 'secondArgsNeighbours'
#         if !haskey( rel.firstArgsNeighbours, e1 )
#             rel.firstArgsNeighbours[e1] = ListDict{Int64}()
#         end
#         if !haskey( rel.firstArgsNeighbours[e1], e2 )
#             add!( rel.firstArgsNeighbours[e1], e2 )
#         end
#         if !haskey( rel.secondArgsNeighbours, e2 )
#             rel.secondArgsNeighbours[e2] = ListDict{Int64}()
#         end
#         if !haskey( rel.secondArgsNeighbours[e2], e1 )
#             add!( rel.secondArgsNeighbours[e2], e1 )
#         end
#     end
#     return nothing
# end
