########################################################################
#                 Subroutines that support weights learning
########################################################################
using StatsBase

mutable struct Relation4Sampling
    idx::Int64
    involvedWithRule::Bool
    onlyPositive::Bool
    activeFacts::Array{ Tuple{Int64,Int64,Int64} }
    numActiveFacts::Int64
    positiveFacts::Array{ Tuple{Int64, Int64, Int64} }
    function Relation4Sampling(i)
        idx = i
        involvedWithRule = false
        onlyPositive = true
        activeFacts = Array{ Tuple{Int64,Int64,Int64} }([])
        numActiveFacts = 0
        positiveFacts = Array{ Tuple{Int64,Int64,Int64} }([])
        new(idx, involvedWithRule, onlyPositive, activeFacts, numActiveFacts, positiveFacts)
    end
end

# Store statistics about positve facts.
mutable struct PositiveFactsHolder
    idx::Int64
    facts::Array{ Tuple{Int64, Int64, Int64} }
    numPosFacts::Int64
    function PositiveFactsHolder(i)
        idx = i
        facts = Array{ Tuple{Int64, Int64, Int64} }([])
        numPosFacts = 0
        new( idx, facts, numPosFacts )
    end
end

# Initailize weight as all zeros
function ResetWeights!(
    rules::Array{Rule}
)
    for rule in rules
        rule.weight = 0.0
    end
    return nothing
end

# Count the number of violated formula groundings (groupped by ruleIdx) given the current 'violatedFormulaGrounding'
function CountNumViolatedGroundings(
    mln::MLN
)
    numRules = length( mln.rules )
    ret = zeros(Int64, numRules)
    for violated in mln.violatedFormulaGrounding.items
        ruleIdx = violated.ruleIdx
        ret[ruleIdx] += 1
    end
    return ret
end

# Discriminative learning the weights of each rule
function DiscriminativeLearning!(
    mln::MLN,
    lr::Float64=1e-1,
    λ::Float64=1e-2,   # regularization
    maxIter::Int64=100
)
    Reset!( mln )
    ResetWeights!( mln.rules )

    numRules = length(mln.rules)
    initialNumViolatedGroundings = CountNumViolatedGroundings(mln)

    grad = zeros(Float64, numRules)
    weights = zeros(Float64, numRules)
    # Main loop
    for iter = 1:maxIter
        # Run MAP inference
        Reset!(mln)
        MAPInference!(mln)
        # Calculate gradient
        curNumViolatedGroundings = CountNumViolatedGroundings(mln)
        grad .=  curNumViolatedGroundings - initialNumViolatedGroundings
        # Gradient update
        weights .= (1-λ)*weights
        axpy!( lr, grad, weights )
        # Projection
        @inbounds for i = 1:length(weights)
            weights[i] = max( 0.0, weights[i] )
        end
        for i = 1:numRules
            mln.rules[i].weight = weights[i]
        end
    end
    @printf("Learning finished!")
    return nothing
end


function ComputeDelta(
    mln::MLN,
    fact::Tuple{Int64,Int64,Int64}
)
    (e1, r, e2) = fact
    
    val = GetValue( mln.kb.relations[r], fact )
    @assert(val==1 || val==0)
    oldVal, newVal = val, 1 - val

    ruleIndices = copy( mln.relationToRule[r] )
    ruleIndices = collect(ruleIndices) # Convert set to array.
    # @show( fact )
    # @show( newVal )
    # for idx in ruleIndices
    #     @show( mln.rules[idx] )
    # end
    values = zeros(Float64, length(ruleIndices))
    for i = 1:length(ruleIndices)
        ruleIdx = ruleIndices[i]
        values[i] = length( mln.ruleToViolatedGrounding[ruleIdx] ) # Assign values
    end
    # @show(values)
    # Add a check
    valuesCheck = copy( values )

    flipFact = fact
    # @show(flipFact)
    # @show(newVal)
    # Flip
    UpdateRelationInfo!( mln.kb.relations, flipFact, newVal )
    UpdateMembers!( mln, flipFact, newVal )

    for i = 1:length(ruleIndices)
        ruleIdx = ruleIndices[i]
        values[i] -= length( mln.ruleToViolatedGrounding[ruleIdx] ) # Assign values
    end

    # Flip back
    UpdateRelationInfo!( mln.kb.relations, flipFact, oldVal )
    UpdateMembers!( mln, flipFact, oldVal )

    # Check
    for i = 1:length(ruleIndices)
        ruleIdx = ruleIndices[i]
        if valuesCheck[i] != length( mln.ruleToViolatedGrounding[ruleIdx].items ) # Assign values
            @show( flipFact )
            @show( newVal )
            error("Flipping back does recover the original result.\n")
        end
    end
    # Construct sparse vector and return
    ret = SparseVector( length(ruleIndices), ruleIndices, values )
    return ret
end


function ComputeDeltaOnlyPositive(
    mln::MLN,
    fact::Tuple{Int64,Int64,Int64}
)
    (e1, r, e2) = fact
    
    val = GetValue( mln.kb.relations[r], fact )
    @assert(val==1 || val==0)
    oldVal, newVal = val, 1 - val

    ruleIndices = copy( mln.relationToRule[r] )
    ruleIndices = collect(ruleIndices) # Convert set to array.
    # @show( fact )
    # @show( newVal )
    # for idx in ruleIndices
    #     @show( mln.rules[idx] )
    # end
    values = zeros(Float64, length(ruleIndices))
    for i = 1:length(ruleIndices)
        ruleIdx = ruleIndices[i]
        values[i] = length( mln.ruleToViolatedGrounding[ruleIdx] ) # Assign values
    end
    # @show(values)
    # Add a check
    valuesCheck = copy( values )

    flipFact = fact
    # @show(flipFact)
    # @show(newVal)
    # Flip
    UpdateRelationInfo!( mln.kb.relations, flipFact, newVal )
    numNewSatisfied, numNewViolated = UpdateMembers!( mln, flipFact, newVal )

    for i = 1:length(ruleIndices)
        ruleIdx = ruleIndices[i]
        # values[i] -= length( mln.ruleToViolatedGrounding[ruleIdx] ) # Assign values
        values[i] = numNewSatisfied[ ruleIdx ]
    end

    # Flip back
    UpdateRelationInfo!( mln.kb.relations, flipFact, oldVal )
    UpdateMembers!( mln, flipFact, oldVal )

    # Check
    for i = 1:length(ruleIndices)
        ruleIdx = ruleIndices[i]
        if valuesCheck[i] != length( mln.ruleToViolatedGrounding[ruleIdx].items ) # Assign values
            @show( flipFact )
            @show( newVal )
            error("Flipping back does recover the original result.\n")
        end
    end
    # Construct sparse vector and return
    ret = SparseVector( length(ruleIndices), ruleIndices, values )
    return ret
end

# Subroutine for computing the gradient.
function ComputeGradient(
    Δ::SparseVector{Float64, Int64},
    weights::Array{Float64}
)
    tmp = exp( sparse_dot( weights, Δ))
    grad = copy(Δ)
    for i = 1:grad.n
        idx = grad.nzind[i]
        val =  Δ.nzval[i]
        grad.nzval[i] = ( val*tmp ) / (1 + tmp)
    end
    return grad
end

# Compute objective value given a sample
function Obj(
    Δ::SparseVector{Float64, Int64},
    weights::Array{Float64}
)
    indices, vals = findnz(Δ)
    tmp = exp( sparse_dot( weights, Δ))
    ret = log(1 + tmp)
    return ret
end


# Generative learning, optimizing the pseudo-log-likelihood
function OptimizePseudoLogLikelihood!(
    mln::MLN,
    optimizer::AbstractOptimizer;
    numNegativeSamples::Int64=1,
    batchSize::Int64=128,
    # lr::Float64=1e-1,
    λ::Float64=1e-2,   # regularization
    maxIter::Int64=Int(1e6),
    resetMLN::Bool=true
)
    startTime = time()
    if resetMLN
        Reset!( mln )
    end
    # Initialize weights
    ResetWeights!( mln.rules )

    numRules = length(mln.rules)
    numEntities = mln.kb.numEntities
    grad = zeros(Float64, numRules)
    weights = zeros(Float64, numRules)       # The iterates

    # Preparation
    holders, prob = ConstructFactsHolder(mln)
    # relation4SamplingArray = ConstructRelation4Sampling(mln)
    # relationIndices = collect( 1:length(relation4SamplingArray) )
    # prob = [ x.numActiveFacts for x in relation4SamplingArray ]
    # prob = prob / sum(prob)
 
    numNullSteps = 0
    objVal = 0.0
    n = 0
    # Main loop
    for iter = 1:maxIter
        # Sampling candidate facts
        facts = NegativeSampling( mln, holders, prob, batchSize, numNegativeSamples )
        # Training this batch
        for fact in facts
            # Calculate delta
            Δ = ComputeDelta( mln, fact )
            # if (e1, e2) in mln.kb.relations[idx].evidences
            #     @show(fact)
            #     @show(Δ)
            # end
            # Compute objective
            objVal += Obj( Δ, weights )
            #@show(  Obj( Δ, weights ) )
            n += 1
            if sum( abs.( Δ.nzval ) ) < 1e-4
                numNullSteps += 1
            end
            # @show(Δ)
            # Compute gradient
            g = ComputeGradient( Δ, weights )
            # Update
            update!( optimizer, weights, g )
        end
        if iter % 100 == 0
            for i = 1:numRules
                mln.rules[i].weight = weights[i]
            end
            @printf("iter: %8i, obj: %5.5e, time elapsed: %5.5e\n", iter*batchSize, objVal/n, time() - startTime)
        end
    end
    @printf("portion of null steps: %f\n", numNullSteps/( maxIter*batchSize ))
    # Assign the resulting weights to rules
    for i = 1:numRules
        mln.rules[i].weight = weights[i]
    end
    return nothing
end

# Subroutine for negative sampling
function NegativeSampling(
    mln::MLN,
    holders::Array{PositiveFactsHolder},
    prob::Array{Float64},
    batchSize::Int64,
    numNegativeSamples::Int64
)
    numEntities = mln.kb.numEntities
    allRelations = collect(1:length(mln.kb.relations))
    ret = Array{ Tuple{Int64, Int64, Int64} }([])
    count = 0
    posFact = Tuple{Int64,Int64,Int64}( (0,0,0) )

    for i = 1:batchSize
        fact = Tuple{Int64,Int64,Int64}( (0,0,0) )
        p = rand(Uniform(0, 1))
        if (count == 0) 
            # Sample positve fact
            idx = wsample( allRelations, prob )
            holder = holders[idx]
            @assert( idx == holder.idx )
            (e1, idx, e2) = rand( holder.facts )
            fact = (e1, idx, e2)
            
            # Check single entity
            if mln.kb.relations[ idx ].singleEntity
                fact = ( e1, idx, e1)
            end
            posFact = fact
        else
            # Recall posFact
            (e1, idx, e2) = posFact
            # Only sample `e2`
            e2 = rand(1:numEntities)
            fact = (e1, idx, e2)

            # Check single entity
            if mln.kb.relations[ idx ].singleEntity
                fact = ( e1, idx, e1)
            end
        end
        # Push to result
        # @show(fact)
        push!( ret, fact )
        # Reset `count`
        if count == numNegativeSamples
            count = 0
        else
            count += 1
        end
    end

    return ret
end

########################################################################
#                 Sampling strategy
#   Case 1: the sampled relation is only involved in negation, then do 
#           brute-force sampling to sample a fact.
#   Case 2: the sampled relation is only involved in positive, then only
#           a few are active facts, store those active facts.
#   Case 3: the sampled relation is involved in both negation and positve,
#           do the same as case 1.
#   We should very ocassionally sample fact whose delta is 0.
########################################################################

function ConstructRelation4Sampling(
    mln::MLN
)
    numRelations = length(mln.kb.relations)
    numRules = length(mln.rules)
    numEntities = mln.kb.numEntities
    ret = Array{Relation4Sampling}([])
    for i = 1:numRelations
        newRel = Relation4Sampling(i)
        newRel.onlyPositive = true      # Set 'onlyPositive' to true initially
        push!( ret, newRel )
    end
    # Track if the relation only involves positive
    for i = 1:numRules
        rule = mln.rules[i]
        for rel in rule.relations
            relIdx = rel.idx
            negation = rel.negation
            ret[relIdx].involvedWithRule = true
            if negation
                ret[relIdx].onlyPositive = false
                ret[relIdx].numActiveFacts = numEntities^2    # All facts in this relation are active
            end
        end
    end
    numPos = sum( [ x.onlyPositive  for x in ret] )
    @printf("number of relations that only involve in positive: %i.\n", numPos)

    # Assign active facts
    for rel4sample in ret
        if !rel4sample.involvedWithRule
            continue
        end
        relIdx = rel4sample.idx
        positiveFacts = Array{ Tuple{Int64,Int64,Int64} }([])
        for e1 in keys( mln.kb.relations[relIdx].firstArgsNeighbours )
            for e2 in mln.kb.relations[relIdx].firstArgsNeighbours[e1].items
                push!( positiveFacts, (e1, relIdx, e2) )
            end
        end
        rel4sample.positiveFacts = positiveFacts

        activeFactsSet = Set{Tuple{Int64,Int64,Int64}}()
        if rel4sample.onlyPositive
            # relatedRuleIndices = mln.relationToRule[relIdx]
            # Add all positive
            for e1 in keys( mln.kb.relations[relIdx].firstArgsNeighbours )
                for e2 in mln.kb.relations[relIdx].firstArgsNeighbours[e1].items
                    push!( activeFactsSet, (e1, relIdx, e2) )
                end
            end
            # Check negative
            relatedRuleIndices = mln.relationToRule[relIdx]
            rulePos = Dict{Int64,Int64}()
            for ruleIdx in relatedRuleIndices
                rule = mln.rules[ruleIdx]
                for i = 1:length(rule.relations)
                    rel = rule.relations[i]
                    if rel.idx == relIdx && !rel.negation
                        @assert( !haskey(rulePos, ruleIdx) )
                        rulePos[ruleIdx] = i
                    end
                end
            end
            @assert( length(rulePos) == length(relatedRuleIndices) )
            for ruleIdx in relatedRuleIndices
                pos = rulePos[ ruleIdx ]
                violatedGroundings = mln.ruleToViolatedGrounding[ ruleIdx ]
                for grounding in violatedGroundings.items
                    fact = grounding.triplets[pos]
                    push!( activeFactsSet, fact )
                end
            end
            rel4sample.activeFacts = collect( activeFactsSet ) 
            rel4sample.numActiveFacts = length( rel4sample.activeFacts )
        end
    end
    return ret
end



# function SampleActiveFact(
#     mln::MLN,
#     activeRelationIndices::Array{Int64}
# )
#     relIdx = rand( activeRelationIndices )
#     # If the relation is only involved in negation

#     # If the relation is only involved in positive

#     # If the relation is involved in both negation and positive

# end

function ConstructFactsHolder(
    mln::MLN
    # holder::PositiveFactsHolder
)
    ret = Array{PositiveFactsHolder}([])
    for r = 1:length(mln.kb.relations)
        holder = PositiveFactsHolder(r)
        if r in keys(mln.relationToRule)
            for e1 in keys(mln.kb.relations[r].firstArgsNeighbours)
                for e2 in mln.kb.relations[r].firstArgsNeighbours[e1].items
                    push!( holder.facts, (e1, r, e2) )
                end
            end
        end
        holder.numPosFacts = length( holder.facts )
        push!( ret, holder )
    end
    prob = zeros(Float64, length(ret))
    for i = 1:length(ret)
        prob[i] = ret[i].numPosFacts
    end
    prob .= prob ./ sum(prob)
    return ret, prob
end




function NumFactInvolved!(
    mln::MLN
)
    ret = 0
    for rule in mln.rules
        for rel in rule.relations
            ret += mln.kb.numEntities^2
        end
    end
    @printf("number of facts involved: %i\n", ret)
    return nothing
end
