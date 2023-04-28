########################################################################
#                 Subroutines that support inference
########################################################################

function MAPInference!(mln::MLN, maxIter = Int(1e6))
    WalkSAT!( mln, maxIter )
end

function tripletInRelation(
    relation::Relation,
    triplet::Tuple{Int64,Int64,Int64}
)
    e1, r, e2 = triplet
    if !haskey( relation.firstArgsNeighbours, e1 ) || !haskey( relation.firstArgsNeighbours[e1].itemToPosition, e2 )
        return false
    end
    return true
end

function flip(
    mln::MLN,
    grounding::FormulaGrounding,
    threshold::Float64 = 0.1
)
    numAtoms = grounding.numAtoms
    ruleIdx = grounding.ruleIdx
    isEvidence = zeros(Bool, numAtoms)
    # Fix evidences
    for i = 1:numAtoms
        e1, r, e2 = grounding.triplets[i]
        if (e1, e2) in mln.kb.relations[r].evidences || mln.kb.relations[r].closed
            isEvidence[i] = true
        else
            isEvidence[i] = false
        end
    end
    # println( rules[ruleIdx] )
    # println( grounding )
    # println( isEvidence )
    # Count the number of free atoms
    numFree = length(isEvidence) - sum(isEvidence)
    if numFree == 0
        # No free atoms, return nothing
        return -1, -1, -1
    end

    freeIndices = zeros(Int64, numFree)
    c = 1
    for i = 1:length(isEvidence)
        if !isEvidence[i]
            freeIndices[c] = i
            c += 1
        end
    end
    flipIdx = -1
    oldVal = - 1
    newVal = -1

    # Generate a random number in [0,1]
    p = rand(Uniform(0, 1))
    # p = 0.6
    if p <= threshold # || numFree == 1
        # Randomly sample a free index and flip
        flipIdx = rand(1:numFree)
        flipIdx = freeIndices[flipIdx]
        e1, r, e2 = grounding.triplets[flipIdx]

        oldVal, newVal = 0, 0
        if tripletInRelation( mln.kb.relations[r], (e1, r, e2) )
            oldVal = 1
        else
            newVal = 1
        end
        # Do a check, can be removed
        rel = mln.rules[ruleIdx].relations[flipIdx]
        if rel.negation
            @assert(newVal == 0)
        else
            @assert(newVal == 1)
        end
    else
        # @printf("running greedy selection! numFree: %i\n", numFree)
        # Greedy selection
        # preCost = computeCost( mln )
        preCost =  mln.obj
        curCost = Inf
        # @printf("numFree: %i\n", numFree)
        for i = 1:numFree
            curFlipIdx = freeIndices[i]
            flipFact = grounding.triplets[curFlipIdx]
            # @show(flipFact)
            # @show( mln.factToViolatedGrounding[flipFact] )
            e1, r, e2 = flipFact

            curOldVal, curNewVal = 0, 0
            if tripletInRelation( mln.kb.relations[r], (e1, r, e2) )
                curOldVal = 1
            else
                curNewVal = 1
            end
            # Do a check, can be removed
            rel = mln.rules[ruleIdx].relations[curFlipIdx]
            if rel.negation
                @assert(curNewVal == 0)
            else
                @assert(curNewVal == 1)
            end

            preLength = length(mln.violatedFormulaGrounding)
            UpdateRelationInfo!( mln.kb.relations, flipFact, curNewVal )
            UpdateMembers!( mln, flipFact, curNewVal )
            # @show( length(mln.violatedFormulaGrounding) )
            # Update relations accordingly

            cost = mln.obj + mln.ΔObj
            # Record flipIdx, oldVal, newVal
            if cost <= curCost + 1e-8
                # @printf("\t cost: %5.5e\n", cost)
                flipIdx = curFlipIdx
                oldVal = curOldVal
                newVal = curNewVal
                curCost = cost
            end

            # Flip back
            curOldVal, curNewVal = curNewVal, curOldVal
            UpdateRelationInfo!( mln.kb.relations, flipFact, curNewVal )
            UpdateMembers!( mln, flipFact, curNewVal )
            postLength = length(mln.violatedFormulaGrounding)


            @assert( isapprox( cost + mln.ΔObj, mln.obj ) )
            # Recover the original violated grounding list.
            @assert( preLength == postLength )
            # @show( length(mln.violatedFormulaGrounding) )
            # Update relations accordingly
        end
        # No descrease in cost, null step
        if curCost >= preCost
            return -1, -1, -1
        end
    end
    # Check
    @assert( flipIdx > 0 && oldVal >=0 && newVal >= 0 )
    # return flipping idx, original value and flipped value.
    return flipIdx, oldVal, newVal
end

# Violated to satisfied
# Remove the idx in indices from violatedFormulaGrounding
function RemoveFromViolatedGroundingList!(
    mln::MLN,
    # violatedFormulaGrounding::Array{FormulaGrounding},
    # indices::Array{Int64}
    grounding::FormulaGrounding
)
    # Remove from 'mln.violatedFormulaGrounding'
    remove!( mln.violatedFormulaGrounding, grounding )
    
    # Remove from 'factToViolatedGrounding'
    for triplet in grounding.triplets
        if haskey(mln.factToViolatedGrounding, triplet) && haskey( mln.factToViolatedGrounding[triplet].itemToPosition, grounding ) # This condition can be false very occasionally, for example [(4825, 223, 4826), (4825, 223, 4826), (4825, 223, 4825)]
            remove!( mln.factToViolatedGrounding[triplet], grounding )
            if length( mln.factToViolatedGrounding[triplet] ) == 0
                pop!( mln.factToViolatedGrounding, triplet )
            end
        end
    end

    # Remove from 'ruleToViolatedGrounding'
    ruleIdx = grounding.ruleIdx
    remove!(mln.ruleToViolatedGrounding[ruleIdx], grounding)

end

# Add a new violated grounding to mln and update 'violatedFormulaGrounding', 'factToViolatedGrounding' and 'ruleToViolatedGrounding' accordingly.
function AddToViolatedGroundingList!(
    mln::MLN,
    # violatedFormulaGrounding::Array{FormulaGrounding},
    newViolatedGrounding::FormulaGrounding
)
    # newViolatedGroundingCpy = copy(newViolatedGrounding)
    add!( mln.violatedFormulaGrounding, newViolatedGrounding )

    # Update 'factToViolatedGrounding'
    for triplet in newViolatedGrounding.triplets
        if haskey( mln.factToViolatedGrounding, triplet )
            add!( mln.factToViolatedGrounding[triplet], newViolatedGrounding )
        else
            mln.factToViolatedGrounding[triplet] = ListDict{FormulaGrounding}()
            add!( mln.factToViolatedGrounding[triplet], newViolatedGrounding )
        end
    end

    # Update 'ruleToViolatedGrounding'
    ruleIdx = newViolatedGrounding.ruleIdx
    add!( mln.ruleToViolatedGrounding[ruleIdx], newViolatedGrounding )

    return nothing
end

function CheckFormulaSatisfied(
    rules::Array{Rule},
    grounding::FormulaGrounding,
    flippedTriplet::Tuple{Int64, Int64, Int64},
    newVal::Int64
)
    ruleIdx = grounding.ruleIdx
    check = 0
    for i = 1:grounding.numAtoms
        curTriplet = grounding.triplets[i]
        if curTriplet == flippedTriplet
            check += 1
            negation = rules[ruleIdx].relations[i].negation
            if (negation && newVal == 1) || (!negation && newVal == 0)
                return false
            end
            break
        end
    end
    # flippedTriplet must appear in the given grounding.
    @assert(check >= 1) 
    return true
end

function NewViolatedGroundings(
    kb::KB,
    rule::Rule,
    flippedTriplet::Tuple{Int64, Int64, Int64},
    newVal::Int64
)
    e1, r, e2 = flippedTriplet
    # Find possible position to put in flippedTriplet
    possiblePos = zeros(Int64, 0)
    relations = rule.relations
    for i = 1:length(relations)
        if relations[i].idx == r && ( (relations[i].negation && newVal == 1  ) || (!relations[i].negation && newVal == 0  ) )
            push!( possiblePos, i )
        end
    end

    ret = Set{Array{ Tuple{Int64,Int64,Int64} }}([])
    total = 0
    for pos in possiblePos

        violations = Array{Array{Tuple{Int64, Int64, Int64}}}([])
        assigned = zeros(Int64, rule.numVars)
        # Assign pos vars
        assigned[ relations[pos].firstArg ] = e1
        assigned[ relations[pos].secondArg ] = e2
        curPath = Array{Tuple{Int64, Int64, Int64}}([])

        traversal!(kb, relations, 1, assigned, violations, curPath)
        # Push violated tr
        for grounding in violations
            push!( ret, copy(grounding) )
        end
    end
    return collect(ret)
end



# Subroutine to update members after flipping
function UpdateMembers!(
    mln::MLN,
    flippedTriplet::Tuple{Int64,Int64,Int64},
    # violatedGrounding::FormulaGrounding,
    # flipIdx::Int64,
    newVal::Int64
)
    # Reset ΔObj
    mln.ΔObj = 0.0

    (e1, r, e2) = flippedTriplet

    # Record some statistics 
    relatedRuleIndices = mln.relationToRule[r]
    numNewSatisfied = Dict{Int64,Int64}()
    numNewViolated = Dict{Int64,Int64}()
    for ruleIdx in relatedRuleIndices
        numNewSatisfied[ruleIdx] = 0
        numNewViolated[ruleIdx] = 0
    end
    # Violated that becomes satisfied.
    if haskey( mln.factToViolatedGrounding, flippedTriplet )    # Should always be true for inference, can be false for learning
        relatedGroundings = copy(mln.factToViolatedGrounding[flippedTriplet].items)

        ####################
        #      Check
        ####################
        # for grounding in relatedGroundings
        #     # grounding = mln.violatedFormulaGrounding[idx]
        #     for triplet in grounding.triplets
        #         if !haskey( mln.factToViolatedGrounding[triplet].itemToPosition, grounding )
        #             @printf("Should not happen.\n")
        #             @assert(1==2)
        #         end
        #     end
        # end
        # @printf("related groundings:\n")
        # println( mln.violatedFormulaGrounding[relatedIndices] )
        # @printf("******************\n")
        # @show(relatedIndices)

        # indicesToRemove = zeros(Int64, 0)
        for grounding in relatedGroundings
            isSatisfied = CheckFormulaSatisfied(mln.rules, grounding, flippedTriplet, newVal)
            # Remove the violated grounding if becoming satisfied.
            if isSatisfied
                # Remove from violated grounding list and corresponding members
                RemoveFromViolatedGroundingList!(mln, grounding)
                # Update ΔObj
                mln.ΔObj -= mln.rules[ grounding.ruleIdx ].weight
                numNewSatisfied[ grounding.ruleIdx ] += 1
            end
        end
    # else
        # error("no violation, should not happen.\n")
    end

    # Satisfied that becomes violated.
    relatedRuleIndices = mln.relationToRule[r]
    # @show( relatedRuleIndices )
    for idx in relatedRuleIndices
        rule = mln.rules[idx]
        newViolatedGroundings = NewViolatedGroundings( mln.kb, rule, flippedTriplet, newVal )
        # @show(idx)
        # @show( newViolatedGroundings )
        for eachNewViolatedGroundings in newViolatedGroundings
            # Construct struct
            eachNewViolatedGroundings = FormulaGrounding( idx, eachNewViolatedGroundings )
            # Add
            AddToViolatedGroundingList!( mln, eachNewViolatedGroundings )
            # Update ΔObj
            mln.ΔObj += mln.rules[ eachNewViolatedGroundings.ruleIdx ].weight
            numNewViolated[ eachNewViolatedGroundings.ruleIdx ] += 1
        end
        ####################
        #      Check
        ####################
        # CheckDuplicateGrounding!( mln.violatedFormulaGrounding )
    end
    return numNewSatisfied, numNewViolated
end

function UpdateRelationInfo!(
    relations::Array{Relation},
    triplet::Tuple{Int64, Int64, Int64},
    newVal::Int64
)
    e1, r, e2 = triplet
    rel = relations[r]
    if newVal == 0
        # Remove (e1, r, e2) from rel
        @assert( haskey( rel.firstArgsNeighbours, e1 ) )
        @assert( haskey( rel.secondArgsNeighbours, e2 ) )
        remove!( rel.firstArgsNeighbours[e1], e2 )
        remove!( rel.secondArgsNeighbours[e2], e1 )
        if length(rel.firstArgsNeighbours[e1]) == 0
            remove!( rel.activeFirstArgs, e1 )
        end
        if length(rel.secondArgsNeighbours[e2]) == 0
            remove!( rel.activeSecondArgs, e2 )
        end
    else
        # Add (e1, r, e2) to rel
        add!( rel.activeFirstArgs, e1 )
        add!( rel.activeSecondArgs, e2 )
        if !haskey( rel.firstArgsNeighbours, e1 )
            rel.firstArgsNeighbours[e1] = ListDict{Int64}()
            add!( rel.firstArgsNeighbours[e1], e2 )
        else
            add!( rel.firstArgsNeighbours[e1], e2 )
        end

        if !haskey( rel.secondArgsNeighbours, e2 )
            rel.secondArgsNeighbours[e2] = ListDict{Int64}()
            add!( rel.secondArgsNeighbours[e2], e1 )
        else
            add!( rel.secondArgsNeighbours[e2], e1 )
        end
    end
    return nothing
end

# The core functionality
# Subroutine to solve the MaxWeightedSAT problem by local search
function WalkSAT!(
    mln::MLN;
    threshold::Float64=0.1,
    maxIter::Int64=Int(10),
    interval::Int64=Int(1e4),
    warmupPeriod::Int64=Int(1e5),
    check::Bool=true,
    save::Bool=true
)
    numRules = length(mln.rules)
    # Main Loop
    startTime = time()
    count = 0
    skip = 0
    bestCost = Inf
    # Temporary
    objList = Array{Float64}([])
    numViolatedList = Array{Int64}([])
    # Record time checking MLN
    if check
        CheckMLNValid( mln )
    end
    curTime = time()
    @printf("Time spent on sanity check : %5.5e \n", curTime - startTime)
    # Keep track of the best assignment
    bestKb = copy( mln.kb )
    for iter = 1:maxIter
        # Reset ΔObj
        mln.ΔObj = 0.0
        # Compute cost and save result every iteration after warm-up.
        if save && iter > warmupPeriod
            #cost = computeCost(mln)
            if mln.obj < bestCost
                bestKb = copy( mln.kb )
                bestCost = mln.obj
            end
        end
        # @show(cost)
        # @show( mln.violatedFormulaGrounding.items[1] )
        if length( mln.violatedFormulaGrounding ) == 0
            # Update best KB and return
            bestKb = copy( mln.kb )
            bestCost = 0.0
            @printf("iter: %6i, all rules are satsified!\n", iter)
            break
        end
        if count % interval == 0
            if check
                CheckMLNValid( mln )
            end
            if save && mln.obj < bestCost
                bestKb = copy( mln.kb )
                bestCost = mln.obj
            end
            push!( objList, mln.obj )
            push!( numViolatedList, length(mln.violatedFormulaGrounding) )
            curTime = time()
            @printf("\niter: %6i, cost: %5.5e, best cost: %5.5e, time elapsed: %5.3e, skip percentage: %5.4e\n\n", count, mln.obj, bestCost, curTime - startTime, skip/(count+1e-8) )
        end
        count += 1
        # Sample a violated triplet
        violatedIdx = rand( 1:length(mln.violatedFormulaGrounding) )
        violated = mln.violatedFormulaGrounding.items[violatedIdx]
        # Flip an atom
        flipIdx, oldVal, newVal = flip(mln, violated, threshold)
        # Skip if no free atom
        if flipIdx == -1
            skip += 1
            continue
        end
        flipFact = violated.triplets[flipIdx]
        # Update members accordingly
        # @show( length(mln.violatedFormulaGrounding) )
        UpdateRelationInfo!( mln.kb.relations, flipFact, newVal )
        UpdateMembers!( mln, flipFact, newVal )
        # @show( length(mln.violatedFormulaGrounding) )
        # Update relations accordingly
        # Update cost
        mln.obj += mln.ΔObj
    end
    # Post-processing
    @printf("Post processing\n")
    # Assign to the best solution
    mln.kb = bestKb
    PrepareMLN!(mln)
    CheckMLNValid( mln )
    bestCost = mln.obj
    @printf("best cost: %5.5e\n", bestCost)
    return objList, numViolatedList
end

# Compute cost that can not be removed.


# Check whether there is duplicate formula groundings in 'violatedFormulatedGrounding'.
function CheckDuplicateGrounding!(
    violatedFormulaGrounding::ListDict{FormulaGrounding}
)
    allGroundings = Set{Array{Int64}}()
    for grounding in violatedFormulaGrounding.items
        groundingArray = Array{Int64}([grounding.ruleIdx])
        for triplet in grounding.triplets
            append!( groundingArray, triplet )
        end
        if !( groundingArray in allGroundings )
            push!( allGroundings, groundingArray )
        else
            error("duplicate grounding exist.\n")
        end
    end
    return nothing
end


