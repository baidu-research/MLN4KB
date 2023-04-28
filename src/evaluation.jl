# Read test facts and convert them into integers
function readTest(kb::KB, testFile::String)
    testFacts = readFacts(testFile)
    ret = Array{Tuple{Int64,Int64,Int64}}([])
    for fact in testFacts
        e1Str, rStr, e2Str = fact
        if haskey( kb.entityMap, e1Str )
            e1 = kb.entityMap[e1Str]
        else
            @printf("Unobserved entity in test file: %s\n", e1Str)
            continue
        end
        if haskey( kb.relationMap, rStr )
            r = kb.relationMap[rStr]
        else
            @printf("Unobserved relation in test file: %s\n", rStr)
            continue
        end
        if haskey( kb.entityMap, e2Str )
            e2 = kb.entityMap[e2Str]
        else
            @printf("Unobserved entity in test file: %s\n", e2Str)
            continue
        end
        push!( ret, (e1, r, e2) )
    end
    @printf("# test facts: %i\n", length(ret))
    return ret, length(testFacts)
end


# Load facts to MLN
function Load!(
    factFiles::Array{String},
    mln::MLN
)
    for factFile in factFiles
        facts = readFacts( factFile )
        for (e1Str, rStr, e2Str) in facts
            if !(e1Str in keys(mln.kb.entityMap)) || !(e2Str in keys(mln.kb.entityMap)) || !(rStr in keys(mln.kb.relationMap))
                continue
            end
            e1 = mln.kb.entityMap[e1Str]
            r = mln.kb.relationMap[rStr]
            e2 = mln.kb.entityMap[e2Str]
            # Add to mln.kb
            add!(mln.kb.relations[r].activeFirstArgs, e1)
            if !haskey(mln.kb.relations[r].firstArgsNeighbours, e1)
                # Rel.numActive += 1
                mln.kb.relations[r].firstArgsNeighbours[e1] = ListDict{Int64}()    
            end
            add!(mln.kb.relations[r].firstArgsNeighbours[e1], e2)

            add!(mln.kb.relations[r].activeSecondArgs, e2)
            if !haskey(mln.kb.relations[r].secondArgsNeighbours, e2)
                # Rel.numActive += 1
                mln.kb.relations[r].secondArgsNeighbours[e2] = ListDict{Int64}()
            end
            add!(mln.kb.relations[r].secondArgsNeighbours[e2], e1)
        end
    end
    return nothing
end

# 
function CalculateHitsAtK(
    testFacts::Array{Tuple{Int64, Int64, Int64}},
    kb::KB,
    k::Int64
)
    correct = 0
    for fact in testFacts
        e1, r, e2 = fact
        if haskey( kb.relations[r].firstArgsNeighbours, e1 ) && haskey( kb.relations[r].firstArgsNeighbours[e1].itemToPosition, e2 )
            correct += 1
        end
    end
    @printf("Test accuracy: %5.5e\n", correct/length(testFacts))
end


# Calculate both Recall and Hits@k with multiple KBs.
function EvaluateWithMultipleKBs(
    testFile::String,
    predictionFileList::Array{String},
    topK::Int64,
    kb::KB=nothing
)
    # testFile: testing file;
    # predictionFiles: a list of predictions;
    # Output: Recall score and Hits@k.

    if isnothing(kb)
        error("kb should not be nothing")
    end

    # Read test and predictions and convert them into integers.
    testFacts, numTest = readTest( kb, testFile )
    predictedFactsCounter = Dict{Tuple{Int,Int,Int}, Int}()
    for predictionFile in predictionFileList
        predictions = readFacts(predictionFile)
        for ( e1Str, rStr, e2Str, v ) in predictions
            @assert( v==1 )
            e1 = kb.entityMap[ e1Str ]
            e2 = kb.entityMap[ e2Str ]
            r = kb.relationMap[ rStr ]
            if !haskey( predictedFactsCounter, (e1, r, e2) )
                predictedFactsCounter[ (e1, r, e2) ] = 1
            else
                predictedFactsCounter[ (e1, r, e2) ] += 1
            end
        end
    end

    @show( length(predictedFactsCounter) )

    correct = 0

    badcases = Array{Tuple{String,String,String}}([])
    for fact in testFacts
        e1, r, e2 = fact
        # @show(fact)
        if haskey( predictedFactsCounter, fact )
            correct += 1
        else
            push!( badcases, ( kb.reverseEntityMap[e1], kb.reverseRelationMap[r], kb.reverseEntityMap[e2] ) )
        end
    end
    recall = correct/length(testFacts)
    @printf("Recall: %5.5e\n", recall)


    # Compute Hit@k
    hit = 0
    mrr = 0
    headRelDict = Dict{ Tuple{Int64, Int64}, Array{ Tuple{Int64, Float64} } }()     # key: (e1, r), val: (e2, val)

    # Construct headRelDict
    for (e1, r, e2) in keys(predictedFactsCounter)
        if !haskey( headRelDict, (e1, r) )
            headRelDict[ (e1, r) ] = Array{ Tuple{Int64, Float64} }([])
        end
        push!( headRelDict[ (e1, r) ], ( e2, predictedFactsCounter[(e1, r, e2)] ) )
    end

    @show( length(predictedFactsCounter) )
    cc = 0
    for (e1, r) in keys(headRelDict)
        cc += length( headRelDict[(e1, r)] )
    end
    @show( cc )

    # Sort
    for key in keys(headRelDict)
        headRelDict[key] = sort( headRelDict[key], by = x->x[2], rev=true )
        #@show( headRelDict[key] )
    end

    numMissing = 0
    numMissingTopk = 0

    for fact in testFacts
        e1, r, e2 = fact
        if haskey( headRelDict, (e1, r) )
            k = 1
            for (e2Candidate, val) in headRelDict[ (e1, r) ]
                if e2Candidate == e2
                    if k <= topK
                        hit += 1
                    end
                    mrr += (1/k)
                    break
                else
                    k += 1
                end
            end
            if length( headRelDict[ (e1, r) ] ) < topK
                numMissingTopk += 1
            end
        else
            numMissing += 1
        end
    end

    hitRatio = hit/numTest
    mrrRate = mrr/numTest
    @printf("Hit@%i: %5.5e\n", topK, hitRatio)
    @printf("MRR: %5.5e\n", mrrRate)
    @printf("# empty inference: %i, # inference small than %i: %i\n", numMissing, topK, numMissingTopk )

    return recall, hitRatio, mrrRate, badcases

   
end


# Calculate both Recall and Hits@k with multiple KBs.
function Evaluate(
    testFile::String,
    topKList::Array{Int64},
    mln::MLN
)
    # testFile: testing file;
    # predictionFiles: a list of predictions;
    # Output: Recall score and Hits@k.

    if isnothing(mln)
        error("mln should not be nothing")
    end

    # Read test and predictions and convert them into integers.
    numEntities = mln.kb.numEntities
    weights = zeros(Float64, length(mln.rules))
    for i = 1:length(mln.rules)
        weights[i] = mln.rules[i].weight
    end
    testFacts, numTest = readTest( mln.kb, testFile )

    # For getting statistics
    numTestPerRelation = Dict{Int64, Float64}()
    numCorrectPerRelation = Dict{Int64, Float64}()
    scoreList = []
    for i = 1:mln.kb.numRelations
        numCorrectPerRelation[i] = 0
        numTestPerRelation[i] = 0
    end
    for fact in testFacts
        e1, r, e2 = fact
        numTestPerRelation[r] += 1
    end


    hit = Dict{Int64,Float64}()
    for topK in topKList
        hit[topK] = 0
    end
    mrr = 0
    counter = 0
    numTailAllZeros = 0
    numTailAllZerosCorrect = 0
    numLargeTies = 0
    hitList = zeros(Int64, numTest)

    for fact in testFacts
        e1, r, e2 = fact
        if !(r in keys(mln.relationToRule))
            @printf("skip relation with no rule\n")
            continue
        end
        rank=1
        numTies = 0
        if isapprox( 1.0, GetValue( mln.kb.relations[r], (e1, r, e2) ), atol=1e-12 )
            # Only test ranking among 1s
            # score = -Inf
            Δ = ComputeDelta(mln, (e1, r, e2))
            score = -sparse_dot( weights, Δ )
            push!(scoreList, score)
            
            # score = zeros( Float64, length( mln.kb.relations[r].firstArgsNeighbours[e1] ) )
            for e2Candidate in mln.kb.relations[r].firstArgsNeighbours[e1].items
                if (e1, e2Candidate) in mln.kb.relations[r].evidences
                    # Skip comparison with evidences
                    continue
                end
                Δ = ComputeDelta(mln, (e1, r, e2Candidate))
                curScore = -sparse_dot( weights, Δ )
                if isapprox(curScore, score, atol=1e-12)
                    # if there are some ties, we randomly choose set the rank
                    numTies += 1
                elseif curScore > score
                    rank += 1
                end
                # To save computation time, we top if the rank is too large
                if rank >= 1000 || numTies >= 1000
                    rank = numEntities
                    break
                end 
            end
            rank += rand(0:numTies)
            rank = min(rank, numEntities)
            if numTies >= 1000
                numLargeTies += 1
            end
            # @show(rank)
        else
            # Only test ranking among 0s
            numEvidences = 0
            # Δ = ComputeDeltaOnlyPositive(mln, (e1, r, e2))
            Δ = ComputeDelta(mln, (e1, r, e2))

            score = sparse_dot( weights, Δ )
            if isapprox(score, 0, atol=1e-12)
                println( (mln.kb.reverseEntityMap[e1], mln.kb.reverseRelationMap[r], mln.kb.reverseEntityMap[e2]) )
                println( (e1, r, e2) )
                numTailAllZeros += 1
            end
            push!(scoreList, score)

            for e2Candidate = 1:numEntities
                if isapprox( 1.0, GetValue( mln.kb.relations[r], (e1, r, e2Candidate) ), atol=1e-12 )
                    if !( (e1, e2Candidate) in mln.kb.relations[r].evidences )
                        rank += 1
                    else
                        numEvidences += 1
                    end
                    continue
                end
                # Δ = ComputeDeltaOnlyPositive(mln, (e1, r, e2Candidate))
                Δ = ComputeDelta(mln, (e1, r, e2Candidate))
                curScore = sparse_dot( weights, Δ )
                if isapprox(curScore, score, atol=1e-12)
                    numTies += 1
                elseif curScore > score
                    rank += 1
                end
                # To save computation time, we top if the rank is too large
                if rank >= 1000 || numTies >= 1000
                    rank = numEntities
                    break
                end 
            end
            rank += rand(0:numTies)
            rank = min(rank, numEntities)
            if numEvidences == length( mln.kb.relations[r].firstArgsNeighbours[e1] )
                # @show(rank)
                # numTailAllZeros += 1
                # numTailAllZerosCorrect += (rank <= topK)
            end
            if numTies >= 1000
                numLargeTies += 1
            end
            # @show(rank)
        end
        counter += 1
        mrr += 1/rank
        for topK in topKList
            hit[topK] += (rank <= topK)
        end
        hitList[counter] = rank
        numCorrectPerRelation[r] += (rank <= 10)

        # @show( counter )
    end

    for topK in topKList
        hit[topK] = hit[topK] / numTest
        @printf("Hit@%i: %5.5e\n", topK, hit[topK])
    end
    # @show( counter )
    # hitRatio = hit/counter
    mrrRate = mrr/numTest
    @printf("MRR: %5.5e\n", mrrRate)
    # @show( numLargeTies )
    for i in keys( numCorrectPerRelation )
        numCorrectPerRelation[i] = numCorrectPerRelation[i] / (numTestPerRelation[i] + 1e-8)
    end
    @printf("num tail all zeros: %i, num correct: %i\n", numTailAllZeros, numTailAllZerosCorrect)
    # @printf("# empty inference: %i, # inference small than %i: %i\n", numMissing, topK, numMissingTopk )

    return hit, mrrRate, numCorrectPerRelation, scoreList, hitList

end






#########################################################
#           Testing reverse ranking
#########################################################


# Calculate the MRR and Hit@k, ranking by (?, r, t)
function EvaluateReverseTriplt(
    testFile::String,
    topKList::Array{Int64},
    mln::MLN
)
    # testFile: testing file;
    # predictionFiles: a list of predictions;
    # Output: Recall score and Hits@k.

    if isnothing(mln)
        error("mln should not be nothing")
    end

    # Read test and predictions and convert them into integers.
    numEntities = mln.kb.numEntities
    weights = zeros(Float64, length(mln.rules))
    for i = 1:length(mln.rules)
        weights[i] = mln.rules[i].weight
    end
    testFacts, numTest = readTest( mln.kb, testFile )
    
    hit = Dict{Int64,Float64}()
    for topK in topKList
        hit[topK] = 0
    end
    mrr = 0
    counter = 0
    numTailAllZeros = 0
    numTailAllZerosCorrect = 0
    numLargeTies = 0

    for fact in testFacts
        e1, r, e2 = fact
        if !(r in keys(mln.relationToRule))
            @printf("skip relation with no rule\n")
            continue
        end
        rank=1
        numTies = 0
        if isapprox( 1.0, GetValue( mln.kb.relations[r], (e1, r, e2) ) )
            # Only test ranking among 1s
            # score = -Inf
            Δ = ComputeDelta(mln, (e1, r, e2))
            score = -sparse_dot( weights, Δ )
            
            # score = zeros( Float64, length( mln.kb.relations[r].firstArgsNeighbours[e1] ) )
            for e1Candidate in mln.kb.relations[r].secondArgsNeighbours[e2].items
                if (e1Candidate, e2) in mln.kb.relations[r].evidences
                    # Skip comparison with evidences
                    continue
                end
                Δ = ComputeDelta(mln, (e1Candidate, r, e2))
                curScore = -sparse_dot( weights, Δ )
                if isapprox(curScore, score)
                    # if there are some ties, we randomly choose set the rank
                    numTies += 1
                elseif curScore > score
                    rank += 1
                end
                # To save computation time, we top if the rank is too large
                if rank >= 1000 || numTies >= 1000
                    rank = numEntities
                    break
                end 
            end
            rank += rand(0:numTies)
            rank = min(rank, numEntities)
            if numTies >= 1000
                numLargeTies += 1
            end
            # @show(rank)
        else
            # Only test ranking among 0s
            numEvidences = 0
            Δ = ComputeDelta(mln, (e1, r, e2))
            score = sparse_dot( weights, Δ )
            for e1Candidate = 1:numEntities
                if isapprox( 1.0, GetValue( mln.kb.relations[r], (e1Candidate, r, e2) ) )
                    if !( (e1Candidate, e2) in mln.kb.relations[r].evidences )
                        rank += 1
                    else
                        numEvidences += 1
                    end
                    continue
                end
                Δ = ComputeDelta(mln, (e1Candidate, r, e2))
                curScore = sparse_dot( weights, Δ )
                if isapprox(curScore, score)
                    numTies += 1
                elseif curScore > score
                    rank += 1
                end
                # To save computation time, we top if the rank is too large
                if rank >= 1000 || numTies >= 1000
                    rank = numEntities
                    break
                end 
            end
            rank += rand(0:numTies)
            rank = min(rank, numEntities)
            if numEvidences == length( mln.kb.relations[r].firstArgsNeighbours[e1] )
                # @show(rank)
                # numTailAllZeros += 1
                # numTailAllZerosCorrect += (rank <= topK)
            end
            if numTies >= 1000
                numLargeTies += 1
            end
            # @show(rank)
        end
        counter += 1
        mrr += 1/rank
        for topK in topKList
            hit[topK] += (rank <= topK)
        end
        # @show( counter )
    end

    for topK in topKList
        hit[topK] = hit[topK] / numTest
        @printf("Hit@%i: %5.5e\n", topK, hit[topK])
    end
    # @show( counter )
    # hitRatio = hit/counter
    mrrRate = mrr/numTest
    @printf("MRR: %5.5e\n", mrrRate)
    # @show( numLargeTies )
    # @printf("num tail all zeros: %i, num correct: %i\n", numTailAllZeros, numTailAllZerosCorrect)
    # @printf("# empty inference: %i, # inference small than %i: %i\n", numMissing, topK, numMissingTopk )

    return hit, mrrRate

end



#########################################################
#           Testing simple inference
#########################################################

function EvaluateSimpleInference(
    testFile::String,
    topKList::Array{Int64},
    mln::MLN
)
    if isnothing(mln)
        error("mln should not be nothing")
    end

    # Read test and predictions and convert them into integers.
    numEntities = mln.kb.numEntities
    weights = zeros(Float64, length(mln.rules))
    for i = 1:length(mln.rules)
        weights[i] = mln.rules[i].weight
    end
    testFacts, numTest = readTest( mln.kb, testFile )

end

function GetScore(
    rule::Rule,
    fact::Tuple{Int64, Int64, Int64}
)
    score = 0.0
    numPos = 0
    for rel in rule.relations
        if !rel.negation
            numPos += 1
        end
    end
    @assert( numPos == 1 )

    for rel in rule.relations
        if !rel.negation
            continue
        else

        end
    end
end

# Using DFS to construct violated formula groundings
function traversal2!(
    kb::KB,
    relations::Array{RelationInRule},
    level::Int64,
    assigned::Array{Int64},
    ret::Array{Array{Tuple{Int64, Int64, Int64}}},
    curPath::Array{Tuple{Int64, Int64, Int64}},
    # stats::Array{Int64} # stats[1]: # miss, stats[2] # search.
)
    # Empty stack, finish
    if level == 0
        return nothing
    end
    if level == length(relations)+1
        push!( ret, copy(curPath) )
        return nothing
    end

    # Extract current relation and its information
    relation = relations[level]
    relIdx = relation.idx
    var1 = relation.firstArg
    var2 = relation.secondArg
    var1Idx = assigned[var1]
    var2Idx = assigned[var2]
    rel = kb.relations[relIdx]

    # Check negation
    if !(relation.negation)
        if var1Idx == 0 || var2Idx == 0
            @show( relations )
        end
        @assert( var1Idx != 0 && var2Idx != 0 )
        if !haskey( rel.firstArgsNeighbours, var1Idx ) || !haskey(rel.firstArgsNeighbours[var1Idx].itemToPosition, var2Idx )
            push!(curPath, (var1Idx, relIdx, var2Idx))
            traversal!( kb, relations, level+1, assigned, ret, curPath )
            pop!(curPath)
            # record
            # stats[2] += 1
        end
        return nothing
    end

    newAssign = Dict{Int64,Int64}()
    if var1Idx != 0 && var2Idx != 0
        # Both variables are assigned
        if !haskey( rel.firstArgsNeighbours, var1Idx ) || !haskey(rel.firstArgsNeighbours[var1Idx].itemToPosition, var2Idx )
            return nothing
        else
            push!(curPath, (var1Idx, relIdx, var2Idx))
            traversal!( kb, relations, level+1, assigned, ret, curPath )
            pop!(curPath)
            # record
            # stats[2] += 1
        end    
    elseif var1Idx !=0 && var2Idx == 0
        # Var1 is assigned, Var2 is not 
        if !haskey( rel.firstArgsNeighbours, var1Idx )
            return nothing
        else
            for connectedIdx in rel.firstArgsNeighbours[var1Idx].items
                assigned[var2] = connectedIdx
                push!(curPath, (var1Idx, relIdx, connectedIdx))
                traversal!( kb, relations, level+1, assigned, ret, curPath )
                pop!(curPath)
                assigned[var2] = 0
                # record
                # stats[2] += 1
            end
        end
    elseif var1Idx ==0 && var2Idx != 0
        # Var1 is not assigned, Var2 is assigned.
        if !haskey( rel.secondArgsNeighbours, var2Idx )
            return nothing
        else
            for connectedIdx in rel.secondArgsNeighbours[var2Idx].items
                assigned[var1] = connectedIdx
                push!(curPath, (connectedIdx, relIdx, var2Idx))
                traversal!( kb, relations, level+1, assigned, ret, curPath )
                pop!(curPath)
                assigned[var1] = 0
                # record
                # stats[2] += 1
            end
        end
    else
        # Both Var1, Var2 are not assigned.
        for idx1 in rel.activeFirstArgs.items
            for idx2 in rel.firstArgsNeighbours[idx1].items
                assigned[var1] = idx1
                assigned[var2] = idx2
                push!(curPath, (idx1, relIdx, idx2))
                traversal!( kb, relations, level+1, assigned, ret, curPath )
                pop!(curPath)
                assigned[var1] = 0
                assigned[var2] = 0
                # record
                # stats[2] += 1
            end
        end
    end
    return nothing
end






