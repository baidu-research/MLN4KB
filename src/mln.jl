########################################################################
#                       Struct for Markov Logic Network
########################################################################
mutable struct FormulaGrounding
    # FormulaGrounding: a grounding for a rule that violated.
    ruleIdx::Int64
    numAtoms::Int64
    triplets::Array{Tuple{Int64, Int64, Int64}}
    function FormulaGrounding(
        ruleIdx::Int64,
        triplets::Array{Tuple{Int64, Int64, Int64}}
    )
        new( ruleIdx, length(triplets), triplets )
    end
end


mutable struct MLN
    kb::KB
    rules::Array{Rule}
    # Members used for efficient local search
    violatedFormulaGrounding::ListDict{FormulaGrounding}
    factToViolatedGrounding::Dict{Tuple{Int64, Int64, Int64}, ListDict{FormulaGrounding}}  # key: triplet, value: FormulaGrounding struct
    ruleToViolatedGrounding::Array{ListDict{FormulaGrounding}} # item: FormulaGrounding struct
    relationToRule::Dict{Int64, Set{Int64}} # key: relation index, value: indices of rule
    obj::Float64
    ΔObj::Float64
    function MLN(
        factFile::String,
        ruleFile::String;
        relationInfoFile::String=""
    )
        facts = readFacts(factFile)
        kb = KB(facts)
        parseRelations!(kb)

        @printf("Read facts complete!\n")

        # Read info and prune rules
        if relationInfoFile != ""
            readRelationInfoToKB!( kb, relationInfoFile )
        end
        rules = readRules(kb, ruleFile)
        checkRules!(rules)

        @printf("Read rules complete!\n")
        @printf("# rules: %5i\n", length(rules))

        violatedFormulaGrounding = ListDict{FormulaGrounding}()
        factToViolatedGrounding = Dict{Tuple{Int64, Int64, Int64}, ListDict{FormulaGrounding}}()
        ruleToViolatedGrounding = Array{ListDict{FormulaGrounding}}([])
        relationToRule = Dict{Int64, Set{Int64}}()
        # numViolationByRule = zeros(Int64, length(rules))
        # numSatisfiedByRule = zeros(Int64, length(rules))

        obj = 0.0
        ΔObj = 0.0
        @printf("Initialization complete!\n")

        new(kb, rules, violatedFormulaGrounding, factToViolatedGrounding, ruleToViolatedGrounding, relationToRule, obj, ΔObj )
    end
end

# Reading whether relation is closed or open.
function readRelationInfoToKB!(
    kb::KB,
    relationInfoFile::String
)
    relationInfo = readRelationInfo(relationInfoFile)
    for (relName, info) in relationInfo
        # Add relation if has not been observed.
        if !haskey( kb.relationMap, relName )
            @warn("unobserved relation in facts:", relName)
            numRel = length( kb.relations )
            kb.relationMap[relName] = numRel+1
            kb.numRelations += 1
            newRelation = Relation(numRel+1)
            newRelation.singleEntity = false    # place-holder, we don't know if the relation's input is single or double entities yet.
            push!( kb.relations, newRelation )
        end
        relIdx = kb.relationMap[relName]
        if info == "closed"
            kb.relations[relIdx].closed = true
        end
    end
    return nothing
end

# Reading the ruleFile to KB.
function readRules(
    kb::KB,
    ruleFile::String
)
    rules = Array{Rule}([])
    io = open(ruleFile, "r")
    while !eof(io)
        line = readline(io)
        newRule = Rule(kb, line)
        # Does not accept rule with non-positive weight (rule with zero weight can be removed).
        if !isnothing(newRule) && !RuleIsClosed( kb, newRule ) && newRule.weight > 1e-14
            push!(rules, newRule )
        end
    end
    close(io)
    return rules
end

function PrepareMLN!(
    mln::MLN
)
    # Reset all auxilary variables.
    mln.violatedFormulaGrounding = ListDict{FormulaGrounding}()
    mln.factToViolatedGrounding = Dict{Tuple{Int64, Int64, Int64}, ListDict{FormulaGrounding}}()
    mln.ruleToViolatedGrounding = Array{ListDict{FormulaGrounding}}([])
    mln.relationToRule = Dict{Int64, Set{Int64}}()

    # Initialize 'violatedFormulaGrounding'
    InitializeViolatedFormulaGroundings!(mln)

    # Initialize 'factToViolatedGrounding'
    InitializeFactToViolatedGrounding!(mln)

    # Initialize 'ruleToViolatedGrounding'
    InitializeRuleToViolatedGrounding!(mln)

    # Initialize 'relationToRule', this should never change after initialization.
    InitializeRelationToRule!(mln)

    # Intialize cost.
    mln.obj = computeCost(mln)

    return nothing
end

# Reset the MLN
function Reset!(
    mln::MLN
)
    # Reset members to empty
    mln.violatedFormulaGrounding = ListDict{FormulaGrounding}()
    mln.factToViolatedGrounding = Dict{Tuple{Int64, Int64, Int64}, ListDict{FormulaGrounding}}()
    mln.ruleToViolatedGrounding = Array{ListDict{FormulaGrounding}}([])
    mln.relationToRule = Dict{Int64, Array{Int64}}()
    mln.numViolationByRule = zeros(Int64, length(mln.rules))
    mln.numSatisfiedByRule = zeros(Int64, length(mln.rules))
    
    # Reset kb, no need to change rules
    Reset!(mln.kb)
    PrepareMLN!(mln)
    return nothing
end

function InitializeViolatedFormulaGroundings!(
    mln::MLN
)
    # mln.violatedFormulaGrounding = Array{FormulaGrounding}([])
    for i = 1:length(mln.rules)
        rule = mln.rules[i]
        violations = FindAllViolationsGivenRule(mln.kb, rule)
        for grounding in violations
            formulaGrounding = FormulaGrounding( i, grounding )
            add!( mln.violatedFormulaGrounding, formulaGrounding )
        end
    end
    return nothing
end


function InitializeFactToViolatedGrounding!(
    mln::MLN
)
    # mln.factToViolatedGrounding = Dict{ Tuple{Int64,Int64,Int64}, ListDict{Int64} }()
    for i =1:length(mln.violatedFormulaGrounding.items)
        grounding = mln.violatedFormulaGrounding.items[i]
        for triplet in grounding.triplets
            if !haskey( mln.factToViolatedGrounding, triplet )
                mln.factToViolatedGrounding[triplet] = ListDict{FormulaGrounding}()
            end
            add!( mln.factToViolatedGrounding[triplet], grounding )
        end
    end
    return nothing
end

function InitializeRuleToViolatedGrounding!(
    mln::MLN
)
    numRules = length(mln.rules)
    for i = 1:numRules
        ld = ListDict{FormulaGrounding}()
        push!( mln.ruleToViolatedGrounding, ld )
    end
    for i = 1:length(mln.violatedFormulaGrounding.items)
        grounding = mln.violatedFormulaGrounding.items[i]
        ruleIdx = grounding.ruleIdx
        add!( mln.ruleToViolatedGrounding[ruleIdx], grounding )
    end
    return nothing
end

function InitializeRelationToRule!(
    mln::MLN
)
    rules = mln.rules
    for i = 1:length(rules)
        rule = rules[i]
        for rel in rule.relations
            relIdx = rel.idx
            if !haskey( mln.relationToRule, relIdx )
                mln.relationToRule[relIdx] = Set{Int64}()
            end
            push!( mln.relationToRule[relIdx], i )
        end
    end
end


# Compute the cost for MLN.
function computeCost(
    mln::MLN
)
    ret = 0.0
    for i = 1:length(mln.rules)
        weight = mln.rules[i].weight
        # increment by weight times number of violation
        ret += ( weight*length(mln.ruleToViolatedGrounding[i].items) )
    end
    return ret
end

# Extract weights of rules
function GetRulesWeights(
    mln::MLN
)
    numRules = length( mln.rules )
    ret = zeros(Float64, numRules)
    for i = 1:numRules
        ret[i] = mln.rules[i].weight
    end
    return ret
end

# Using DFS to construct violated formula groundings
function traversal!(
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


function FindAllViolationsGivenRule(
    kb::KB,
    rule::Rule
)
    violations = Array{Array{Tuple{Int64, Int64, Int64}}}([])
    relations = rule.relations
    assigned = zeros(Int64, rule.numVars)
    curPath = Array{Tuple{Int64, Int64, Int64}}([])
    # stats = [0, 0] # record statistics 

    traversal!(kb, relations, 1, assigned, violations, curPath)
    # @printf("hit rate: %5.5e\n", length(violations)/stats[2] )
    return violations
end


# Check whether a grounding is satisfied.
function CheckFormulaSatisfied2(
    mln::MLN,
    grounding::FormulaGrounding
)
    ruleIdx = grounding.ruleIdx
    relations = mln.rules[ruleIdx].relations
    @assert( length(grounding.triplets) == length(relations) )

    for i = 1:length(relations)
        e1, r, e2 = grounding.triplets[i]
        rel = relations[i]
        @assert( r == rel.idx )
        negation = rel.negation
        isTrue = false
        if haskey( mln.kb.relations[r].firstArgsNeighbours , e1 ) && haskey( mln.kb.relations[r].firstArgsNeighbours[e1].itemToPosition, e2 )
            isTrue = true
        end
        # If 1 then true
        if (negation && !isTrue) || (!negation && isTrue)
            return true
        end
    end
    return false
end


function CheckMLNValid(
    mln::MLN
)
    # Check if violated groundings are all valid
    for grounding in mln.violatedFormulaGrounding.items
        ruleIdx = grounding.ruleIdx
        isSatisfied = CheckFormulaSatisfied2( mln, grounding)
        if isSatisfied
            println( grounding )
            error("MLN check failed, some violated groundings are satisfied.\n")
        end
    end

    # Check duplicated groundings
    CheckDuplicateGrounding!(mln.violatedFormulaGrounding)
    
    # Check 'factToViolatedGrounding'
    for grounding in mln.violatedFormulaGrounding.items
        for triplet in grounding.triplets
            if !haskey( mln.factToViolatedGrounding, triplet )
                error("MLN check failed, violated tripet does not appear in 'violatedFormulaGrounding'.\n")
            end
        end
    end
    for i = 1:length(mln.violatedFormulaGrounding.items)
        grounding = mln.violatedFormulaGrounding.items[i]
        for triplet in grounding.triplets
            if !haskey( mln.factToViolatedGrounding[triplet].itemToPosition, grounding )
                error("Check failed, formula grounding does not appear in 'factToViolatedGrounding[triplet]'.\n")
            end
        end
    end
    for triplet in keys(mln.factToViolatedGrounding)
        ld = mln.factToViolatedGrounding[triplet]
        for grounding in ld.items
            if !(triplet in grounding.triplets)
                error("Check failed, keys from 'mln.factToViolatedGrounding' does not appear in its formula grounding.\n")
            end
        end
    end
    
    # Additional check
    # Check 'ruleToViolatedGrounding'
    totalViolation = sum( [ length(x.items) for x in mln.ruleToViolatedGrounding ] )
    @assert( totalViolation == length( mln.violatedFormulaGrounding.items ) )
    # @show( length(mln.rules) )
    # @show( length(mln.ruleToViolatedGrounding) )
    @assert( length(mln.ruleToViolatedGrounding) == length(mln.rules) )
    for i = 1:length(mln.ruleToViolatedGrounding)
        for grounding in mln.ruleToViolatedGrounding[i].items
            @assert( i == grounding.ruleIdx )
        end
    end

    # Check groundings
    numViolatedCheck = 0
    for rule in mln.rules
        violatedFormulaGroundingCheck = FindAllViolationsGivenRule( mln.kb, rule )
        numViolatedCheck += length( violatedFormulaGroundingCheck )
    end
    if length( mln.violatedFormulaGrounding ) != numViolatedCheck
        @printf("num violated: %i, real num violated: %i\n", length(mln.violatedFormulaGrounding), numViolatedCheck)
        error("Check failed, num violated incorrect.\n")
    end

    # Check relations
    for rel in mln.kb.relations
        check!(rel)
    end

    # Check cost
    cost = computeCost(mln)
    if !isapprox( cost, mln.obj )
        @error("objective value not consistent!\n")
    end

    @printf("mln valid!\n")
end
