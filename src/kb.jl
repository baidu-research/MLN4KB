########################################################################
#                       Struct for Knowledge Base
########################################################################
mutable struct KB
    numEntities::Int64
    numRelations::Int64
    entityMap::Dict{String, Int64}
    relationMap::Dict{String, Int64}
    reverseEntityMap::Dict{Int64, String}
    reverseRelationMap::Dict{Int64, String}
    observedFacts::Array{Tuple{Int64, Int64, Int64, Int64}}
    relations::Array{Relation}
    function KB(
        triplets::Array{Tuple{String, String, String, Int64}};
        verbose = true
    )
        observedFacts = Array{Tuple{Int64, Int64, Int64, Int64}}([])
        entityMap = Dict{String, Int64}()
        relationMap = Dict{String, Int64}()
        reverseEntityMap = Dict{Int64, String}()
        reverseRelationMap = Dict{Int64, String}()
        # Keep track of of number of entity and relation
        entityIdx = 1
        relationIdx = 1
        for (e1, r, e2, val) in triplets
            if r[1] == '!'
                @assert( val == 0 )
                r = r[2:end]
            end
            if !haskey(entityMap, e1)
                entityMap[e1] = entityIdx
                entityIdx += 1
            end
            if !haskey(entityMap, e2)
                entityMap[e2] = entityIdx
                entityIdx += 1
            end
            if !haskey(relationMap, r)
                relationMap[r] = relationIdx
                relationIdx += 1
            end
            # Record observed facts
            push!( observedFacts, ( entityMap[e1], relationMap[r], entityMap[e2], val ) )
        end
        numEntities = entityIdx-1
        numRelations = relationIdx-1
        if verbose
            @printf("# entities: %8i, # relations: %8i, # facts: %8i\n", numEntities, numRelations, length(observedFacts))
        end
        # Construct reverse mapping
        for key in keys(entityMap)
            val = entityMap[key]
            reverseEntityMap[ val ] = key
        end
        for key in keys(relationMap)
            val = relationMap[key]
            reverseRelationMap[ val ] = key
        end
        # Construct empty relations
        relations = Array{Relation}([])
        for i = 1:numRelations
            newRelation = Relation(i)
            push!(relations, newRelation)
        end
        # parseRelations!(kb)
        # @printf("Relations in KB constucted successfully!\n")
        new(numEntities, numRelations, entityMap, relationMap, reverseEntityMap, reverseRelationMap, observedFacts, relations)
    end
end

function parseRelations!(
    kb::KB
)
    # Add evidences.
    for (e1, r, e2, val) in kb.observedFacts
        add!( kb.relations[r], (e1, r, e2), val, true)
    end
    return nothing
end

function Reset!(
    kb::KB
)
    numRelations = length( kb.relations )
    for i = 1:numRelations
        singleEntity = kb.relations[i].singleEntity
        kb.relations[i] = Relation(i)
        kb.relations[i].singleEntity = singleEntity
    end
    parseRelations!( kb )
    return nothing
end

function Base.show(
    kb::KB
)
    @printf("\n\t# entities: %8i\n", kb.numEntities)
    @printf("\t# relations: %8i\n", kb.numRelations)
    @printf("\t# observed facts: %8i\n", length(kb.observedFacts))
end

# Extract facts from KB.
function ExtractFacts(
    kb::KB,
    relName::String
)
    if !haskey( kb.relationMap, relName )
        error("KB does not have relation: ", relName)
    end
    relIdx = kb.relationMap[relName]
    rel = kb.relations[relIdx]
    ret = Array{Tuple{String,String,String}}([])
    for e1 in keys( rel.firstArgsNeighbours )
        for e2 in rel.firstArgsNeighbours[e1].items
            e1Name = kb.reverseEntityMap[ e1 ]
            e2Name = kb.reverseEntityMap[ e2 ]
            push!( ret, (e1Name, relName, e2Name) )
        end
    end
    return ret
end

function GetNumActive(
    kb::KB
)
    for rel in kb.relations
        UpdateNumActive!( rel )
    end
    return [ rel.numActive for rel in kb.relations ]
end

# Deep copy of KB, used for record the best KB during optimziation
function Base.copy(
    kb::KB
)
    triplets = Array{Tuple{String, String, String, Int64}}([])
    kbCpy = KB( triplets, verbose=false )
    
    kbCpy.numEntities = deepcopy( kb.numEntities )
    kbCpy.numRelations = deepcopy( kb.numRelations )
    kbCpy.entityMap = deepcopy( kb.entityMap )
    kbCpy.relationMap = deepcopy( kb.relationMap )
    kbCpy.reverseEntityMap = deepcopy( kb.reverseEntityMap )
    kbCpy.reverseRelationMap = deepcopy( kb.reverseRelationMap )
    kbCpy.observedFacts = deepcopy( kb.observedFacts )

    kbCpy.relations = Array{Relation}([])
    for rel in kb.relations
        push!( kbCpy.relations, copy( rel ) )
    end

    return kbCpy
end


########################################################################
#                       Struct for Rules
########################################################################

# Struct for rules
mutable struct Rule
    originalRule::String
    numRelations::Int64
    weight::Float64
    relations::Array{RelationInRule}
    numVars::Int64
    activeEntitiesPerVal::Array{Array{Int64}}    # activeEntitiesPerVal[i] stores the active entities for the i-th var.
    adjList::Dict{Int64, Array{Int64}}
    function Rule(
        kb::KB,
        rule::String
    )
        # Predicates should be seperated by space
        line = split(rule, " ")
        originalRule = join( line[2:end], " ")
        numNegation = 0
        weight = parse(Float64, line[1] )
        relations = Array{RelationInRule}([])
        # varSet = Set{String}([])
        varIdxDict = Dict{String, Int64}()
        varIdx = 1
        # println(rule)
        for i = 2:length(line)
            if !occursin("(", line[i]) && line[i] != "v"
                @printf("Rule not valid!\n")
                @printf("%s\n", rule)
                @assert(1==2)
                # exit(0)
            elseif occursin("(", line[i])
                # Parse rule
                curRelation = String(line[i])
                curNegation = (line[i][1] == '!')
                # println(curNegation)
                if curNegation
                    curRelation = curRelation[2:end]
                    numNegation += 1
                end
                # Parse relation
                curRelation = split(curRelation, r"[(,)]")
                curRelation = [ x for x in curRelation if length(x) != 0 ]
                curRelation = [ replace(x, " "=>"") for x in curRelation ]
                relName, valName1, varName2 = "", "", ""
                if length(curRelation) == 2
                    relName, varName1 = curRelation
                    varName2 = varName1
                elseif length(curRelation) == 3
                    relName, varName1, varName2 = curRelation
                else
                    @printf("invalid relation: %s\n", line[i])
                    error("invalid relation")
                end
                if !haskey(kb.relationMap, relName)
                    # @warn("unobserved relation:", relName)
                    numRel = length( kb.relations )
                    kb.relationMap[relName] = numRel+1
                    kb.numRelations += 1
                    newRelation = Relation(numRel+1)
                    if varName1 == varName2
                        newRelation.singleEntity = true
                    end
                    push!( kb.relations, newRelation )
                    # exit(0)
                # Check if single entity.
                elseif varName1 == varName2
                    kb.relations[ kb.relationMap[relName] ].singleEntity = true
                end
                relIdx = kb.relationMap[relName]
                # store variables, map variable to idx
                if !haskey(varIdxDict, varName1)
                    varIdxDict[varName1] = varIdx
                    varIdx += 1
                end
                if !haskey(varIdxDict, varName2)
                    varIdxDict[varName2] = varIdx
                    varIdx += 1
                end
                newRelation = RelationInRule( varIdxDict[varName1], varIdxDict[varName2], relIdx, curNegation )
                # Add new relation to relation list
                push!(relations, newRelation)
            else
                @assert( line[i] == "v" || line[i] == "" )
                continue
            end    
        end
        if numNegation == 0
            @printf("Invalid rule, no negation\n")
            @printf("%s\n", rule)
            assert(1==2)
            # exit(0)
        end
        # Re-order relations, negation before positive.
        sort!( relations, rev=true, by = x -> x.negation )

        # Check number of variables.
        distinctVars = Set([])
        for relation in relations
            push!( distinctVars, relation.firstArg )
            push!( distinctVars, relation.secondArg )
        end
        activeEntitiesPerVal = Array{Set{Int64}}([])

        # Construct ajacency list.
        adjList = ConstructAdjListForRule(relations)

        new( originalRule, length(relations), weight, relations, length(distinctVars), activeEntitiesPerVal, adjList )
    end
end

# Check if the relations in a given rule are all closed
function RuleIsClosed(
    kb::KB,
    rule::Rule
)
    allClosed = true
    for rel in rule.relations
        relIdx = rel.idx
        if !kb.relations[relIdx].closed
            allClosed = false
            break
        end
    end
    return allClosed
end

# Subroutine to construct adjacency list.
function ConstructAdjListForRule(
    relations::Array{RelationInRule}
)
    ret = Dict{Int64,Array{Int64}}()
    var2relidx = Dict{Int64, Array{Int64}}()
    # Collect information: var2relidx
    for i = 1:length(relations)
        rel = relations[i]
        var1 = rel.firstArg
        var2 = rel.secondArg
        if !haskey( var2relidx, var1 )
            var2relidx[var1] = Array{Int64}([])
        end
        if !haskey( var2relidx, var2 )
            var2relidx[var2] = Array{Int64}([])
        end
        push!( var2relidx[var1], i )
        push!( var2relidx[var2], i )
    end

    # Construct result
    for i = 1:length(relations)
        rel = relations[i]
        ret[i] = Array{Int64}([])
        var1 = rel.firstArg
        var2 = rel.secondArg
        for j in var2relidx[var1]
            if !(j in ret[i])
                push!( ret[i], j )
            end
        end
        for j in var2relidx[var2]
            if !(j in ret[i])
                push!( ret[i], j )
            end
        end
    end
    return ret
end


function checkRules!(
    rules::Array{Rule}
)
    for i = 1:length(rules)
        rule = rules[i]
        numVars = rule.numVars
        assinged = zeros(Int64, numVars)
        visited = Set{Int64}()
        for rel in rule.relations
            if rel.negation 
                var1, var2 = rel.firstArg, rel.secondArg
                push!( visited, var1 )
                push!( visited, var2 )
            end
        end
        for rel in rule.relations
            if !rel.negation
                var1, var2 = rel.firstArg, rel.secondArg
                if !( var1 in visited ) || !(var2 in visited)
                    error("invalid rule: ", i)
                end
            end
        end
    end
    return nothing
end

# Save result
function SavePrediction!(
    kb::KB,
    filename::String
)
    # Store positive facts that exclude evidences.
    f = open(filename, "w")
    for relName in keys(kb.relationMap)
        relIndex = kb.relationMap[relName];        
        headList = keys( kb.relations[relIndex].firstArgsNeighbours );
        for head in headList
            tailList = kb.relations[relIndex].firstArgsNeighbours[head].items
            for tail in tailList
                @assert( tail in keys( kb.relations[relIndex].secondArgsNeighbours ) && head in keys(kb.relations[relIndex].secondArgsNeighbours[tail].itemToPosition ) )
                if (head, tail) in kb.relations[relIndex].evidences
                    continue
                end
                write(f, kb.reverseEntityMap[head])
                write(f, "\t")
                write(f, relName)
                write(f, "\t")
                write(f, kb.reverseEntityMap[tail])
                write(f, "\n")
            end
        end
    end
    close(f)
    return nothing
end

# Store all rules
function WriteRules!(
    rules::Array{Rule},
    output::String
)
    f = open(output, "w")
    for rule in rules
        line = string(rule.weight)*" "*rule.originalRule
        println(f, line)
    end
    close(f)
    return nothing
end

# Construct active entities
function ConstructActiveEntities(
    kb::KB,
    rule::Rule
)
    numVars = rule.numVars
    rule.activeEntitiesPerVal = Array{Set{Int64}}([])
    for i = 1:numVars
        push!(rule.activeEntitiesPerVal, Set{Int64}([]))
    end

end

# Show 
function Base.show(
    rule::Rule
)
    n = length(rule.relations)
    @printf("\n\t")
    for i = 1:n
        rel = rule.relations[i]
        neg = rel.negation
        idx = rel.idx
        arg1 = rel.firstArg
        arg2 = rel.secondArg
        if neg
            @printf("!")
        end
        @printf("R%i(%i, %i)", idx, arg1, arg2)
        if i != n
            @printf(" v ")
        end
    end
    @printf("\n")
end

