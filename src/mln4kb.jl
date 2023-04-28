module mln4kb

using LinearAlgebra
using SparseArrays
using Printf
using Random
using Distributions
using StatsBase

import Base: show, length

include("utils.jl")
include("relation.jl")
include("kb.jl")
include("mln.jl")
include("inference.jl")
include("optimizer.jl")
include("learning.jl")
include("evaluation.jl")

export readFacts, ListDict, add!, remove!, sample, update!
export KB, parseRelations!, GetNumActive, SavePrediction!, WriteRules!
export RelationInRule, Rule, Relation, MLN, CheckMLNValid, PrepareMLN!, Reset!, GetRulesWeights, ExtractFacts, computeCost, FindAllViolationsGivenRule, GetValue
export WalkSAT!, MAPInference!, NumFactInvolved!, OptimizePseudoLogLikelihood!, ComputeDelta, ComputeDeltaOnlyPositive
export readTest, CalculateHitsAtK, EvaluateWithMultipleKBs, Evaluate, Load!, EvaluateReverseTriplt
export SGDOptimizer, AdaGradOptimizer, AdamOptimizer

end # module
