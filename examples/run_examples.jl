push!(LOAD_PATH, pwd())
using Revise
using mln4kb
using Printf

########################################################
#               Smoker and friends example
########################################################
factFile = "./smoke/facts.txt"
ruleFile = "./smoke/rules.txt"

mln = MLN(factFile, ruleFile);
PrepareMLN!(mln)

# Extract facts
ExtractFacts(mln.kb, "Friends")

# Inference
objList, numViolatedList = WalkSAT!(mln, maxIter=Int(1e2), warmupPeriod=Int(1e2))

# Weight learning
iterate = zeros( Float64, length( mln.rules ) )
lr = 1e-1
optimizer = AdaGradOptimizer(lr, iterate)
OptimizePseudoLogLikelihood!( mln, optimizer, numNegativeSamples=1, maxIter=Int(1e2), resetMLN=false )

# Print rule weights
for rule in mln.rules
    @printf( "weight %3.4f, Rule: %s\n", rule.weight, rule.originalRule )
end



########################################################
#               Kinship example (5000 entities)
########################################################
factFile = "./kinship/facts.txt"
ruleFile = "./kinship/rules.txt"

mln = MLN(factFile, ruleFile);
PrepareMLN!(mln)

# Inference
CheckMLNValid(mln)  # Optional, a sanity check.
objList, numViolatedList = WalkSAT!(mln, maxIter=Int(1e6), warmupPeriod=Int(1e6), check=false) 



########################################################
#               UMLS example
########################################################
factFile = "./UMLS/facts.txt"
ruleFile = "./UMLS/rules.txt"
infoFile = "./UMLS/info.txt"
testFile = "./UMLS/test.txt"


mln = MLN(factFile, ruleFile);
PrepareMLN!(mln)
# Learning weights
iterate = zeros( Float64, length( mln.rules ) )
lr = 1e-1
optimizer = AdaGradOptimizer(lr, iterate)
num = 4  # Set the number of negative samples.
OptimizePseudoLogLikelihood!( mln, optimizer, numNegativeSamples=num, maxIter=Int(1e4), resetMLN=false )

# Save rules with new weights
WriteRules!(mln.rules, "./UMLS/rules_nsample"*string((num))*".txt")


# Inference
ruleFile = "./UMLS/rules_nsample"*string((num))*".txt"  # Load rules with learned weights.
mln = MLN(factFile, ruleFile);
PrepareMLN!(mln)
maxIter = Int(4e5)  # run more iterations can usually yield better result.
WalkSAT!(mln, threshold=0., maxIter=maxIter, warmupPeriod=Int(maxIter*0.8), check=false )


# Evaluation
testFacts, _ = readTest( mln.kb, testFile );
k =[5,10]
_, _, numCorrectPerRelation, scoreList, hitList = Evaluate( testFile, k, mln ) # Need to wait for about 1min to get result.

