## MLN4KB: an efficient Markov logic network engine for large-scale knowledge bases and structured logic rules

### Introduction
This repository contains the code for our paper: 

**Title:** MLN4KB: an efficient Markov logic network engine for large-scale knowledge bases and structured logic rules.

**Authors:** Huang Fang, Yang Liu, Yunfeng Cai, Mingming Sun.

**Affiliation:** Baidu Research, Cognitive Computing Lab (CCL).


### Quick start
Open Julia in terminal under this folder and go to the package REPL by pressing `]`, type `activate .` to activate the package. Then go back to Julia REPL by pressing the backslash.

The "smokers and friends" toy example:

**Load packages**:
```
push!(LOAD_PATH, pwd())
using Revise, mln4kb, Printf
```

**MLN inference**:
```
factFile = "./examples/smoke/facts.txt"
ruleFile = "./examples/smoke/rules.txt"

mln = MLN(factFile, ruleFile);
PrepareMLN!(mln)

# Extract facts
ExtractFacts(mln.kb, "Friends")

# Inference
objList, numViolatedList = WalkSAT!(mln, maxIter=Int(1e2), warmupPeriod=Int(1e2))
```

**Weight learning**:
```
iterate = zeros( Float64, length( mln.rules ) )
lr = 1e-1
optimizer = AdaGradOptimizer(lr, iterate)
OptimizePseudoLogLikelihood!( mln, optimizer, numNegativeSamples=1, maxIter=Int(1e2), resetMLN=false )
```

More test examples can be found in `./examples/run_examples.jl`.

### Citation
If you find this project helpful, please cite the code with the following bibtex.
```
@inproceedings{fang2023mln4kb,
  title={MLN4KB: an efficient Markov logic network engine for large-scale knowledge bases and structured logic rules},
  author={Huang Fang, Yang Liu, Yunfeng Cai, Mingming Sun},
  booktitle={The International World Wide Web Conference 2023},
  year={2023}
}
```

### Contact
Please feel free to send your comments and contact us by `fangazq877@gmail.com`. We are considering to develop a C/C++ version of MLN4KB (with multi-CPU parallelization), please let us know if you find MLN4KB.jl is still too slow for your application.


