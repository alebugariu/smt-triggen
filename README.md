# SMT-Triggen
This repository provides the artifact for the paper: ["Identifying Overly Restrictive Matching Patterns in SMT-Based Program Verifiers"](https://link.springer.com/chapter/10.1007/978-3-030-90870-6_15), A. Bugariu, A. Ter-Gabrielyan, and P. Müller, FM'21. 

Given a formula for which an SMT solver returns *unknown* due to insufficient quantifier instantiations, SMT-Triggen 
(our tool) synthesizes missing triggering terms that enable the solver to refute the formula via E-matching. These
triggering terms allow the users and the developers of program verifiers to remedy the effects of overly
restrictive matching patterns. That is, they can detect and report/fix completeness issues in the verifiers
and soundness errors in their axiomatizations.

The benchmarks used in our evaluation can be found in the folder [benchmarks](/smt-inputs/benchmarks).

# Setup
We provide a [Docker](https://www.docker.com/) container, which already includes all the dependencies. 

## Usage
Download our Docker image:
```
docker pull https://hub.docker.com/r/aterga/smt-triggen:latest
```
or build it from the [Dockerfile](/Dockerfile) provided in our repository (this takes ~1 min):
```
git clone https://github.com/alebugariu/smt-triggen.git smt-triggen
cd smt-triggen
docker build -t aterga/smt-triggen .
```

### Running SMT-Triggen over the benchmarks

Run the 5 configurations of our tool (from Table 1 and Table 2) on a set of benchmarks:
```
docker run -it aterga/smt-triggen /bin/bash -c "scripts/group.sh <path to the benchmarks folder>;
/bin/bash scripts/parallel-stat-collector.sh . <path to the benchmarks folder> <results file.csv>
```
For example, the following command runs the 5 configurations of our tool on the Gobra benchmarks (may take up to 50 min):

```
docker run -it aterga/smt-triggen /bin/bash -c "scripts/group.sh smt-inputs/benchmarks/verifiers/gobra;
/bin/bash scripts/parallel-stat-collector.sh . smt-inputs/benchmarks/verifiers/gobra gobra.csv"
```

### Running Z3 with MBQI over the benchmarks

Run Z3 with MBQI on a set of benchmarks (to generate unsatisfiability proofs):
```
docker run -it aterga/smt-triggen /bin/bash -c "scripts/group.sh <path to the benchmarks folder>;
scripts/parallel-proof-mbqi.sh . <path to the benchmarks folder>"
```

For example, the following command runs Z3 with MBQI on the Gobra benchmarks:
```
docker run -it aterga/smt-triggen /bin/bash -c "scripts/group.sh smt-inputs/benchmarks/verifiers/gobra;
scripts/parallel-proof-mbqi.sh . smt-inputs/benchmarks/verifiers/gobra"
```

### Running CVC4 with enumerative instantation over the benchmarks

Run CVC4 with enumerative instantation on a set of benchmarks (to generate unsatisfiability proofs):
```
docker run -it aterga/smt-triggen /bin/bash -c "scripts/group.sh <path to the benchmarks folder>;
scripts/parallel-proof-cvc4.sh . <path to the benchmarks folder>"
```

For example, the following command runs CVC4 with enumerative instantation on the Gobra benchmarks:
```
docker run -it aterga/smt-triggen /bin/bash -c "scripts/group.sh smt-inputs/benchmarks/verifiers/gobra;
scripts/parallel-proof-cvc4.sh . smt-inputs/benchmarks/verifiers/gobra"
```

### Running Vampire over the benchmarks

Run Vampire on a set of benchmarks (to generate unsatisfiability proofs):
```
docker run -it aterga/smt-triggen /bin/bash -c "scripts/group.sh <path to the benchmarks folder>;
scripts/parallel-proof-vampire.sh . <path to the benchmarks folder>"
```

For example, the following command runs Vampire on the Gobra benchmarks.
```
docker run -it aterga/smt-triggen /bin/bash -c "scripts/group.sh smt-inputs/benchmarks/verifiers/gobra;
scripts/parallel-proof-vampire.sh . smt-inputs/benchmarks/verifiers/gobra"
```
