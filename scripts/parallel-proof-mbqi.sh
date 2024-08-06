###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2024 ETH Zurich.
###############################################################################

#!/bin/bash

TIMEOUT_PER_RUN_SEC=600
AXIOMS_REPO_DIR=$1
BENCHMARKS_DIR=$2
MAIN_SCRIPT="$AXIOMS_REPO_DIR/src/minimizer/main.py"
PIPELINE="proof_mbqi"
OPTIONS="--is_debug=false --timeout $TIMEOUT_PER_RUN_SEC --location"

function get_command() {
	opts="$OPTIONS $1"
	cmd="python $MAIN_SCRIPT $PIPELINE $opts"
	echo $cmd
}

for dir in $BENCHMARKS_DIR/group_*;
do 
	$(get_command $dir) & 
	echo $! >> active_pids.txt; 
done; 

wait;

echo ">>> MBQI tried to generate proofs for all the files. <<<";
rm -f active_pids.txt