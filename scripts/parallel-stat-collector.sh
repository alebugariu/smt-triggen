###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2022 ETH Zurich.
###############################################################################

#!/bin/bash

AXIOMS_REPO_DIR=$1
BENCHMARKS_DIR=$2
MAIN_SCRIPT="$AXIOMS_REPO_DIR/src/frontend/stat_collector.py"
OUTPUT=$3

function get_command() {
	opts="$1 $OUTPUT"
	cmd="python $MAIN_SCRIPT $opts"
	echo $cmd
}

for dir in $BENCHMARKS_DIR/group_*;
do 
	$(get_command $dir) & 
	echo $! >> active_pids.txt; 
done; 

wait; 

echo ">>> Processed all groups. <<<"; 
rm -f active_pids.txt