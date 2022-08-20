###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2022 ETH Zurich.
###############################################################################

#!/bin/bash

TIMEOUT_PER_RUN_SEC=600
AXIOMS_REPO_DIR=$1
MAIN_SCRIPT="$AXIOMS_REPO_DIR/src/frontend/cvc4.py"
ENUM_INST=$3
OPTIONS="--timeout $TIMEOUT_PER_RUN_SEC $ENUM_INST"
OUTPUT=$2

function get_command() {
	opts="$OPTIONS $1 $OUTPUT"
	cmd="python $MAIN_SCRIPT $opts"
	echo $cmd
}

for dir in group_*; 
do 
	$(get_command $dir) & 
	echo $! >> active_pids.txt; 
done; 

wait; 

echo ">>> CVC4 processed all groups. <<<";
rm -f active_pids.txt