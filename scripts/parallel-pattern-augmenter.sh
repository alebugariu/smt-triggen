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
MAIN_SCRIPT="$AXIOMS_REPO_DIR/src/frontend/test_runner.py"
ALGORITHM="PatternAugmenter"
OPTIONS="--timeout $TIMEOUT_PER_RUN_SEC --location "

function get_command() {
	opts="$OPTIONS $1"
	cmd="python $MAIN_SCRIPT $ALGORITHM $opts"
	echo $cmd
}

rm -fr group_*/pattern_augmenter/

for dir in group_*; 
do 
	$(get_command $dir) & 
	echo $! >> active_pids.txt; 
done; 

wait; 

echo ">>> Processed all groups. <<<"; 
rm -f active_pids.txt