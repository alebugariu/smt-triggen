###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2022 ETH Zurich.
###############################################################################

#!/bin/bash

BENCHMARKS_DIR=$1
NUM_CORES=26
NUM_FILES=$(find $BENCHMARKS_DIR -maxdepth 1 -name "*.smt2" | wc -l)
N_FILES_IN_GROUP=$(( ($NUM_CORES - 1 + $NUM_FILES) / $NUM_CORES ))  # Round to the nearest integer

i=0; 
for f in $BENCHMARKS_DIR/*.smt2;
do
	d=$BENCHMARKS_DIR/group_$(printf %03d $((i/$N_FILES_IN_GROUP+1)));
	mkdir -p $d;
	cp "$f" $d;
	let i++;
done

NUM_GROUPS=$(ls -1q $BENCHMARKS_DIR | grep "group_" | wc -l)
echo "Distributed $NUM_FILES .smt2 files into $NUM_GROUPS groups."
