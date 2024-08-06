###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2024 ETH Zurich.
###############################################################################

import argparse
import random
import time
import os
from os import listdir
from os.path import join, isfile

from src.frontend.test_runner import run_benchmarks
from src.frontend.optset import Optset


# Suggestions on how to make Python reproducible taken from:
# https://medium.com/@ODSC/properly-setting-the-random-seed-in-ml-experiments-not-as-simple-as-you-might-imagine-219969c84752
# Set a seed value
from src.session.random_seed import SEED

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(SEED)

# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(SEED)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('location', type=str, help='Directory with SMT files')
    arg_parser.add_argument('output', type=str, help='File for writing the results into (CSV)')
    args = arg_parser.parse_args()

    location = args.location
    results_file = args.output

    all_files = [join(location, f) for f in listdir(location) if isfile(join(location, f))
                 and f.endswith("smt2")]

    opts = [
        Optset(),
        Optset(similarity_threshold=0.1),
        Optset(batch_size=1),
        Optset(type_constraints=True),
        Optset(top_level=False, similarity_threshold=0.1)
    ]

    with open(results_file, mode='a') as file:
        file.write("Configuration No,Test file,Outcome,Runtime (sec),Dummy,Final similarity threshold,%s\n" %
                   ",".join(vars(opts[-1]).keys()))

    start_time = time.time()
    for num, opt in enumerate(opts):
        print("Configuration #%d/%d" % (num+1, len(opts)))
        report = run_benchmarks(all_files, opt)
        csv_rows = ["%s,%s,%s" % (entry, ",".join(report[entry]), ",".join(map(str, vars(opt).values())))
                    for entry in report]

        with open(results_file, mode='a') as file:
            for csv_row in csv_rows:
                file.write("%d,%s\n" % (num, csv_row))

    print("%%%%%%%%%%%% All statistics written to %s %%%%%%%%%%%%" % results_file)
    print("%%%%%%%%%%%% Total runtime:  %.1f seconds %%%%%%%%%%%%" % (time.time() - start_time))


if __name__ == "__main__":
    main()
