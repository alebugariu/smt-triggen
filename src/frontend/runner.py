###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2022 ETH Zurich.
###############################################################################

import argparse
from os import listdir
from os.path import join, isfile

from sortedcontainers import SortedSet

from src.frontend.cvc import run_benchmarks_via_cvc
from src.frontend.mbqi import run_benchmarks_via_mbqi
from src.frontend.optset import Optset
from src.frontend.test_runner import run_benchmarks
from src.frontend.vampire_runner import run_benchmarks_via_vampire


from src.session import debug_mode


def main():
    global debug_mode

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('location', type=str, help='Directory with SMT files')
    arg_parser.add_argument('output', type=str, help='File for writing the results into (CSV)')
    arg_parser.add_argument('--timeout', type=int, default=None, help='Timeout for testing one file (sec)')
    arg_parser.add_argument('--debug', type=bool, default=False, help='Debug mode enables detailed output and '
                                                                      'persistent logging')
    args = arg_parser.parse_args()

    debug_mode.flag = args.debug
    location = args.location
    results_file = args.output

    all_files = [join(location, f) for f in listdir(location) if isfile(join(location, f))
                 and f.endswith("smt2")]

    if not args.timeout and len(all_files) < 2:
        timeout = None
        raise_exceptions = True
    else:
        timeout = args.timeout
        raise_exceptions = False

    opt = Optset("GroupsAxiomTester", timeout=timeout, debug=False, disable_preprocessor=True)

    with open(results_file, mode='a') as file:
        file.write("Option set: %s\n" % str(opt))

        legend_o = "Outcome Triggen, Runtime Triggen (sec), Dummy"
        legend_a = "Outcome Vampire, Runtime Vampire (sec)"
        legend_b = "Outcome MBQI, Runtime MBQI (sec)"
        legend_c = "Outcome CVC4-enum-inst, Runtime CVC4-enum-inst (sec)"

        file.write("Example, %s, %s, %s, %s\n" % (legend_o, legend_a, legend_b, legend_c))

    report_o = run_benchmarks(all_files, opt)
    report_a = run_benchmarks_via_vampire(all_files, timeout, raise_exceptions)
    report_b = run_benchmarks_via_mbqi(all_files, timeout, raise_exceptions)
    report_c = run_benchmarks_via_cvc(all_files, timeout, raise_exceptions, True)

    all_tests = SortedSet(report_o.keys()) | \
                SortedSet(report_a.keys()) | \
                SortedSet(report_b.keys()) | \
                SortedSet(report_c.keys())

    csv_rows = []
    for test in all_tests:

        if test in report_o:
            cell_o = ",".join(report_o[test])
        else:
            cell_o = ",".join(["<no info from Triggen>"] * 5)
        if test in report_a:
            cell_a = ",".join(report_a[test])
        else:
            cell_a = ",".join(["<no info from Vampire>"] * 2)
        if test in report_b:
            cell_b = ",".join(report_b[test])
        else:
            cell_b = ",".join(["<no info from MBQI>"] * 2)
        if test in report_c:
            cell_c = ",".join(report_c[test])
        else:
            cell_c = ",".join(["<no info from CVC4-enum-inst>"] * 2)

        row = "%s,%s,%s,%s,%s" % (test, cell_o, cell_a, cell_b, cell_c)
        csv_rows.append(row)

    with open(results_file, mode='a') as file:
        for csv_row in csv_rows:
            file.write("%s\n" % csv_row)

    print("~~~~~~~ Wrote stats to %s ~~~~~~~" % results_file)


if __name__ == "__main__":
    main()
