###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2024 ETH Zurich.
###############################################################################

import argparse
import multiprocessing
import time
from collections import OrderedDict
from os import listdir
from os.path import join, isfile, dirname, basename
from typing import Iterator, Tuple, List, Generator, Dict

from src.frontend.test_runner import AxiomatizationTesterTimeout
from src.minimizer.test_runner import TestRunner
from src.utils.enums import SMTSolver
from src.utils.file import ensure_dir_structure


from src.session import debug_mode


class CvcWorker:
    def __init__(self, cvc: TestRunner, test_file_name: str):
        self.cvc = cvc
        self.test_file_name = test_file_name
        self.result = multiprocessing.Array('c', 10000)

    def run(self):
        try:
            result, _ = self.cvc.run_test(self.test_file_name)
            self.result.value = str(result).encode()
        except Exception as e:
            print("$$$$$$$$$$ %sWorker crashed: %s $$$$$$$$$$$" % (self.cvc.solver, str(e)))
            self.result.value = ("crash: %s" % str(e)).encode()


def run_benchmarks_via_cvc(tests: Iterator,
                           version: int,
                           timeout: int or None,
                           raise_exceptions: bool,
                           enum_inst: bool) -> Dict[str, Tuple[str, str]]:

    assert isinstance(tests, List) or isinstance(tests, Tuple) or isinstance(tests, Generator), \
        "run_benchmarks: argument tests should be Generator, List, or Tuple"

    report = OrderedDict()
    time_info = OrderedDict()
    suite_start_time = time.time()

    solver_name = 'CVC' + str(version)

    for test in tests:
        assert isfile(test), "tests must be existing SMT2 files"

        logs_dir = join(dirname(test), "sessions")
        ensure_dir_structure(logs_dir)

        tmp_file_dir = join(dirname(test), "tmp")
        ensure_dir_structure(tmp_file_dir)

        start_time = time.perf_counter()
        try:
            solver = None
            if version == 4:
                solver = SMTSolver.CVC4
            elif version == 5:
                solver = SMTSolver.CVC5
            cvc = TestRunner(solver, timeout=timeout, enum_inst=enum_inst)

            if timeout:
                worker = CvcWorker(cvc, test)
                p = multiprocessing.Process(target=worker.run)

                p.start()
                p.join(timeout)

                if p.is_alive():
                    report[test] = "timeout (%ds)" % timeout
                    msg = "%s: %s timed out after %.3f seconds" % (test, solver_name, timeout)
                    p.terminate()
                    p.join()
                    raise AxiomatizationTesterTimeout(msg)

                report[test] = worker.result.value.decode()

            else:
                result = cvc.run_test(test)
                report[test] = result

        except AxiomatizationTesterTimeout as e:
            print(str(e))
            report[test] = "timeout: %s" % str(e)
        except Exception as e:
            print("%s crashed: %s" % str(solver_name, e))
            report[test] = "crash: %s" % str(e)
            if raise_exceptions or debug_mode.flag:
                raise e

        time_info[test] = time.perf_counter() - start_time
        print("Finished testing %s after %.3f seconds"
              % (test, time_info[test]))

    suite_end_time = time.time()

    print("=================================\n"
          "===== %s Results Summary ======\n"
          "=================================" % solver_name)

    clean_report: Dict[str, Tuple[str, str]] = dict()
    for (test, outcome), runtime in zip(report.items(), time_info.values()):
        clean_report[basename(test)] = (outcome, str(runtime))

    timeouts = [(test, label) for test, (label, _) in clean_report.items() if label.startswith("timeout")]
    crashes = [(test, label) for test, (label, _) in clean_report.items() if label.startswith("crash")]
    sats = [(test, label) for test, (label, _) in clean_report.items() if label.startswith("sat")]
    unsats = [(test, label) for test, (label, _) in clean_report.items() if label.startswith("unsat")]
    unknowns = [(test, label) for test, (label, _) in clean_report.items() if label.startswith("unknown")]

    summary = tuple([len(unsats), len(sats), len(unknowns), len(timeouts), len(crashes)])

    if timeout:
        print("   Timeout:   %.3f seconds" % timeout)
    else:
        print("   No timeout specified")
    print("   Overall: UNSAT in %d cases, SAT in %d cases, UNKNOWN in %d cases, "
          "TIMEOUT in %d cases, CRASHED in %d cases" % summary)
    print("   Overall runtime: %.3f seconds" % (suite_end_time - suite_start_time))

    return clean_report


def main():
    global debug_mode

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('location', type=str, help='Directory with SMT files')
    arg_parser.add_argument('output', type=str, help='File for writing the results into (CSV)')
    arg_parser.add_argument('--version', type=int, choices=[4, 5], default=None, help='Major version (4 or 5)')
    arg_parser.add_argument('--timeout', type=int, default=None, help='Timeout for testing one file (sec)')
    arg_parser.add_argument('--debug', type=bool, default=False, help='Debug mode enables detailed output and persistent logging')
    arg_parser.add_argument('--enumerative', type=bool, default=False, help='Use enumerative instantiation when E-matching saturates')

    args = arg_parser.parse_args()

    debug_mode.flag = args.debug
    location = args.location
    results_file = args.output
    enum_inst = args.enumerative
    version = args.version

    all_files = [join(location, f) for f in listdir(location) if isfile(join(location, f))
                 and f.endswith("smt2")]

    if not args.timeout and len(all_files) < 2:
        timeout = None
        raise_exceptions = True
    else:
        timeout = args.timeout
        raise_exceptions = False

    with open(results_file, mode='a') as file:
        file.write("Example, Outcome, Runtime (sec), Time limit\n")

    report = run_benchmarks_via_cvc(all_files, version, timeout, raise_exceptions, enum_inst)
    csv_rows = ["%s,%s,%s" % (entry, ",".join(report[entry]), str(timeout)) for entry in report]

    with open(results_file, mode='a') as file:
        for csv_row in csv_rows:
            file.write("%s\n" % csv_row)

    print("~~~~~~~ Wrote stats to %s ~~~~~~~" % results_file)


if __name__ == "__main__":
    main()
