###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2022 ETH Zurich.
###############################################################################

import argparse
import multiprocessing
import time
from collections.abc import Generator
from collections import OrderedDict
from os import listdir
from os.path import isfile, join, dirname, basename, getsize
from typing import List, Tuple, Iterator, Dict
from src.frontend.optset import Optset, Algorithms
from src.frontend.util import Camel2_snake
from src.utils.file import ensure_dir_structure

from src.session import debug_mode

########## do not remove these lines!
from src.algorithms.groups import GroupsAxiomTester
from src.algorithms.pattern_augmenter import PatternAugmenter
##########


class AxiomatizationTesterTimeout(Exception):
    pass


class AxiomTestingWorker:

    def __init__(self, tester: GroupsAxiomTester):
        self.t = tester
        self.result = multiprocessing.Array('c', 100000)

    def run(self):
        try:
            res = self.t.run()
            self.result.value = str(res).encode()
        except Exception as e:
            self.t.print_dbg("$$$$$$$$$$ crashed! $$$$$$$$$$$")
            self.result.value = ("crash (%s)" % str(e)).encode()


def run_benchmarks(tests: Iterator, opt: Optset) -> Dict[str, Tuple[str, str, str, str]]:

    assert isinstance(tests, List) or isinstance(tests, Tuple) or isinstance(tests, Generator), \
        "run_benchmarks: argument tests should be Generator, List, or Tuple"

    algo_class = globals()[opt.algorithm]

    report = OrderedDict()
    time_info = OrderedDict()
    suite_start_time = time.time()

    for test in tests:
        assert isfile(test), "tests must be existing SMT2 files"

        logs_dir = join(dirname(test), "sessions")
        ensure_dir_structure(logs_dir)

        tmp_file_dir = join(dirname(test), "tmp")
        ensure_dir_structure(tmp_file_dir)

        start_time = time.perf_counter()
        try:
            if opt.algorithm == "GroupsAxiomTester":
                t = algo_class(test, logs_dir=logs_dir, opt=opt)
            else:
                assert not opt.disable_preprocessor, "doesn't make sense to disable preprocessing in the preprocessor"
                pipeline = Camel2_snake(opt.algorithm)
                target_dir = join(dirname(test), pipeline)
                ensure_dir_structure(target_dir)

                t = algo_class(filename=test, target_dir=target_dir,
                               logs_dir=logs_dir, debug=debug_mode.flag,
                               tmp_file_dir=tmp_file_dir)

            if opt.timeout:
                worker = AxiomTestingWorker(t)
                p = multiprocessing.Process(target=worker.run)

                p.start()
                p.join(opt.timeout)

                if p.is_alive():
                    report[test] = "timeout (%ds)" % opt.timeout
                    msg = "%s: testing `%s` timed out after %.3f seconds" \
                          % (test, opt.algorithm, opt.timeout)
                    p.terminate()
                    p.join()
                    raise AxiomatizationTesterTimeout(msg)

                report[test] = worker.result.value.decode()

            else:
                res = t.run()
                report[test] = res

            t.cleanup()

        except AxiomatizationTesterTimeout as e:
            print(str(e))
            report[test] = "timeout: %s" % str(e)
        except Exception as e:
            print("%s crashed: %s" % (opt.algorithm, str(e)))
            report[test] = "crash: %s" % str(e)

        time_info[test] = time.perf_counter() - start_time
        print("Finished testing %s after %.3f seconds"
              % (test, time_info[test]))

    suite_end_time = time.time()

    print("====================================\n"
          "===== Triggen Results Summary ======\n"
          "====================================")

    if opt.algorithm == "PatternAugmenter":
        clean_report: Dict[str, Tuple[str, str, str, str]] = dict()
        for (test, res), runtime in zip(report.items(), time_info.values()):

            if isinstance(res, str) and (res.startswith("timeout") or res.startswith("crash")):
                if res.startswith("timeout"):
                    print("`%s`: not augmented with patterns due to a timeout after %.3f seconds" % (test, runtime))
                    label = "timeout"
                else:
                    print("`%s`: not augmented with patterns due to a crash after %.3f seconds" % (test, runtime))
                    print("  %s" % res)
                    label = "crash"

            else:
                if res == "True":
                    label = "success"
                else:
                    label = "not_augmented"

            clean_report[basename(test)] = (label, str(runtime), "")

        timeouts = [(test, label) for test, (label, _, _) in clean_report.items() if label == "timeout"]
        crashes = [(test, label) for test, (label, _, _) in clean_report.items() if label == "crash"]
        successes = [(test, label) for test, (label, _, _) in clean_report.items() if label == "success"]
        not_augmented = [(test, label) for test, (label, _, _) in clean_report.items() if label == "not_augmented"]

        summary = tuple([len(successes), len(not_augmented), len(timeouts), len(crashes)])

        print("   Overall: augmented %d examples with patterns, not augmented (no suitable patterns) %d examples, "
              "timed out in %d cases, crashed in %d cases" % summary)
        print("   Overall runtime: %.3f seconds" % (suite_end_time - suite_start_time))

        return clean_report

    else:
        for (test, outcome), runtime in zip(report.items(), time_info.values()):
            print("`%s`: %s runtime: %.3f seconds" % (test, outcome, runtime))

        inconsistencies = [(test, outcome) for test, outcome in report.items() if str(outcome).startswith("(True")]
        gaveups = [(test, outcome) for test, outcome in report.items() if str(outcome).startswith("(False")]
        timeouts = [(test, outcome) for test, outcome in report.items() if str(outcome).startswith("timeout")]
        crashes = [(test, outcome) for test, outcome in report.items() if str(outcome).startswith("crash")]

        summary = tuple([len(inconsistencies), len(gaveups), len(timeouts), len(crashes)])

        print("   Opset:   %s" % opt)
        print("   Overall: found %d inconsistencies, gave up in %d cases, timed out in %d cases, crashed in %d cases" %
                  summary)
        print("   Overall runtime: %.3f seconds" % (suite_end_time-suite_start_time))

        clean_report: Dict[str, Tuple[str, str, str, str]] = dict()
        for (test, outcome), runtime in zip(report.items(), time_info.values()):
            dummy = "<no triggering terms>"
            if str(outcome).startswith("(True"):
                label = "detected_inconsistency"
                dummy = outcome[len("(True, ")+1:-6]
                final_sigma = outcome[-4:-1]
            elif str(outcome).startswith("(False"):
                label = "gaveup"
                final_sigma = outcome[-4:-1]
            elif str(outcome).startswith("timeout"):
                label = "timeout"
                final_sigma = opt.similarity_threshold
            else:
                label = "crash: %s" % outcome
                final_sigma = opt.similarity_threshold
            clean_report[basename(test)] = (label, str(runtime), dummy, str(final_sigma))

        return clean_report


def main():

    def restricted_float(x):
        try:
            x = float(x)
        except ValueError:
            raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

        if x < 0.0 or x > 1.0:
            raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % (x,))
        return x

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('algorithm', type=str, choices=Algorithms, help="Which algorithm to run")
    arg_parser.add_argument('--timeout', type=int, default=None, help='Hard timeout per file')
    arg_parser.add_argument('--location', type=str, default='.', help='SMT file or directory with SMT files')
    arg_parser.add_argument('--debug', type=lambda x: (str(x).lower() == 'true'), default=debug_mode.flag,
                            help="Whether to run the pipeline in debug mode")
    arg_parser.add_argument('--similarity_threshold', type=restricted_float, default=0.3,
                            help='Similarity threshold used by the Jaccard algorithm from [0, 1]')
    arg_parser.add_argument('--max_different_models', type=int, default='4',
                            help='Max number of different models per test')
    arg_parser.add_argument('--max_depth', type=int, default='2',
                            help='Max depth for clustering in the groups algorithm')
    arg_parser.add_argument('--max_axiom_frequency', type=int, default='1',
                            help='Max number of times an axiom should be included in the same cluster '
                                 'in the groups algorithm')
    arg_parser.add_argument('--max_distributivity', type=int, default='1',
                            help='Max number of times we should apply distributivity of | over &')
    arg_parser.add_argument('--max_formulas', type=int, default='100',
                            help='Max formulas which should be synthesized for each conjunct')
    arg_parser.add_argument('--multiple_dummies', type=lambda x: (str(x).lower() == 'true'), default=False,
                            help='Find multiple dummies, instead of stopping at the first one')
    arg_parser.add_argument('--type_constraints', type=lambda x: (str(x).lower() == 'true'), default=False,
                            help='Whether type-based constraints should be used during unification')
    arg_parser.add_argument('--top_level', type=lambda x: (str(x).lower() == 'true'), default=True,
                            help='Whether only top level functions should be unified')
    arg_parser.add_argument('--diverse_models', type=lambda x: (str(x).lower() == 'true'), default=True,
                            help='Whether to use a heuristic for getting more diverse models')
    arg_parser.add_argument('--batch_size', type=int, default='64',
                            help='(Max) number of dummies that should be tested together in batch mode')
    arg_parser.add_argument('--disable_preprocessor', type=lambda x: (str(x).lower() == 'true'), default=False,
                            help='Whether Skolemization and pattern augmentation should be run before the algorithm')
    arg_parser.add_argument('--keep_duplicate_assertions', type=lambda x: (str(x).lower() == 'true'), default=False,
                            help='Whether duplicate assertions should be kept')

    args = arg_parser.parse_args()
    opt = Optset().set(vars(args))
    debug_mode.flag = opt.debug

    location = args.location
    timeout = args.timeout

    if isfile(location):
        if opt.debug and opt.timeout:
            print("+===============================+\n"
                  "| Warning:                      |\n"
                  "|  you probably should not set  |\n"
                  "|  --timeout and --debug=True   |\n"
                  "|  at the same time.            |\n"
                  "+===============================+")
        opt.raise_exceptions = True
        run_benchmarks((location,), opt)
    else:
        all_files_sizes = [(join(location, f), getsize(join(location, f))) for f in listdir(location)
                           if isfile(join(location, f)) and f.endswith("smt2")]
        # Sort by size: smaller files go first
        all_files = [fpath for fpath, _ in sorted(all_files_sizes, key=lambda item: item[1], reverse=False)]

        if not timeout:
            timeout = 600  # we always set a timeout for batch mode
            opt.timeout = timeout

        run_benchmarks(all_files, opt)


if __name__ == "__main__":
    main()
