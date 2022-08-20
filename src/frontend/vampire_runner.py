###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2022 ETH Zurich.
###############################################################################

import multiprocessing
import time
from collections import OrderedDict

from typing import Iterator, Tuple, List, Generator, Dict
from os.path import join, isfile, dirname, basename

from src.algorithms.vampire import Vampire
from src.frontend.test_runner import AxiomatizationTesterTimeout
from src.utils.file import ensure_dir_structure


from src.session import debug_mode


class VampireWorker:
    def __init__(self, vamp: Vampire):
        self.vamp = vamp
        self.result = multiprocessing.Array('c', 10000)

    def run(self):
        try:
            result = self.vamp.run()
            self.result.value = str(result).encode()
        except Exception as e:
            print("$$$$$$$$$$ VampireWorker crashed: %s $$$$$$$$$$$" % str(e))
            self.result.value = ("crash: %s" % str(e)).encode()


def run_benchmarks_via_vampire(tests: Iterator,
                               timeout: int or None,
                               raise_exceptions: bool) -> Dict[str, Tuple[str, str]]:

    assert isinstance(tests, List) or isinstance(tests, Tuple) or isinstance(tests, Generator), \
        "run_benchmarks: argument tests should be Generator, List, or Tuple"

    report = OrderedDict()
    time_info = OrderedDict()
    suite_start_time = time.time()

    for test_num, test in enumerate(tests):
        assert isfile(test), "tests must be existing SMT2 files"

        print("\n-------======= TEST %d out of %d =======-------" % (test_num+1, len(tests)))

        logs_dir = join(dirname(test), "sessions")
        ensure_dir_structure(logs_dir)

        tmp_file_dir = join(dirname(test), "tmp")
        ensure_dir_structure(tmp_file_dir)

        start_time = time.perf_counter()
        try:
            vamp = Vampire(test, tmp_file_dir=tmp_file_dir, logs_dir=logs_dir, hard_timeout=timeout)

            if timeout:
                worker = VampireWorker(vamp)
                p = multiprocessing.Process(target=worker.run)

                p.start()
                p.join(1.05*timeout)

                if p.is_alive():
                    report[test] = "timeout (%ds)" % timeout
                    msg = "%s: Vampire timed out after %.3f seconds" % (test, timeout)
                    p.terminate()
                    p.join()
                    raise AxiomatizationTesterTimeout(msg)

                report[test] = worker.result.value.decode()

            else:
                report[test] = vamp.run()

        except AxiomatizationTesterTimeout as e:
            print(str(e))
            report[test] = "timeout: %s" % str(e)
        except Exception as e:
            print("Vampire crashed: %s" % str(e))
            report[test] = "crash: %s" % str(e)
            if raise_exceptions or debug_mode.flag:
                raise e

        time_info[test] = time.perf_counter() - start_time
        print("======= Finished testing %s with outcome %s after %.3f seconds ======="
              % (basename(test), report[test].upper(), time_info[test]))

    suite_end_time = time.time()

    print("====================================\n"
          "===== Vampire Results Summary ======\n"
          "====================================")

    clean_report: Dict[str, Tuple[str, str]] = dict()
    for (test, outcome), runtime in zip(report.items(), time_info.values()):
        clean_report[basename(test)] = (outcome, str(runtime))

    timeouts = [(test, label) for test, (label, _) in clean_report.items() if label.startswith("timeout")]
    crashes = [(test, label) for test, (label, _) in clean_report.items() if label.startswith("crash") or label.startswith("error")]
    sats = [(test, label) for test, (label, _) in clean_report.items() if label.startswith("sat")]
    unsats = [(test, label) for test, (label, _) in clean_report.items() if label.startswith("unsat")]

    summary = tuple([len(unsats), len(sats), len(timeouts), len(crashes)])

    if timeout:
        print("   Timeout:   %.3f seconds" % timeout)
    else:
        print("   No timeout specified")
    print("   Overall: UNSAT in %d cases, SAT in %d cases, TIMEOUT in %d cases, CRASHED in %d cases" % summary)
    print("   Overall runtime: %.3f seconds" % (suite_end_time - suite_start_time))

    return clean_report
