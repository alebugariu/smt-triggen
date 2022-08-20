###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2022 ETH Zurich.
###############################################################################

import argparse
import datetime
import re
import subprocess
import time
from os.path import isfile
from typing import Callable, Tuple

import psutil
from func_timeout import FunctionTimedOut, func_timeout

from src.session.exceptions import SmtTimeout, SmtError
from src.session.session import Session

from src.session import debug_mode


class RobustVampireRunner(Session):

    DBG_PREFIX = "RobustVampireRunner"

    def __init__(self,
                 file: str,
                 hard_timeout=600.0,
                 logs_path=None):

        assert isfile(file), "first argument of RobustVampireRunner shoudl be an existing .smt2 file; " \
                             "got %s " % file

        self.file = file
        cmd = ('vampire',
               '--mode', 'casc',
               '--input_syntax', 'smtlib2',
               '--time_limit', '%ds' % round(hard_timeout),
               '--random_seed', '0',
               '--proof', 'off',
               '--statistics', 'none',
               file)

        super().__init__(name="RobustVampireRunner", preamble=[], cmd=cmd,
                         hard_timeout_per_query=hard_timeout, logs_path=logs_path)

    def __enter__(self):
        self.__start_time = datetime.datetime.now()
        self.log_comment("This %s has been started on %s via:" % (self.name, self.__start_time))
        self.log_comment(" ".join(self.cmd))

        self.print_dbg("Running %s via `%s`" % (self.name, " ".join(self.cmd)))

        if debug_mode.flag and self.logs_path:
            self.print_dbg("  the logs will be saved to %s" % self.logs_path)

        self._proc = subprocess.Popen(self.cmd,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT)

        self.print_dbg("@@@ process started at %s"
                       % str(datetime.datetime.now()))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__stop_time = datetime.datetime.now()
        self.log_comment("This %s has been stopped after %.3f seconds."
                         % (self.name, (self.__stop_time - self.__start_time).total_seconds()))

        if debug_mode.flag:
            self.print_dbg("Stopping %s..." % self.name)

        if self.logs_path:
            self.save_to_disk()
            if debug_mode.flag:
                self.print_dbg("  the logs are written to %s" % self.logs_path)

        self.finalize()

    def __vamp_termination_policy(self, line: str or None) -> bool:
        assert line is not None, "expected string, got None"
        if line.find("error") != -1:
            raise SmtError(self.name, line)

        if line.startswith("% SZS status"):
            return True
        if line.startswith("User error"):
            return True
        if line.startswith("% Time elapsed"):
            return True
        if line.startswith("% Refutation not found"):
            return True
        if line.startswith("% Refutation found"):
            return True

        return False

    def run(self) -> Tuple[str, float]:
        """ Run stuff """

        self.__enter__()

        res = "crash"  # should get overridden if operation succeeds
        start_time = time.perf_counter()
        try:
            res = self._readlines(self.__vamp_termination_policy,
                                  self._hard_timeout)
            # print("\n".join(res))
            res = "   ".join(res)
        except SmtTimeout:
            res = "timeout"
        except SmtError as e:
            res = "error: %s" % str(e)
        finally:
            t = time.perf_counter() - start_time
            self.__exit__(None, None, None)

        ind = res.find("Refutation not found")
        if ind != -1:
            res = "unknown"

        ind = res.find("Refutation found")
        if ind != -1:
            res = "unsat"

        ind = res.find("Success in time")
        if ind != -1:
            res = "sat"

        label = "SZS status "
        ind = res.find(label)
        if ind != -1:
            reason = res[ind+len(label):].lower()
            if re.match(r"time", reason):
                res = "timeout"
            elif re.match(r"satisfiable", reason):
                res = "sat"

        return res, t

    def finalize(self, comment=None):
        try:
            parent_pid = self._proc.pid
            try:
                parent_proc = psutil.Process(parent_pid)
            except psutil.NoSuchProcess:
                print(">>>> process %d is already dead." % parent_pid)
                pass
            else:
                for proc in parent_proc.children(recursive=True):
                    print(">>>> killing process with PID %d" % proc.pid)
                    proc.kill()
                print(">>>> killing parent process with PID %d" % parent_pid)
                parent_proc.kill()
        except BrokenPipeError:
            pass
        self.print_dbg("@@@ process finalized at %s"
                       % str(datetime.datetime.now()))
        return self._proc.returncode

    def _sync_line(self):
        line = self._proc.stdout.readline()
        line = line.decode(self.ENCODING)
        line = line.strip()
        return line

    def _read_till_end(self, terminator: Callable[[str], bool]):
        lines = []
        while True:
            line = self._sync_line()
            if not line:
                break
            print("]  " + str(line.encode()))
            lines.append(line)
            t = datetime.datetime.now() - self.__start_time
            self.log_comment("[%s] %s" % (str(t), line))
            if terminator(line):
                break

        return lines

    def _sync_with_policy(self, policy: Callable[[str], bool]):
        return self._read_till_end(policy)

    def _readlines(self, policy: Callable[[str], bool], timeout: float or None) -> str:
        if not timeout:
            timeout = self._hard_timeout
        else:
            timeout = min(self._hard_timeout, timeout)
        try:
            res = func_timeout(timeout, self._sync_with_policy, args=(policy,))
        except FunctionTimedOut:
            if self.logs_path:
                self.save_to_disk()
            self.print_dbg("Timeout after %.3f seconds" % timeout)
            raise SmtTimeout(self.name, timeout)

        return res

    def _add(self, smt_command: str, comment: str or None = None):
        raise Exception("%s does not support incremental solving" % self.DBG_PREFIX)

    def push(self, comment):
        raise Exception("%s does not support incremental solving" % self.DBG_PREFIX)

    def pop(self, comment):
        raise Exception("%s does not support incremental solving" % self.DBG_PREFIX)


def main():

    from os.path import dirname, join

    global debug_mode

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('file', type=str, default='.', help='Path to an SMT files')
    arg_parser.add_argument('timeout', type=int, default=None, help='Hard timeout, in seconds')
    arg_parser.add_argument('--debug', type=lambda x: (str(x).lower() == 'true'), default=debug_mode.flag,
                            help="Whether to run the pipeline in debug mode")

    args = arg_parser.parse_args()
    debug_mode.flag = args.debug

    logs_path = join(dirname(args.file), "vampire.log")

    vamp = RobustVampireRunner(args.file, hard_timeout=args.timeout, logs_path=logs_path)
    result = vamp.run()
    print(result)


if __name__ == "__main__":
    main()
