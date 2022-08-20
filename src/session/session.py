###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2022 ETH Zurich.
###############################################################################

import datetime
import psutil
import re
import subprocess
import time
from typing import List, Callable, Union, Tuple, Dict, Set
from func_timeout import func_timeout, FunctionTimedOut

# ATTENTION:
#  Do not remove the following imports:
#  They synchronize global variables across different modules
from src.session import debug_mode
from src.session.exceptions import SmtError, SmtTimeout, SessionApiError
from src.utils.preamble import STANDARD_PREAMBLE


class Session:

    DBG_PREFIX = "Session"
    ENCODING = "utf-8"

    def print_dbg(self, msg):
        if debug_mode.flag:
            print("%s.DBG >> %s" % (self.name, msg))

    def print(self, msg):
        print("%s >> %s" % (self.name, msg))

    def __str__(self):
        return "\n".join(self._log)

    def __repr__(self):
        return "\n".join(self._log)

    def __write_log_impl(self, line: str):
        self._log.append(self._tab_level * "  " + line)

    def log_cmd(self, cmd: str):
        self.__write_log_impl(cmd)

    def log_comment(self, comment: str):
        line = "; %s" % comment
        self.__write_log_impl(line)

    def log_with_comment(self, msg: str, comment: str):
        line = "%s  ; %s" % (msg, comment)
        self.__write_log_impl(line)

    def log(self, cmd: str or None, comment: str or None):
        if cmd and comment:
            self.log_with_comment(cmd, comment)
        elif cmd:
            self.log_cmd(cmd)
        elif comment:
            self.log_comment(comment)
        else:
            raise SessionApiError("Session.log: at least one argument must be not None")

    def get_z3_log(self):
        # The assumption is that the comments are not submitted to Z3, and that is the only difference.
        return list(filter(lambda x: not x.startswith(";"), map(lambda x: x.strip(), self._log)))

    def __init__(self,
                 name: str,
                 preamble=STANDARD_PREAMBLE,
                 cmd=('z3', '-smt2', '-t:1000', '-in'),
                 hard_timeout_per_query=1.0,
                 memory_limit_mb: int or None = 6000,
                 logs_path=None):

        self.name = name
        self.cmd = cmd
        self._log = []
        self._hard_timeout = hard_timeout_per_query
        self._memory_limit = memory_limit_mb
        self.logs_path = logs_path

        self._proc = None

        if memory_limit_mb:
            proper_preamble = []
            found_memory_setting = False
            last_setting_ind = -1
            memory_setting = "(set-option :memory_max_size %d)" % memory_limit_mb
            for ind, cmd in enumerate(preamble):
                if cmd.startswith("(set-option"):
                    last_setting_ind = ind
                if re.match("\(set-option\s+\:memory_max_size", cmd):
                    found_memory_setting = True
                    proper_preamble.append(memory_setting)
                else:
                    proper_preamble.append(cmd)

            if not found_memory_setting:
                proper_preamble.insert(last_setting_ind+1, memory_setting)

            self._preamble = proper_preamble

        self._preamble = preamble
        self._tab_level = 0
        self._query_counter = 0

        self.__last_query_duration: int or None = None
        self.__start_time = None
        self.__stop_time = None

    def __enter__(self):
        self.__start_time = datetime.datetime.now()
        self.log_comment("This %s has been started on %s via `%s`"
                         % (self.name, self.__start_time, " ".join(self.cmd)))

        self.print_dbg("Starting %s via `%s`" % (self.name, " ".join(self.cmd)))

        if debug_mode.flag and self.logs_path:
            self.print_dbg("  the logs will be saved to %s" % self.logs_path)

        self._proc = subprocess.Popen(self.cmd,
                                      stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT)
        try:
            for cmd in self._preamble:
                self._add(cmd)
        except BrokenPipeError:
            err = self._proc.stdout.readline()
            raise SmtError(self.name, "The solver rejected the preamble: %s" % err)

        return self

    def save_to_disk(self):
        assert self.logs_path
        with open(self.logs_path, mode='w') as logfile:
            logfile.writelines(str(self))

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

    def _add(self, smt_command: str, comment: str or None = None):
        assert not comment or comment != ""
        self.log(smt_command, comment)
        msg = ("%s\n" % smt_command).encode(self.ENCODING)
        self._proc.stdin.write(msg)

    def _sync_line(self):
        line = self._proc.stdout.readline()
        # print(line)
        line = line.decode(self.ENCODING)
        line = line.strip()
        return line

    def _read_till(self, terminator: Callable[[str], bool]):
        lines = []
        while True:
            line = self._sync_line()
            lines.append(line)
            if terminator(line):
                break
            if line.startswith("(error"):
                if '"' in line:  # otherwise it's not an error (it is just a variable whose name starts with "error")
                    line = line[line.index('"')+1:]
                    line = line[:line.index('"')]
                    m = re.match(r"line\s+(\d+)\s+column\s+(\d+):\s+(.*)", line)
                    if m and len(m.groups()) == 3:
                        # Try to get the offending line from the logs
                        row_num = int(m.group(1)) - 1
                        col_num = int(m.group(2))
                        precise_msg = m.group(3)
                        z3_log = self.get_z3_log()
                        offending_line = z3_log[row_num]
                        line = "`%s`: %s in position %d" % (offending_line, precise_msg, col_num)
                    raise SmtError(self.name, line)
        return lines[:-1]

    def _sync_with_policy(self, policy: Callable[[str], bool]):
        return self._read_till(terminator=policy)

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
            self._add('(echo "ping")')  # this hack prevents STDOUT from being blocked forever
            raise SmtTimeout(self.name, timeout)

        return res

    def get_last_query_duration_in_secs(self) -> int:
        assert self.__last_query_duration is not None, \
            "no queries have been submitted so far"
        return self.__last_query_duration

    def _query(self, query_str: str, timeout: float or None = None,
               policy: Callable[[str], bool] or None = None) -> Union[str, List[str]]:

        self.print_dbg("Submitting query %s" % query_str)

        if policy:
            # Use user-provided policy for detecting end of results stream
            self._add(query_str)
        else:
            # Use standard policy
            pseudo_unique_str = "Completed Query %d" % self._query_counter
            self._add('%s(echo "%s")' % (query_str, pseudo_unique_str))
            policy = (lambda line: line.strip("\"") == pseudo_unique_str)

        try:
            self._proc.stdin.flush()
        except BrokenPipeError:
            raise SmtError(self.name, "The solver has closed the session; cannot submit new queries.")

        res = ["crash"]  # should get overridden if operation succeeds
        start_time = time.perf_counter()
        try:
            res = self._readlines(policy, timeout)
        except SmtTimeout as e:
            res = ["timeout"]
            raise e
        except SmtError as e:
            res = ["error"]
            raise e
        finally:
            t = time.perf_counter() - start_time
            self.log_comment("%s in %.3f seconds" % (" ".join(res), t))
            self.print_dbg("  response: %s, duration: %.3f seconds" % (" ".join(res), t))
            self.__last_query_duration = t
            self._query_counter += 1

        return res

    def push(self, comment: str):
        self._tab_level += 1
        self._add("(push)", comment)

    def pop(self, comment: str):
        self._add("(pop)", comment)
        self._tab_level -= 1

    def reset(self, comment: str, recover_preamble=False, recover_declarations=False):
        self._tab_level = 0
        self._add("(reset)", comment)

        # The code below is untested.
        # new_preamble_commands = []
        # current_z3_log = self.get_z3_log()
        # for cmd_num, cmd in enumerate(current_z3_log):
        #     if cmd.startswith("(reset)") or cmd.startswith("(push)"):
        #         break
        #     if recover_preamble and cmd_num < len(self._preamble):
        #         new_preamble_commands.append(cmd)
        #     elif recover_declarations and (cmd.startswith("(declare-") or cmd.startswith("(define-")):
        #         new_preamble_commands.append(cmd)
        #
        # for pre_cmd in new_preamble_commands:
        #     self._add(pre_cmd)

    def add_constraint(self, condition: str, comment=None):
        constraint = "(assert %s)" % condition
        self._add(constraint, comment)

    def add_soft_constraint(self, condition: str, weight=1, comment=None):
        constraint = "(assert-soft %s :weight %d)" % (condition, weight)
        self._add(constraint, comment)

    def add_constraints(self, conditions: List[str], comment=None):
        if comment:
            self.log_comment(comment)
        for cond in conditions:
            self.add_constraint(cond)

    def add_previous_commands(self, commands: List[str], comment=None):
        if comment:
            self.log_comment(comment)
        for cmd in commands:
            if '(push)' in cmd:
                self._tab_level += 1
                self._add(cmd)
            elif '(pop)' in cmd:
                self._add(cmd)
                self._tab_level -= 1
            else:
                self._add(cmd)

    def _declare(self, var_name: str, var_signature: str, comment=None) -> str:
        full_name = var_name
        decl = "(declare-fun %s %s)" % (full_name, var_signature)
        self._add(decl, comment)
        return decl

    def declare_const(self, var_name: str, var_type: str, comment=None) -> str:
        return self._declare(var_name, "() %s" % var_type, comment=comment)

    def declare_consts(self, funs: List[Tuple[str, str]], comment=None) -> List[str]:
        if comment:
            self.log_comment(comment)
        decls = []
        for name, typ in funs:
            decl = self.declare_const(name, typ)
            decls.append(decl)
        if len(decls) == 0:
            self.log_comment("<Nothing to add for `%s`>" % comment)
        return decls

    def declare_fun(self, fun_name: str, fun_args: Tuple[str, ...], fun_type, comment=None) -> str:
        args_str = " ".join(fun_args)
        return self._declare(fun_name, "(%s) %s" % (args_str, fun_type), comment=comment)

    def declare_funs(self, funs: List[Tuple[str, Tuple[str], str]], comment=None) -> List[str]:
        if comment:
            self.log_comment(comment)
        decls = []
        for fun_name, fun_args, fun_type in funs:
            decl = self.declare_fun(fun_name, fun_args, fun_type)
            decls.append(decl)
        return decls

    def check_sat_assuming(self, assumptions: Dict[str, bool],
                           timeout: float or None = None) -> Tuple[str, any]:

        arg = " ".join(["     %s " % flag if assumptions[flag] else
                        "(not %s)" % flag for flag in assumptions])
        try:
            res = self._query("(check-sat-assuming (%s))" % arg, timeout=timeout)
        except SmtTimeout:
            return "timeout", self.__last_query_duration

        res = " ".join(res)

        if res == "unknown":
            reason = self.__get_reason_unknown()
            return "unknown", reason

        return res, None

    def check_sat(self, timeout: float or None = None) -> Tuple[str, any]:
        try:
            res = self._query("(check-sat)", timeout=timeout)
        except SmtTimeout:
            return "timeout", self.__last_query_duration

        res = " ".join(res)

        if res == "unknown":
            reason = self.__get_reason_unknown()
            return "unknown", reason

        return res, None

    def __get_reason_unknown(self) -> str:
        res = self._query("(get-info :reason-unknown)")
        return " ".join(res)

    @staticmethod
    def kill(pid):
        process = psutil.Process(pid)
        for proc in process.children(recursive=True):
            proc.kill()
        process.kill()

    def finalize(self, comment=None):
        self._add("(exit)", comment)
        try:
            self.kill(self._proc.pid)
            self._proc.communicate()
        except BrokenPipeError:
            pass
        self._proc.poll()
        return self._proc.returncode

    def get_values(self, model_vars: Set[str]) -> str or None:
        assert len(model_vars) > 0, "need to provide some vars for which the solver should provide values"

        model_var_str = " ".join(model_vars)
        query = "(get-value (%s))" % model_var_str
        try:
            res = self._query(query)
        except SmtTimeout:
            return None
        except SmtError as err:
            if "model is not available" in err.message:
                return None
            else:
                raise err

        res = " ".join(res)
        return res

    def simplified_log(self):
        log = self.get_z3_log()

        index_first_push = next((index for index, line in enumerate(log) if line.startswith('(push)')), -1)
        assert index_first_push != -1, "invalid log"
        # remove everything before the first push (options, type declarations, etc)
        log = log[index_first_push:]

        index_first_pop = next((index for index, line in enumerate(log) if line.startswith('(pop)')), -1)
        while index_first_pop != -1:
            index_last_push = next((index for index, line in enumerate(reversed(log[0: index_first_pop]))
                                    if line.startswith('(push)')), -1)
            index_last_push = len(log[0: index_first_pop]) - index_last_push - 1
            del log[index_last_push: index_first_pop + 1]
            index_first_pop = next((index for index, line in enumerate(log) if line.startswith('(pop)')), -1)
        return log

