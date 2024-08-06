###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2024 ETH Zurich.
###############################################################################

import subprocess
import time

from src.minimizer.exceptions import SolverErrorException
from src.utils.enums import SMTSolver


class TestRunner:
    DBG_PREFIX = "  TestRunner >> "

    # Static helper methods
    @classmethod
    def print_dbg(cls, msg, tabs=0):
        ttt = (tabs * "  ")
        print(cls.DBG_PREFIX + ttt + msg.replace('\n', '\n' + ttt))

    def __init__(self, solver, seed=0, timeout=15, enum_inst=False):
        self.solver = solver
        self.seed = seed
        if timeout > 10800:
            self.print_dbg('Timeout in TestRunner should be set in seconds; you sure %d is a good choice?' % timeout)
        self.timeout = timeout
        self.enum_inst = enum_inst
        self._last_run_command = ''

    def get_last_command(self):
        return self._last_run_command

    def run_test_timed(self, *args, **keywords):
        start = time.time()
        *result, = self.run_test(*args, **keywords)
        end = time.time()
        return (*result, end - start)

    def __get_cmd(self, test_file_name: str, use_soft_timeout):
        options = [str(self.solver)]
        seed_options = []
        timeout_options = []
        tool_specific_options = []

        if self.solver == SMTSolver.Z3:
            if self.seed:
                seed_options = ['sat.random_seed=%s' % self.seed, 'smt.random_seed=%s' % self.seed]
            timeout_options = ['%s:%d' % ('-t' if use_soft_timeout else '-T',
                                          self.timeout * 1000 if use_soft_timeout else max(1, self.timeout))]
        if self.solver == SMTSolver.CVC4 or self.solver == SMTSolver.CVC5:
            if self.seed:
                seed_options = ['--seed=%d' % self.seed, '--random-seed=%d' % self.seed]
            timeout_options = ['--tlimit=%d' % (self.timeout * 1000)]
            tool_specific_options = ['--lang=smt2']
            if self.enum_inst:
                tool_specific_options += ['--full-saturate-quant']

        options += seed_options + timeout_options + tool_specific_options + [test_file_name]
        return options

    def run_test(self, test_file_name: str, get_raw_output=False, ignore_warnings=True, use_soft_timeout=False,
                 quiet=False):

        if self.solver is SMTSolver.VAMPIRE:
            return self._run_vampire(test_file_name, get_raw_output, quiet)

        options = self.__get_cmd(test_file_name, use_soft_timeout)
        self._last_run_command = ' '.join(options)

        if not quiet:
            self.print_dbg("Running `%s`..." % self._last_run_command)

        try:
            result = subprocess.run(options, stdout=subprocess.PIPE, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            return 'timeout', []

        output = str(result.stdout)
        if result.stderr:
            output += '\n' + str(result.stderr)
        output = output.replace('b\'', '').replace('\\n\'', '')
        if ignore_warnings:
            output = output.replace('unsupported\\n', '')
        output_lines = output.split('\\n')

        if get_raw_output:
            return output_lines

        if 'timeout' in output:
            return 'timeout', []

        actual_status = output_lines[0]
        attachment = []
        if len(output_lines) > 1:
            attachment = output_lines[1:]

        if "error" in actual_status:
            raise SolverErrorException("  failed with status `%s`" % actual_status)
        if (self.solver is SMTSolver.CVC4 or self.solver is SMTSolver.CVC5) and \
                "Error! Proofs not yet supported" in attachment[-2]:
            self.print_dbg("  failed with status `%s` %s" % (actual_status, attachment[-2]))
            return actual_status, []

        elif not quiet:
            if attachment:
                self.print_dbg("  succeeded with status '%s' %s" % (actual_status, attachment))
            else:
                self.print_dbg("  succeeded with status '%s'" % actual_status)

        return actual_status, attachment

    def _run_vampire(self, test_file_name: str, get_raw_output=False, quiet=False):

        options = ['vampire',
                   '--mode', 'casc',
                   '--input_syntax', 'smtlib2',
                   '--time_limit', '%ds' % round(self.timeout),
                   '--memory_limit', '%d' % 2000000,
                   '--random_seed', '%d' % self.seed,
                   test_file_name]

        if not quiet:
            self.print_dbg("Running `%s`..." % ' '.join(options))

        self._last_run_command = ' '.join(options)

        try:
            result = subprocess.run(args=options, stdout=subprocess.PIPE, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            return 'timeout', []

        output = str(result.stdout)
        if result.stderr:
            output += '\n' + str(result.stderr)
        output = output.replace('b\'', '').replace('\\n\'', '')

        output_lines = output.split('\\n')

        if get_raw_output:
            return output_lines

        proof = []
        out_str = (''.join(output_lines)).lower()
        if 'success' in out_str:
            actual_status = 'unsat'
            index = output_lines.index("% Refutation found. Thanks to Tanya!")
            if index != -1:
                proof = output_lines[index:]
        elif 'timeout' in out_str:
            actual_status = 'timeout'
        elif 'error' in out_str:
            e = ""
            if result.stdout:
                e += str(result.stdout)
            if result.stderr:
                e += "\n%s" % str(result.stderr)
            self.print_dbg("Vampire returned an error: %s" % e)
            actual_status = 'error'
        else:
            actual_status = 'unknown'

        return actual_status, proof
