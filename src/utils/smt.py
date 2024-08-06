###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2024 ETH Zurich.
###############################################################################

import io
import shutil
from typing import List, Tuple, Iterable, Set

from pysmt.fnode import FNode
from pysmt.shortcuts import get_env
from pysmt.smtlib.printers import SmtPrinter
from pysmt.smtlib.script import SmtLibCommand
from pysmt.typing import PySMTType
from src.algorithms.formulas.fresh import FreshVariable
from src.minimizer.test_runner import TestRunner
from src.utils.enums import SMTSolver


def drop_aux_info(status):
    return list(filter(lambda resp: \
                       not resp.startswith('(') and \
                       not resp.startswith('<') and 
                       'success' not in resp, status))


def get_test_outcome(runner, file, use_soft_timeout=False):
    outcome = runner.run_test(file, get_raw_output=True, use_soft_timeout=use_soft_timeout, ignore_warnings=False)
    return drop_aux_info(outcome)


def are_smt_files_equisat(solver, orig_file, new_file, timeout=5, use_soft_timeout=False, multiquery=False):

    """Returns whether the solver outputs the same stuff for both files."""

    assert solver is not None
    assert orig_file is not None
    assert new_file is not None
    runner = TestRunner(solver, timeout=timeout)
    original_outcome = get_test_outcome(runner, orig_file, use_soft_timeout)
    new_outcome = get_test_outcome(runner, new_file, use_soft_timeout)

    if multiquery:
        return ' '.join(original_outcome) == ' '.join(new_outcome), original_outcome, new_outcome
    else:
        return str(original_outcome[0]) == str(new_outcome[0]), str(original_outcome[0]), str(new_outcome[0])


def run_test(original_file_name: str, test_file_name: str, fresh_vars: Set[FreshVariable],
             dummy_constraints: List[str], batch_size: int, timeout_per_query: float) -> str:
    shutil.copy(original_file_name, test_file_name)

    with open(test_file_name, "a") as test_file:
        # declare the fresh variables:
        for fresh_var in fresh_vars:
            test_file.write(fresh_var.declaration + "\n")

        for constraint in dummy_constraints:
            test_file.write(constraint + "\n")
        test_file.write("(check-sat)")

    runner = TestRunner(SMTSolver.Z3, timeout=batch_size * timeout_per_query)
    outcome, _ = runner.run_test(test_file_name)
    return outcome


def serialize_node_with_annotations(n: FNode) -> str:
    env = get_env()
    annotations = env._parser.cache.annotations
    with io.StringIO() as buf:
        p = SmtPrinter(stream=buf, annotations=annotations)
        p.printer(n)
        res = buf.getvalue()
    return res


def serialize_cmd_with_annotations(cmd: SmtLibCommand) -> str:
    env = get_env()
    annotations = env._parser.cache.annotations
    with io.StringIO() as buf:
        p = SmtPrinter(stream=buf, annotations=annotations)
        cmd.serialize(outstream=buf, printer=p)
        res = buf.getvalue()
    return res


PySMTObj = SmtLibCommand or FNode or PySMTType


def smt2(t: PySMTObj) -> str:
    if isinstance(t, SmtLibCommand):
        return serialize_cmd_with_annotations(t)
    elif isinstance(t, FNode):
        return serialize_node_with_annotations(t)
    elif isinstance(t, PySMTType):
        return str(t)
    else:
        return str(t)


def smt2_list(xs: Iterable[PySMTObj]) -> List[str]:
    result = []
    for x in xs:
        res = smt2(x)
        result.append(res)

    return result


def smt2_tup(tt: Tuple[PySMTObj]) -> Tuple[str]:
    result = smt2_list(tt)
    return tuple(result)


def smt2_tup_tup(ttt):
    """ Seriaizes tuples of tuples of FNodes, e.g. complete pattern objects. """
    result = []
    for arg in ttt:
        if isinstance(arg, tuple):
            res = smt2_tup(arg)
        else:
            res = smt2(arg)
        result.append(res)

    return tuple(result)
