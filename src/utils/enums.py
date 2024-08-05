###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2022 ETH Zurich.
###############################################################################

from enum import Enum


class SMTSolver(Enum):
    Z3 = "z3"
    CVC4 = "cvc4"
    CVC5 = "cvc5"
    VAMPIRE = "vampire"

    def __str__(self):
        return self.value


class IncrementalityMode(Enum):
    INCREMENTAL     = "mode.incremental"
    NON_INCREMENTAL = "mode.non-incremental"

    def __str__(self):
        return self.value


# List of SMT2 standard commands
# The order of elements in this enum is used for sorting lines in SMT files
class SmtCommand(Enum):
    SET_INFO          = "set-info"
    SET_LOGIC         = "set-logic"
    SET_OPTION        = "set-option"
    LABELS            = "labels"  # Found in Boogie
    DECLARE_SORT      = "declare-sort"
    DEFINE_SORT       = "define-sort"
    DECLARE_DATATYPES = "declare-datatypes"
    DEFINE_FUN        = "define-fun"
    DECLARE_FUN       = "declare-fun"
    DECLARE_CONST     = "declare-const"
    DEFINE_CONST      = "define-const"
    PUSH              = "push"
    ASSERT            = "assert"
    POP               = "pop"
    CHECK_SAT         = "check-sat"
    GET_UNSAT_CORE    = "get-unsat-core"
    GET_PROOF         = "get-proof"

    GET_INFO          = "get-info"
    EVAL              = "eval"
    ECHO              = "echo"

    RESET             = "reset"
    EXIT              = "exit"

    def __str__(self):
        return self.value


class SmtResponse(Enum):
    SAT     = "sat"
    UNSAT   = "unsat"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"

    def __str__(self):
        return self.value
