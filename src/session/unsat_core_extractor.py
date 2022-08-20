###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2022 ETH Zurich.
###############################################################################

import re
from collections import OrderedDict
from enum import Enum
from typing import List, Dict

from src.minimizer.minimizer import Minimizer
from src.session.exceptions import SmtTimeout
from src.session.session import Session
from src.utils.preamble import STANDARD_PREAMBLE


class UnsatCoreStrategy(Enum):

    ASK_SOLVER = "ask_solver"
    LINEAR_SEARCH = "linear_search"
    BINARY_SEARCH = "nibary_search"
    BEST_EFFORT = "best_effort"

    def __str__(self):
        return self.value


class UnsatCoreExtractor(Session):

    DBG_PREFIX = "UnsatCoreExtracter"

    PSEUDO_UNIQUE_NUMBER = 21796879

    def __init__(self,
                 name: str,
                 preamble=STANDARD_PREAMBLE,
                 cmd=('z3', '-smt2', '-t:300000', '-in'),
                 hard_timeout_per_query=300.0,
                 logs_path=None):

        self.assert_counter = 0
        self.id_to_name_map = OrderedDict()
        super().__init__(name=name, preamble=preamble, cmd=cmd,
                         hard_timeout_per_query=hard_timeout_per_query, logs_path=logs_path)

    def __get_assert_flag(self, aid: int) -> str:
        flag = "A%04d_%d" % (aid, self.PSEUDO_UNIQUE_NUMBER)
        return flag

    def add_constraint(self, condition: str, comment=None):
        flag = self.__get_assert_flag(self.assert_counter)
        flag_decl = "(declare-const %s Bool)" % flag

        m = re.match(Minimizer.ASSERTION_NAME_PATTERN, condition)
        assert len(m.groups()) == 1, "Each assertion must have a unique name"
        name = m.group(1)
        self.id_to_name_map[self.assert_counter] = name

        constraint = "(assert (=> %s %s))" % (flag, condition)
        self._add(flag_decl)
        self._add(constraint, comment)
        self.assert_counter += 1

    def __get_assumptions_from_flags(self, flags: List[bool]) -> Dict[str, bool]:
        assumptions = OrderedDict()
        for aid, flag in enumerate(flags):
            assumptions[self.__get_assert_flag(aid)] = flag
        return assumptions

    def linear_search(self) -> List[str] or None:

        flags = [True] * len(self.id_to_name_map)

        # Check-sat with all assertions enabled and measure the time
        assumptions = self.__get_assumptions_from_flags(flags)
        try:
            outcome, _ = self.check_sat_assuming(assumptions)
        except SmtTimeout as e:
            outcome = "timeout"
            expected_time = e.duration_sec
        else:
            expected_time = self.get_last_query_duration_in_secs()

        if outcome != "unsat":
            self.print_dbg("Original query outcome is %s; cannot extract unsat core"
                           % outcome)
            return None

        # status = "unsat"
        for aid, name in self.id_to_name_map.items():
            # Disable assertion with current id
            flags[aid] = False
            assumptions = self.__get_assumptions_from_flags(flags)
            try:
                outcome, _ = self.check_sat_assuming(assumptions, timeout=(1.2*expected_time))
            except SmtTimeout:     # removing this assertion results in timeout:
                flags[aid] = True  # put back this assertion
            else:
                if outcome != "unsat":  # removing this assertion does not result in unsat:
                    flags[aid] = True   # put back this assertion

        return [self.id_to_name_map[aid] for aid, flag in enumerate(flags) if flag]

    def extract_unsat_core(self, strategy: UnsatCoreStrategy = UnsatCoreStrategy.LINEAR_SEARCH):
        return self.linear_search()
