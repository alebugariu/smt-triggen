###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2022 ETH Zurich.
###############################################################################

import re
from collections import OrderedDict
from typing import Dict

from src.session.exceptions import SmtTimeout, PatterProviderTimeout
from src.session.session import Session
from src.utils.preamble import STANDARD_PREAMBLE, PRETTY_PRINTER_NOWRAP


class PatternProvider(Session):

    DBG_PREFIX = "PatternProvider"

    def __init__(self,
                 name: str,
                 preamble=STANDARD_PREAMBLE,
                 cmd=('z3', '-smt2', '-t:4000', '-v:10', '-in'),
                 hard_timeout_per_query=4.0,
                 logs_path=None):

        preamble = list(preamble) + PRETTY_PRINTER_NOWRAP
        super().__init__(name=name, preamble=preamble, cmd=cmd,
                         hard_timeout_per_query=hard_timeout_per_query, logs_path=logs_path)

    def get_pattern(self, formula: str) -> Dict[str, str]:

        self.push("Begin pattern inference...")
        #self.reset(comment="Begin pattern inference...", recover_preamble=True, recover_declarations=True)

        self.add_constraint(formula)
        try:
            res = self._query("(check-sat)")
        except SmtTimeout as e:
            raise PatterProviderTimeout(e.session, e.duration_sec)
        finally:
            self.pop("End pattern inference.")

        pattern_dict = OrderedDict()
        current_qid = None
        for line in res:
            if "(smt.maximizing-bv-sharing)" in line:
                break
            if line.startswith("(smt.inferred-patterns"):
                m = re.match(r"^.*:qid\s+(.*?)$", line)
                qid = m.group(1)
                current_qid = qid
                continue
            if current_qid:
                if line.startswith(")"):
                    continue
                if current_qid in pattern_dict:
                    pattern_dict[current_qid] += " %s" % line
                else:
                    pattern_dict[current_qid] = line

        return pattern_dict
