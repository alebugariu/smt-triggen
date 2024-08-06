###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2024 ETH Zurich.
###############################################################################

from src.algorithms.singletons import IndividualAxiomTester
from src.session.robust_mbqi_runner import RobustMbqiRunner
from src.utils.preamble import STANDARD_MBQI_PREAMBLE


class Mbqi(IndividualAxiomTester):

    DBG_PREFIX = "MBQI"

    def __init__(self, filename: str, tmp_file_dir: str,
                 logs_dir: str, hard_timeout: float or None):

        global debug_mode
        if hard_timeout:
            self.hard_timeout = hard_timeout
        else:
            self.hard_timeout = 600.0
        super().__init__(filename=filename, tmp_file_dir=tmp_file_dir, logs_dir=logs_dir)

    def run(self) -> str:

        preamble = STANDARD_MBQI_PREAMBLE + \
                   self.m.get_setinfo() + \
                   self.m.get_setlogic() + \
                   self.m.get_declarations() + \
                   self.m.get_assertions()

        with RobustMbqiRunner(name="MBQI session", preamble=preamble, hard_timeout=self.hard_timeout) as mbqi:

            outcome, attach = mbqi.check_sat(timeout=self.hard_timeout)

        if outcome == "unknown":
            result = "unknown: %s" % str(attach)
        elif outcome == "timeout":
            result = "timeout after %.3f seconds"
        else:
            result = outcome

        return result
