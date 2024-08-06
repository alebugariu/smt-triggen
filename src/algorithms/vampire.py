###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2024 ETH Zurich.
###############################################################################

from src.algorithms.singletons import IndividualAxiomTester
from src.session.robust_vampire_runner import RobustVampireRunner

from src.session import debug_mode


class Vampire(IndividualAxiomTester):

    DBG_PREFIX = "Vampire"

    def __init__(self, filename: str, tmp_file_dir: str,
                 logs_dir: str, hard_timeout: float or None):

        global debug_mode
        if hard_timeout:
            self.hard_timeout = hard_timeout
        else:
            self.hard_timeout = 600.0

        super().__init__(filename=filename, tmp_file_dir=tmp_file_dir, logs_dir=logs_dir)

        self.vamp_file_name = self.m.save_to_disk(new_suffix='vamp', quiet=False,
                                                  for_vamp=True, as_artifact=True, as_out_file=False)

    def run(self) -> str:

        vamp = RobustVampireRunner(file=self.vamp_file_name,
                                   hard_timeout=self.hard_timeout,
                                   logs_path=self.tests_logs_path)
        result, runtime = vamp.run()

        self.print_dbg("Response from Vampire: %s (took %.3f seconds)" % (str(result), runtime))

        return result
