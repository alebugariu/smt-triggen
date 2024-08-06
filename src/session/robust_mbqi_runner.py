###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2024 ETH Zurich.
###############################################################################

from src.session.session import Session
from src.utils.preamble import STANDARD_MBQI_PREAMBLE


class RobustMbqiRunner(Session):

    DBG_PREFIX = "RobustMbqiRunner"

    def __init__(self,
                 name: str,
                 preamble=STANDARD_MBQI_PREAMBLE,
                 hard_timeout=600.0,
                 logs_path=None):

        cmd = ('z3', '-smt2', '-t:%d' % round(hard_timeout*1000), '-in')
        super().__init__(name=name, preamble=preamble, cmd=cmd,
                         hard_timeout_per_query=hard_timeout, logs_path=logs_path)

    def push(self, comment):
        raise Exception("%s does not support incremental solving" % self.DBG_PREFIX)

    def pop(self, comment):
        raise Exception("%s does not support incremental solving" % self.DBG_PREFIX)
