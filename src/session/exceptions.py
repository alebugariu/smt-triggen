###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2022 ETH Zurich.
###############################################################################


class SmtError(Exception):

    def __init__(self, session_name: str, message_from_solver: str):
        self.session = session_name
        self.message = message_from_solver

    def __str__(self):
        return "%s failed with error: `%s`" % (self.session, self.message)


class SmtTimeout(Exception):

    def __init__(self, session_name: str, duration_sec: float):
        self.session = session_name
        self.duration_sec = duration_sec

    def __str__(self):
        return "%s timed out after %.3f seconds" % (self.session, self.duration_sec)


class SkolemizationTimeout(SmtTimeout):
    pass


class PatterProviderTimeout(SmtTimeout):
    pass


class SimplifierTimeout(SmtTimeout):
    pass


class SessionApiError(Exception):
    pass
