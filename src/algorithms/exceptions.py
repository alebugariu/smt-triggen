###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2022 ETH Zurich.
###############################################################################

class PatternException(Exception):
    def __init__(self, msg: str):
        self.msg = msg

    def __str__(self):
        return self.msg


class InferredPatternInvalid(PatternException):
    def __init__(self, quantifier: str, invalid_pattern: str):
        self.quantifier = quantifier
        self.invalid_pattern = invalid_pattern

        super().__init__(str(self))

    def __str__(self):
        return "inferred pattern `%s` is invalid for quantifier `%s`" % (self.invalid_pattern, self.quantifier)
