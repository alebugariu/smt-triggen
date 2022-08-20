###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2022 ETH Zurich.
###############################################################################

import sys

def tail_call_opt(condition):
    def decorator(func):
        if not condition:
            return func
        else:
            if 'macropy.experimental.tco'not in sys.modules:
                from macropy.experimental.tco import macros, hq, tco
            return tco(func)
    return decorator

