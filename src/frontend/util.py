###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2024 ETH Zurich.
###############################################################################

import re


def Camel2_snake(camel: str) -> str:
    return re.sub(r"([A-Z])", r"_\g<1>", camel).strip('_').lower()

