###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2024 ETH Zurich.
###############################################################################

from typing import Set

from pysmt.fnode import FNode


class ProtoTerms:

    def __init__(self, terms: Set[FNode], qvars: Set[FNode], constants: Set[FNode]):
        self.terms = terms
        self.qvars = qvars
        # these are the constants of the axioms from which this proto term was constructed
        self.constants = constants

    def __str__(self):
        return str(self.terms)

    def just_constants(self) -> bool:
        return len(self.qvars) == 0
