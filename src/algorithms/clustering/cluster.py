###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2024 ETH Zurich.
###############################################################################

from typing import Set

from sortedcontainers import SortedSet

from src.algorithms.formulas.axiom import Axiom


class Cluster:

    def __init__(self, similar_axioms: Set[Axiom]):
        self.similar_axioms = similar_axioms
        self.ids: Set[str] = SortedSet([axiom.axiom_id for axiom in self.similar_axioms], key=lambda x: x)
        self.size = len(self.similar_axioms)

    def __str__(self):
        return " ".join(axiom.axiom_id for axiom in self.similar_axioms)

    def __hash__(self):
        result = 0
        for axiom in self.similar_axioms:
            result += hash(axiom)
        return result

    def __eq__(self, other: 'Cluster'):
        return self.similar_axioms == other.similar_axioms

    def __lt__(self, other: 'Cluster'):
        return self.size < other.size
