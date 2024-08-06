###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2024 ETH Zurich.
###############################################################################

from typing import Set, List

from sortedcontainers import SortedSet

from pysmt.fnode import FNode
from src.algorithms.clustering.cluster import Cluster
from src.algorithms.formulas.axiom import Axiom


# a group is a (sub)cluster where each quantified variable has at most one rewriting
class Group(Cluster):

    def __init__(self, similar_axioms: Set[Axiom], formula_manager):
        super().__init__(similar_axioms)

        self.formula_manager = formula_manager
        self.instantiations: List[List[FNode]] = []
        self.triggers: Set[FNode] = SortedSet(key=lambda x: True)
        self.qvars: Set[FNode] = SortedSet(key=lambda x: x.symbol_name())
        self.constants: Set[FNode] = SortedSet(key=lambda x: True)
        self.valid_instantiations: List[Set[FNode]] = []

        # instantiate all the similar axioms and collect their triggers
        for similar_axiom in self.similar_axioms:
            new_inst: List[FNode] = []
            for inst in similar_axiom.instantiations(mgr=self.formula_manager):
                if not self.already_contained(inst, self.instantiations):
                    new_inst.append(inst)
            if len(new_inst) > 0:
                self.instantiations.append(new_inst)
            self.triggers.update(similar_axiom.triggers)
            self.qvars.update(similar_axiom.quantified_variables)
            self.constants.update(similar_axiom.constants)

        self.only_qfree_axioms = len(self.qvars) == 0

    @staticmethod
    def already_contained(instantiation: FNode, all_instantiations: List[List[FNode]]) -> bool:
        for instantiations in all_instantiations:
            if instantiation in instantiations:
                return True
        return False
