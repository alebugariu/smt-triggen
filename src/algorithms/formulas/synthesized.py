###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2024 ETH Zurich.
###############################################################################

from typing import List, Set

from sortedcontainers import SortedSet

from pysmt.fnode import FNode
from src.algorithms.formulas.axiom import Axiom
from src.algorithms.formulas.rewritings import Rewriting


class SatFormula:

    def __init__(self, declarations_and_assertions: List[str], similar_axioms: Set[Axiom], formula: 'Formula'):
        self.declarations_and_assertions = declarations_and_assertions
        self.similar_axioms = similar_axioms
        self.negated_axiom = formula.negated_axiom
        self.instantiations = formula.instantiations
        self.rewritings = formula.rewritings
        self.triggers = formula.triggers
        self.qvars: Set[FNode] = SortedSet(self.negated_axiom.quantified_variables, key=lambda x: x.symbol_name())
        self.constants: Set[FNode] = SortedSet(self.negated_axiom.constants, key=lambda x: x.symbol_name())
        for axiom in self.similar_axioms:
            self.qvars.update(axiom.quantified_variables)
            self.constants.update(axiom.constants)

    def __str__(self):
        return '\n'.join(self.declarations_and_assertions)

    def __repr__(self):
        return self.__str__()

    def unbalanced_push_commands(self):
        push_commands = [line for line in self.declarations_and_assertions if line.strip().startswith("(push")]
        return len(push_commands)


class Formula:

    def __init__(self, negated_axiom: Axiom, instantiations: Set[FNode],
                 rewritings: Set[Rewriting], triggers: Set[FNode]):
        self.negated_axiom = negated_axiom
        self.instantiations = instantiations
        self.rewritings = rewritings
        self.triggers = triggers
        self.formula_hash = -1

    def __str__(self):
        formula = str(self.negated_axiom) + '\n'
        for instantiation in self.instantiations:
            formula += str(instantiation) + '\n'
        for rewriting in self.rewritings:
            formula += str(rewriting) + '\n'
        return formula

    def __repr__(self):
        return self.__str__()

    def contained(self, other: 'Formula') -> bool:
        if self.negated_axiom != other.negated_axiom:
            return False
        for instantiation in self.instantiations:
            if instantiation not in other.instantiations:
                return False
        for rewriting in self.rewritings:
            if rewriting not in other.rewritings and rewriting.reversed() not in other.rewritings:
                return False
        return True

    def __hash__(self):
        if self.formula_hash == -1:
            self.formula_hash = hash(self.negated_axiom)
            for instantiation in self.instantiations:
                self.formula_hash += hash(instantiation)
            for rewriting in self.rewritings:
                self.formula_hash += hash(rewriting)
            for trigger in self.triggers:
                self.formula_hash += hash(trigger)
        return self.formula_hash

    def __eq__(self, other: 'Formula') -> bool:
        if self.negated_axiom != other.negated_axiom:
            return False
        if self.instantiations != other.instantiations:
            return False
        if self.rewritings != other.rewritings:
            return False
        if self.triggers != other.triggers:
            return False
        return True
