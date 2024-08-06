###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2024 ETH Zurich.
###############################################################################

from typing import Set

from sortedcontainers import SortedSet

from pysmt.fnode import FNode
from src.utils.smt import smt2


class Rewriting:

    number = 0

    def __init__(self, qvar: FNode, equivalent_expr: FNode, just_qvars: bool):
        self.qvar = qvar
        self.equivalent_expr = equivalent_expr
        # just_qvars is True if the equivalent_expr is also a quantified variable
        self.just_qvars = just_qvars
        self.rewriting_id: int = Rewriting.number
        Rewriting.number += 1

    def __str__(self):
        return "R" + str(self.rewriting_id) + ": " + smt2(self.qvar) + " = " + smt2(self.equivalent_expr)

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other: 'Rewriting'):
        if str(self.qvar) < str(other.qvar):
            return True
        if str(self.qvar) > str(other.qvar):
            return False
        # str(self.qvar) == str(other.qvar)
        return str(self.equivalent_expr) < str(other.equivalent_expr)

    def __eq__(self, other):
        return self.qvar == other.qvar and self.equivalent_expr == other.equivalent_expr

    def __hash__(self):
        return 1000 * hash(self.qvar) * hash(self.equivalent_expr)

    def reversed(self):
        return Rewriting(self.equivalent_expr, self.qvar, self.just_qvars)


# this class represents Rewritings (i.e, equalities between quantified variables and their equivalent expressions) that
# should be applied together
class ComposedRewriting:

    def __init__(self, rewritings: Set[Rewriting], similar_axioms: Set['Axiom']):
        self.rewritings = rewritings
        self.similar_axioms = similar_axioms
        self.parent_ids: Set[str] = SortedSet([axiom.parent_id for axiom in self.similar_axioms], key=lambda x: x)
        self.just_qvars = all(rewriting.just_qvars for rewriting in self.rewritings)
        self.rewriting_id: int = int("".join([str(rw.rewriting_id) for rw in self.rewritings]))
        self.qvars = SortedSet([r.qvar for r in self.rewritings], key=lambda x: x.symbol_name())

    def __lt__(self, other):
        if len(self.rewritings) < len(other.rewritings):
            return True
        if len(self.rewritings) > len(other.rewritings):
            return False
        # len(self.rewritings) == len(other.rewritings)
        if len(self.similar_axioms) < len(other.similar_axioms):
            return True
        if len(self.similar_axioms) > len(other.similar_axioms):
            return False
        # len(self.rewritings) == len(other.rewritings) && len(self.similar_axioms) == len(other.similar_axioms)
        return self.rewriting_id < other.rewriting_id

    def __str__(self):
        result: str = '<similar axioms: '
        for similar_axiom in self.similar_axioms:
            result += similar_axiom.axiom_id + ', '
        result += 'rewritings: '
        for rewriting in self.rewritings:
            result += str(rewriting) + ', '
        result = result[:-2] + '>'
        return result

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.rewritings == other.rewritings and (self.similar_axioms == other.similar_axioms or
                                                        self.parent_ids == other.parent_ids)

    def __hash__(self):
        result = 0
        for rewriting in self.rewritings:
            result += hash(rewriting)
        return result

    def equivalent(self, other: 'ComposedRewriting') -> bool:
        if not self.just_qvars or not other.just_qvars:
            return False
        if len(self.rewritings) != len(other.rewritings):
            return False
        for r in self.rewritings:
            if r not in other.rewritings and not r.reversed() in other.rewritings:
                return False
        return True

    def join(self, other: 'ComposedRewriting'):
        return ComposedRewriting(self.rewritings.union(other.rewritings),
                                 self.similar_axioms.union(other.similar_axioms))

    def get_equivalent_expression(self, qvar: FNode) -> FNode:
        for r in self.rewritings:
            if r.qvar == qvar:
                return r.equivalent_expr

    # returns True if self and other can be put in the same subset (they have max one rewriting per quantified var)
    def same_subset(self, other: 'ComposedRewriting'):
        if self == other or self.qvars == other.qvars:
            return False
        if self.qvars.isdisjoint(other.qvars):
            return True
        common_qvars: Set[FNode] = self.qvars.intersection(other.qvars)
        for qvar in common_qvars:
            if self.get_equivalent_expression(qvar) != other.get_equivalent_expression(qvar):
                return False
        # they share the same rewritings
        return True
