###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2024 ETH Zurich.
###############################################################################

import itertools
from collections import OrderedDict
from typing import Set, Dict, List

from sortedcontainers import SortedSet

from pysmt.fnode import FNode
from pysmt.formula import FormulaManager
from pysmt.typing import PySMTType, _FunctionType
from src.algorithms.formulas.rewritings import ComposedRewriting, Rewriting
from src.utils.ast import collect, inner_disjuncts, equivalent_functions, collect_quantified_variables, \
    collect_top_level, extract_prenex_body, is_constant
from src.utils.smt import smt2
from pysmt.shortcuts import And, Not


class Axiom:

    def __lt__(self, other):
        return self.axiom_id < other.axiom_id

    def __init__(self, axiom_id: str, axiom_node: FNode, quantifiers: Set[FNode], triggers: Set[FNode],
                 skolem_functions: Set[FNode], parent_id: str):

        self.axiom_id = axiom_id
        self.axiom_node = axiom_node
        self.cluster_id = -1  # will be set during clustering
        # parent_id != axiom_id if the parent axiom was split into multiple axioms
        self.parent_id = parent_id

        #assert not ast_contains(axiom_node, lambda x: x.is_and()), 'the axiom should not contain ands'

        self.quantifiers = quantifiers
        self.triggers = triggers
        self.quantified_variables: Set[FNode] = collect_quantified_variables(self.quantifiers)
        self.disjuncts: List[FNode] = inner_disjuncts(self.axiom_node)

        # these are Symbols, not Functions
        self.skolem_functions = skolem_functions

        self.inst: List[FNode] = []

        self.axioms_with_common_symbols: Set['Axiom'] = SortedSet(key=lambda x: x.axiom_id)
        self.new_similar_axioms: bool = False  # will be set to true if the set of axioms_with_common_symbols changes
                                               # for different values of the similarity threshold

        # set of user-defined function symbols; also includes free variables (constants), which are functions with
        # no args; these are Symbols, not Functions
        self.function_symbols = self.axiom_node.get_free_variables()
        self.user_defined_constants = SortedSet([symbol for symbol in self.function_symbols if
                                                 is_constant(symbol)], key=lambda x: True)
        self.built_in_constants: Set[FNode] = SortedSet(key=lambda x: True)
        collect(self.axiom_node, lambda n: n.is_constant(), self.built_in_constants)

        # set of top-level user-defined functions collected from the body of the axiom; they are used in unification
        # (if the body is f(g(x)), we just include f); these are Functions, not Symbols
        self.functions: Set[FNode] = SortedSet(key=lambda x: True)
        collect_top_level(self.axiom_node, lambda n: n.is_function_application(), self.functions)

        # all the user-defined functions, self.user_defined_constants and self.built_in_constants grouped by their
        # (return) type; they are used in the extended unification (the type-based rewritings)
        self.types_to_functions: Dict[PySMTType, Set[FNode]] = OrderedDict()

        self.all_functions: Set[FNode] = SortedSet(key=lambda x: True)
        collect(self.axiom_node, lambda n: n.is_function_application(), self.all_functions)
        for a_function in self.all_functions.union(self.user_defined_constants).union(self.built_in_constants):
            typ = a_function.get_type()
            if not isinstance(typ, _FunctionType):
                if typ not in self.types_to_functions.keys():
                    sorted_functions: Set[FNode] = SortedSet({a_function}, key=lambda x: len(x.args()))
                    self.types_to_functions[typ] = sorted_functions
                else:
                    self.types_to_functions[typ].add(a_function)

        # additional user-defined function symbols from the triggers
        self.function_symbols_triggers: Set[FNode] = SortedSet(key=lambda x: True)
        for trigger in self.triggers:
            functions_from_trigger: Set[FNode] = SortedSet(key=lambda x: True)
            collect(trigger, lambda n: n.is_function_application(), functions_from_trigger)
            symbols_from_trigger: Set[FNode] = SortedSet([fun._content.payload for fun in functions_from_trigger],
                                                          key=lambda x: True)
            self.function_symbols_triggers.update(symbols_from_trigger)

        constants_from_triggers = SortedSet([symbol for symbol in self.function_symbols_triggers if
                                             is_constant(symbol)], key=lambda x: True)
        self.constants = SortedSet(self.user_defined_constants, key=lambda x: True).union(constants_from_triggers)

        self.cached_rewritings: Dict['Axiom', Set['ComposedRewriting']] = OrderedDict()
        # similar axioms already used for the unification of this with the key axiom
        self.cached_similar_axioms: Dict['Axiom', Set['Axiom']] = OrderedDict()

    def __str__(self):
        return self.axiom_id + ": " + smt2(self.axiom_node)

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.axiom_node)

    def is_quantified(self):
        return len(self.quantifiers) != 0

    def instantiations(self, mgr: FormulaManager) -> List[FNode]:
        if len(self.inst) == 0:
            qfree_disjuncts = [extract_prenex_body(disj, mgr) for disj in self.disjuncts]
            # generate the instantiations:
            for index, disjunct in enumerate(qfree_disjuncts):
                if index > 0:
                    negated_disjuncts = []
                    for previous_disjunct in qfree_disjuncts[0:index]:
                        negated_disjuncts.append(Not(previous_disjunct))
                    negated_disjuncts.append(disjunct)
                    self.inst.append(And(*negated_disjuncts))
                else:
                    self.inst.append(disjunct)
        return self.inst

    @staticmethod
    def cached_info(cached_rewritings: Set[ComposedRewriting], other_axioms: Set['Axiom'])-> Set[ComposedRewriting]:

        rewritings: Set[ComposedRewriting] = SortedSet()

        for rewriting in cached_rewritings:
            if rewriting.similar_axioms.issubset(other_axioms):
                rewritings.add(rewriting)

        return rewritings

    # identify type-based constraints which were not yet computed
    def additional_rewritings(self, applicable_rewritings: Set[ComposedRewriting], other_axiom: 'Axiom',
                              qvars: Set[FNode], remaining_axioms: Set['Axiom'], limit: int) -> Set[ComposedRewriting]:

        type_based_rewritings: Set[ComposedRewriting] = SortedSet()

        for composed_rewriting in applicable_rewritings:
            extended_rewritings: List[Set[ComposedRewriting]] = []
            found_type_based_rewritings: bool = False

            for r in composed_rewriting.rewritings:
                qvar = r.qvar
                equivalent_expr = r.equivalent_expr

                # transform r in a Composed rewriting (used for the extended rewritings)
                rw: Set[ComposedRewriting] = SortedSet({ComposedRewriting({r}, composed_rewriting.similar_axioms)})

                if len(remaining_axioms) != 0 and equivalent_expr in qvars:
                    # rewriting of the type x = y, which might be too imprecise
                    type_based_constraints = self.type_based_constraints(qvar, equivalent_expr, other_axiom,
                                                                         remaining_axioms)
                    if len(type_based_constraints) > 0:
                        found_type_based_rewritings = True
                        extended_rewritings.append(type_based_constraints)
                    else:
                        extended_rewritings.append(rw)
                else:
                    extended_rewritings.append(rw)

            if found_type_based_rewritings:
                type_based_rewritings.update(self.type_based_rewritings(extended_rewritings, limit))
        return type_based_rewritings

    # unify self with the other_axiom
    def unify(self, other_axiom: 'Axiom', similar_axioms: Set['Axiom'], type_constraints: bool, top_level: bool,
              limit: int) -> Set[ComposedRewriting]:

        qvars = self.quantified_variables.union(other_axiom.quantified_variables)
        skolem_functions = self.skolem_functions.union(other_axiom.skolem_functions)

        if other_axiom in self.cached_rewritings.keys():
            cached_rewritings = self.cached_rewritings[other_axiom]

            if len(cached_rewritings) == 0:
                # we already know that this axiom cannot unify with the 'other_axiom'
                # (so the similar_axioms don't matter)
                return cached_rewritings

            if not type_constraints:
                return self.cached_info(cached_rewritings, {other_axiom})

            # type constraints
            applicable_rewritings = self.cached_info(cached_rewritings, similar_axioms.union({other_axiom}))
            remaining_axioms = similar_axioms.difference(self.cached_similar_axioms[other_axiom])
            if len(remaining_axioms) == 0:
                return applicable_rewritings

            type_rewritings = self.additional_rewritings(applicable_rewritings, other_axiom, qvars,
                                                         remaining_axioms, limit)
            self.cached_rewritings[other_axiom].update(type_rewritings)
            self.cached_similar_axioms[other_axiom].update(remaining_axioms)
            return applicable_rewritings.union(type_rewritings)

        if top_level:
            functions = self.functions
            other_functions = other_axiom.functions
        else:
            functions = self.all_functions
            other_functions = other_axiom.all_functions

        rewritings: Set[ComposedRewriting] = SortedSet()

        if len(functions) != 0 and len(other_functions) != 0:
            # unification for quantified variables that appear as arguments to user-defined functions
            rewritings = self.unification(functions, other_functions, skolem_functions, qvars,
                                          other_axiom, similar_axioms, type_constraints, limit)

        self.cached_rewritings[other_axiom] = rewritings
        self.cached_similar_axioms[other_axiom] = similar_axioms

        # switch other_axiom with self, to compute the symmetric rewritings
        other_axiom.cached_rewritings[self] = self.symmetric_rewritings(rewritings, other_axiom)
        other_axiom.cached_similar_axioms[self] = similar_axioms
        return rewritings

    def symmetric_rewritings(self, rewritings: Set[ComposedRewriting], other_axiom: 'Axiom') -> Set[ComposedRewriting]:
        other_rewritings: Set[ComposedRewriting] = SortedSet()
        for r in rewritings:
            other_similar_axioms: Set['Axiom'] = SortedSet(key=lambda x: x.axiom_id)
            other_similar_axioms.update(r.similar_axioms.difference({other_axiom}).union({self}))
            other_rewritings.add(ComposedRewriting(r.rewritings, other_similar_axioms))
        return other_rewritings

    @staticmethod
    def type_based_constraints(qvar: FNode, other_qvar: FNode, other_axiom: 'Axiom',
                               all_similar_axioms: Set['Axiom']) -> Set[ComposedRewriting]:

        rewritings: Set[ComposedRewriting] = SortedSet()
        qvar_typ = qvar.get_type()

        for similar_axiom in all_similar_axioms:
            if qvar_typ in similar_axiom.types_to_functions.keys():
                for equivalent_expr in similar_axiom.types_to_functions[qvar_typ]:
                    simple_rewritings: Set[Rewriting] = SortedSet({Rewriting(qvar, equivalent_expr, False),
                                                                   Rewriting(other_qvar, equivalent_expr, False)})
                    other_axioms: Set[Axiom] = SortedSet({similar_axiom, other_axiom}, key=lambda x: x.axiom_id)
                    rewritings.add(ComposedRewriting(simple_rewritings, other_axioms))

        return rewritings

    def unification(self, functions1: Set[FNode], functions2: Set[FNode], skolem_functions: Set[FNode],
                    qvars: Set[FNode], other_axiom: 'Axiom', similar_axioms: Set['Axiom'],
                    type_constraints: bool, limit: int) -> Set[ComposedRewriting]:

        rewritings: Set[ComposedRewriting] = SortedSet()
        names_to_functions1 = self.names_to_functions(functions1)
        names_to_functions2 = self.names_to_functions(functions2)
        common_names = SortedSet(names_to_functions1.keys(), key=lambda x: x).intersection(
                       SortedSet(names_to_functions2.keys(), key=lambda x: x))

        for function_name in common_names:
            first_funs = names_to_functions1[function_name]
            other_funs = names_to_functions2[function_name]
            for first_fun in first_funs:
                len_first_args = len(first_fun.args())
                for other_fun in other_funs:
                    len_other_args = len(other_fun.args())
                    if len_first_args != len_other_args:
                        continue

                    # alternative rewritings for each hole (i.e., each argument of the function)
                    rewritings_list: List[Rewriting] = []
                    equivalent = equivalent_functions(first_fun, other_fun, skolem_functions, qvars, rewritings_list)
                    if equivalent:
                        simple_rewritings: Set[Rewriting] = SortedSet()
                        extended_rewritings: List[Set[ComposedRewriting]] = []
                        found_type_based_rewritings: bool = False
                        for r in rewritings_list:
                            simple_rewritings.add(r)
                            equivalent_expr = r.equivalent_expr

                            # transform r in a Composed rewriting (used for the extended rewritings)
                            rw: Set[ComposedRewriting] = SortedSet({ComposedRewriting({r}, {other_axiom})})
                            if type_constraints and len(similar_axioms) != 0 and equivalent_expr in qvars:
                                # rewriting of the type x = y, which might be too imprecise
                                type_based_constraints = self.type_based_constraints(r.qvar, equivalent_expr,
                                                                                     other_axiom, similar_axioms)
                                if len(type_based_constraints) > 0:
                                    found_type_based_rewritings = True
                                    extended_rewritings.append(type_based_constraints)
                                else:
                                    extended_rewritings.append(rw)
                            else:
                                extended_rewritings.append(rw)

                        if len(simple_rewritings) > 0:
                            other_axioms: Set[Axiom] = SortedSet({other_axiom}, key=lambda x: x.axiom_id)
                            rewritings.add(ComposedRewriting(simple_rewritings, other_axioms))

                            if type_constraints and found_type_based_rewritings:
                                rewritings.update(self.type_based_rewritings(extended_rewritings, limit))
        return rewritings

    @staticmethod
    def type_based_rewritings(extended_rewritings: List[Set[ComposedRewriting]], limit: int) -> Set[ComposedRewriting]:
        type_rewritings: Set[ComposedRewriting] = SortedSet()

        for extended_rewritings_product in itertools.product(*extended_rewritings):
            # merge all the composed rewritings into one
            rewritings: Set[Rewriting] = SortedSet()
            similar_axioms: Set[Axiom] = SortedSet(key=lambda x: x.axiom_id)

            for r in extended_rewritings_product:
                rewritings.update(r.rewritings)
                similar_axioms.update(r.similar_axioms)

            type_rewritings.add(ComposedRewriting(rewritings, similar_axioms))

            if len(type_rewritings) == limit:
                break

        return type_rewritings

    @staticmethod
    def names_to_functions(functions: Set[FNode]) -> Dict[str, Set[FNode]]:
        names_map: Dict[str, Set[FNode]] = OrderedDict()
        for f in functions:
            name = f.get_function_name()
            if name not in names_map.keys():
                names_map[name] = SortedSet([f], key=lambda x: True)
            else:
                names_map[name].add(f)
        return names_map
