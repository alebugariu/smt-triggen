###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2022 ETH Zurich.
###############################################################################

import multiprocessing
from typing import List, Tuple, Set

from sortedcontainers import SortedSet

from pysmt.fnode import FNode
from src.algorithms.formulas.fresh import FreshVariable
from src.utils.ast import remove_redundant_terms
from src.utils.file import add_fname_suffix
from src.utils.smt import run_test, smt2


class Dummy:

    def __init__(self, name: str, suffix: str, terms: Set[FNode], fresh_vars: Set[FreshVariable]):
        self.name = name
        self.suffix = suffix
        self.terms = terms
        self.fresh_vars = fresh_vars
        self.fresh_vars_names: Set[str] = SortedSet({fresh_var.var_name for fresh_var in fresh_vars}, key=lambda x: x)
        self.abstract_terms = self.abstract(terms)

    def __str__(self):
        return self.dummy_constraint(self.name, self.abstract_terms, self.suffix)

    def __eq__(self, other):
        return SortedSet(self.abstract_terms, lambda x: x) == SortedSet(other.abstract_terms, lambda x: x)

    def __hash__(self):
        dummy_hash = 0
        for term in self.abstract_terms:
            dummy_hash += hash(term)
        return dummy_hash

    def abstract(self, terms: Set[FNode]) -> List[str]:
        terms_str: List[str] = list(map(lambda t: smt2(t), terms))
        var_index = 0
        for index in range(len(terms_str)):
            term = terms_str[index]
            while term:
                if " " in term:
                    space_index = term.index(" ")
                    old_name = term[0: space_index].replace(")", "")
                else:
                    # last element
                    space_index = -1
                    old_name = term.replace(")", "")
                if old_name in self.fresh_vars_names:
                    new_name = "_" + str(var_index)
                    term = self.rename(terms_str[index], old_name, new_name)
                    terms_str[index] = term
                    # propagate the renaming to the other terms
                    for other_index, other_term in enumerate(terms_str[index + 1:]):
                        terms_str[index + 1 + other_index] = self.rename(other_term, old_name, new_name)
                    var_index += 1
                    if space_index == -1:
                        term = []
                    term = term[len(new_name)+1:]
                elif space_index == -1:
                    term = []
                else:
                    term = term[space_index+1:]
        return terms_str

    @staticmethod
    def rename(term: str, old_name: str, new_name: str) -> str:
        if old_name not in term:
            return term

        renamed_term = ""
        for word in term.split(" "):
            if word.replace(")", "") == old_name:
                renamed_term += word.replace(old_name, new_name) + " "
            else:
                renamed_term += word + " "
        return renamed_term[:-1]

    @staticmethod
    def declare(name: str, types: Tuple[str, ...], constraint: str) -> List[str]:
        dummy_declarations: List[str] = []
        # Dummy declaration
        types_str = " ".join(types)
        dummy_declarations.append("(declare-fun %s %s)" % (name, "(%s) %s" % (types_str, "Bool")))
        # Dummified triggering term
        dummy_declarations.append("(assert %s)" % constraint)
        return dummy_declarations

    @staticmethod
    def dummy_constraint(name: str, terms: List[str], suffix: str) -> str:
        return "(! (%s %s) :named Dummy_%s )" % (name, " ".join(terms), suffix)

    def dummified_constraints(self) -> List[str]:
        types_str = tuple(map(lambda t: str(t.get_type()), self.terms))
        terms_str: List[str] = list(map(lambda t: smt2(t), self.terms))
        return self.declare(self.name, types_str, self.dummy_constraint(self.name, terms_str, self.suffix))

    def minimize_linear(self, file_name: str, test_file_name: str, timeout_per_query: float,
                        expected_outcome: str, partial_result) -> str:
        if len(self.terms) == 1:
            return self.dummy_constraint(self.name, self.abstract_terms, self.suffix)

        required_terms: Set[FNode] = SortedSet(key=lambda x: str(x))
        not_required_terms: Set[FNode] = SortedSet(key=lambda x: str(x))
        for index, term in enumerate(self.terms):
            remaining_terms = self.terms.difference({term}).difference(not_required_terms)

            if len(remaining_terms) == 0:
                # the last term is the only required term
                required_terms.add(term)
                break

            remaining_types_str = tuple(map(lambda t: str(t.get_type()), remaining_terms))
            remaining_terms_str: List[str] = list(map(lambda t: smt2(t), remaining_terms))
            name = self.name + str(index)

            constraint = self.dummy_constraint(name, remaining_terms_str, self.suffix + "_" + str(index))

            dummy_constraints = self.declare(name, remaining_types_str, constraint)
            outcome = run_test(file_name, test_file_name, self.fresh_vars, dummy_constraints,
                               len(remaining_terms), timeout_per_query)

            if outcome == "timeout" and expected_outcome != "timeout":
                # we cannot further minimize this dummy efficiently
                return self.abstract_constraint(self.terms.difference(not_required_terms), over_approximation=True)

            if not (outcome == expected_outcome):
                # required term
                required_terms.add(term)
            else:
                not_required_terms.add(term)
                if partial_result is not None:
                    # store partial results
                    partial_result.value = self.abstract_constraint(self.terms.difference(not_required_terms),
                                                                    over_approximation=True).encode()

        if len(required_terms) == len(self.terms):
            # all the terms are required
            return self.dummy_constraint(self.name, self.abstract_terms, self.suffix)

        return self.abstract_constraint(required_terms)

    def abstract_constraint(self, required_terms, over_approximation=False):
        required_indexes = sorted([list(self.terms).index(term) for term in required_terms])
        required_abstract_terms = [self.abstract_terms[index] for index in required_indexes]
        if over_approximation:
            return self.dummy_constraint(self.name + "_large", required_abstract_terms, self.suffix + "_large")
        return self.dummy_constraint(self.name, required_abstract_terms, self.suffix)

    def minimize(self, file_name: str, test_file_name: str, timeout_per_query: float,
                 expected_outcome: str, partial_result) -> str:
        middle = int(len(self.terms) / 2)
        first_half_terms = self.terms[0: middle]
        second_half_terms = self.terms[middle:]
        while len(first_half_terms) > 0 or len(second_half_terms) > 0:
            outcome = ""
            if len(first_half_terms) > 0:

                first_half_types_str = tuple(map(lambda t: str(t.get_type()), first_half_terms))
                first_half_terms_str: List[str] = list(map(lambda t: smt2(t), first_half_terms))

                constraint = self.dummy_constraint(self.name, first_half_terms_str, self.suffix)

                dummy_constraints = self.declare(self.name, first_half_types_str, constraint)
                outcome = run_test(file_name, test_file_name, self.fresh_vars, dummy_constraints,
                                   len(first_half_terms), timeout_per_query)

                if outcome == expected_outcome:
                    if len(first_half_terms) == 1:
                        return self.abstract_constraint(first_half_terms)

                    if partial_result is not None:
                        # store partial results
                        partial_result.value = self.abstract_constraint(first_half_terms, True).encode()

                    middle = int(len(first_half_terms) / 2)
                    second_half_terms = first_half_terms[middle:]
                    first_half_terms = first_half_terms[0: middle]
            if len(second_half_terms) > 0 and outcome != expected_outcome:

                second_half_types_str = tuple(map(lambda t: str(t.get_type()), second_half_terms))
                second_half_terms_str: List[str] = list(map(lambda t: smt2(t), second_half_terms))

                constraint = self.dummy_constraint(self.name, second_half_terms_str, self.suffix)

                dummy_constraints = self.declare(self.name, second_half_types_str, constraint)
                outcome = run_test(file_name, test_file_name, self.fresh_vars, dummy_constraints,
                                   len(second_half_terms), timeout_per_query)

                if outcome == expected_outcome:
                    if len(second_half_terms) == 1:
                        return self.abstract_constraint(second_half_terms)

                    if partial_result is not None:
                        # store partial results
                        partial_result.value = self.abstract_constraint(second_half_terms, True).encode()

                    middle = int(len(second_half_terms) / 2)
                    first_half_terms = second_half_terms[0: middle]
                    second_half_terms = second_half_terms[middle:]
                else:
                    merged_terms: Set[FNode] = SortedSet(first_half_terms + second_half_terms, key=lambda x: str(x))
                    merged_dummy: Dummy = Dummy(self.name, self.suffix, merged_terms, self.fresh_vars)

                    if partial_result is not None:
                        # store partial results
                        partial_result.value = merged_dummy.abstract_constraint(merged_dummy.terms, True).encode()
                    return merged_dummy.minimize_linear(file_name, test_file_name, timeout_per_query, expected_outcome,
                                                        partial_result)

        raise Exception("We should have found the required terms already")


class ComposedDummy:

    def __init__(self, dummies: Set[Dummy], fresh_vars: Set[FreshVariable], file_name: str, timeout_per_query: float):
        self.dummies = dummies
        self.fresh_vars = fresh_vars
        self.file_name = file_name  # name of the standardized file
        self.test_file_name = add_fname_suffix(file_name, 'dummy_min')
        self.timeout_per_query = timeout_per_query

    def merge(self, all_dummies, remove_redundancy=True) -> Dummy:
        # merge the dummies into one and remove the redundant terms
        composed_terms: Set[FNode] = SortedSet(key=lambda x: str(x))
        for dummy in all_dummies:
            composed_terms.update(dummy.terms)

        if len(composed_terms) > 1 and remove_redundancy:
            composed_terms = remove_redundant_terms(composed_terms)

        return Dummy("dummy_merged", "merged", composed_terms, self.fresh_vars)

    # the expected_outcome is unsat when minimizing after we found the dummy,
    #  and timeout for identifying matching loops
    def minimize(self, expected_outcome: str, partial_result=None) -> str:
        if partial_result is not None:
            # compute partial results
            merged_dummy: Dummy = self.merge(self.dummies, False)
            partial_result.value = merged_dummy.abstract_constraint(merged_dummy.terms, True).encode()

        middle = int(len(self.dummies) / 2)
        first_half = self.dummies[0: middle]
        second_half = self.dummies[middle:]
        while len(first_half) > 0 or len(second_half) > 0:
            outcome = ""
            if len(first_half) > 0:
                dummy_constraints: List[str] = []
                for dummy in first_half:
                    dummy_constraints += dummy.dummified_constraints()

                outcome = run_test(self.file_name, self.test_file_name, self.fresh_vars, dummy_constraints,
                                   len(first_half), self.timeout_per_query)
                if outcome == expected_outcome:
                    if len(first_half) == 1:
                        return first_half[0].minimize(self.file_name, self.test_file_name,
                                                      self.timeout_per_query, expected_outcome, partial_result)

                    if partial_result is not None:
                        # compute partial results
                        merged_dummy: Dummy = self.merge(first_half, False)
                        partial_result.value = merged_dummy.abstract_constraint(merged_dummy.terms, True).encode()

                    middle = int(len(first_half) / 2)
                    second_half = first_half[middle:]
                    first_half = first_half[0: middle]
                    
            if len(second_half) > 0 and outcome != expected_outcome:
                dummy_constraints: List[str] = []
                for dummy in second_half:
                    dummy_constraints += dummy.dummified_constraints()

                outcome = run_test(self.file_name, self.test_file_name, self.fresh_vars, dummy_constraints,
                                   len(second_half), self.timeout_per_query)
                if outcome == expected_outcome:
                    if len(second_half) == 1:
                        return second_half[0].minimize(self.file_name, self.test_file_name, self.timeout_per_query,
                                                       expected_outcome, partial_result)

                    if partial_result is not None:
                        # compute partial results
                        merged_dummy: Dummy = self.merge(second_half, False)
                        partial_result.value = merged_dummy.abstract_constraint(merged_dummy.terms, True).encode()

                    middle = int(len(second_half) / 2)
                    first_half = second_half[0: middle]
                    second_half = second_half[middle:]
                else:
                    merged_dummy = self.merge(first_half + second_half)

                    if partial_result is not None:
                        # store partial results
                        partial_result.value = merged_dummy.abstract_constraint(merged_dummy.terms, True).encode()
                    return merged_dummy.minimize(self.file_name, self.test_file_name, self.timeout_per_query,
                                                 expected_outcome, partial_result)

        raise Exception("We should have found the dummy already")

    def run(self, partial_result) -> str:
        return self.minimize(expected_outcome="unsat", partial_result=partial_result)


class DummyWorker:

    def __init__(self, dummy: 'ComposedDummy'):
        self.dummy = dummy
        self.result = multiprocessing.Array('c', 100000)
        self.partial_result = multiprocessing.Array('c', 100000)

    def run(self):
        try:
            res = self.dummy.run(self.partial_result)
            self.result.value = str(res).encode()
        except Exception as e:
            self.result.value = ("crash (%s)" % str(e)).encode()

