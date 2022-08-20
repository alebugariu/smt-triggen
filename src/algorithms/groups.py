###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2022 ETH Zurich.
###############################################################################

from collections import OrderedDict
from typing import List, Dict, Set, Tuple

from sortedcontainers import SortedSet

from pysmt.fnode import FNode
from pysmt.shortcuts import And
from src.algorithms.formulas.axiom import Axiom
from src.algorithms.formulas.group import Group
from src.algorithms.formulas.synthesized import SatFormula, Formula
from src.algorithms.formulas.rewritings import Rewriting, ComposedRewriting
from src.frontend.optset import Optset
from src.session.session import Session
from src.session.tactic import Tactic
from src.algorithms.clustering.similarity import SimilarityIndex
from src.algorithms.singletons import IndividualAxiomTester
from src.algorithms.wrappers import PhaseSwitcher
from src.utils.ast import collect, collect_triggers
from src.utils.enums import SmtResponse
from src.utils.smt import smt2, smt2_list


class GroupsAxiomTester(IndividualAxiomTester):

    DBG_PREFIX = "GroupsAxiomTester"

    def __init__(self, filename: str, logs_dir=None, opt: Optset=Optset()):

        super().__init__(filename=filename, logs_dir=logs_dir, opt=opt)

        with PhaseSwitcher(self, "init_groups", next_phase_name="ready"):
            self.opt = opt

            self.simpler_sat_formulas: Dict[int, Dict[str, List[SatFormula]]] = OrderedDict()
            self.number_of_tests: Dict[str, int] = OrderedDict()
            self.depth = 0
            self.one_cluster: bool = False

            # cache the already generated rewritings for a group
            self.generated_rewritings: Set[ComposedRewriting] = SortedSet(key=lambda x: True)
            # cache the already generated groups
            self.generated_groups: Set[Group] = SortedSet(key=lambda x: True)

            # cache the already generated formulas
            self.unsat_formulas: Set[Formula] = SortedSet(key=lambda x: True)
            self.sat_formulas: Set[Formula] = SortedSet(key=lambda x: True)

            self.triggering_terms: List[str] = []

    def direct_rewritings(self, axiom: Axiom, other_axioms: Set[Axiom], previous_rewritings: Set[Rewriting],
                          max_test_number: int) -> Tuple[Set[ComposedRewriting], Set[Axiom]]:

        with PhaseSwitcher(self, "find_rewritings(axiom=%s)" % axiom.axiom_id):
            return self.__direct_rewritings_impl(axiom, other_axioms, previous_rewritings, max_test_number)

    def __direct_rewritings_impl(self, axiom: Axiom, other_axioms: Set[Axiom], previous_rewritings: Set[Rewriting],
                                 max_test_number: int) -> Tuple[Set[ComposedRewriting], Set[Axiom]]:

        all_rewritings: Set[ComposedRewriting] = SortedSet()
        axioms_no_rewritings: Set[Axiom] = SortedSet(key=lambda x: True)

        if len(other_axioms) == 0:
            # no similar axioms
            return all_rewritings, axioms_no_rewritings

        for other_axiom in other_axioms:

            remaining_axioms = other_axioms.difference({other_axiom})

            rewritings: Set[ComposedRewriting] = axiom.unify(other_axiom, remaining_axioms, self.opt.type_constraints,
                                                             self.opt.top_level, max_test_number)
            if not rewritings.issubset(all_rewritings):  # at least one new rewriting
                all_rewritings.update(rewritings)
            elif len(rewritings) > 0 and axiom.parent_id == other_axiom.parent_id:  # at least one rewriting
                axioms_no_rewritings.add(other_axiom)
            elif not other_axiom.is_quantified():
                axioms_no_rewritings.add(other_axiom)

        if len(all_rewritings) == 0:
            # no rewritings
            return all_rewritings, axioms_no_rewritings

        previously_rewritten_qvars = SortedSet([rewriting.qvar for rewriting in previous_rewritings],
                                               key=lambda x: x.symbol_name())
        # remove the new rewritings for the previously rewritten variables
        remaining_rewritings = [r for r in all_rewritings if r.qvars.isdisjoint(previously_rewritten_qvars)]
        all_rewritings = SortedSet(remaining_rewritings)

        return all_rewritings, axioms_no_rewritings

    def get_group(self, similar_axioms: SortedSet) -> Group or None:
        for group in self.generated_groups:
            if group.similar_axioms == similar_axioms:
                return group
        return None

    def already_invalid(self, similar_axioms: SortedSet) -> bool:
        for group in self.generated_groups:
            if group.similar_axioms.issubset(similar_axioms) and len(group.valid_instantiations) == 0:
                return True
        return False

    # create groups (subsets) of rewritings with max one rewriting per quantified variable within the subset
    def groups_to_rws(self, rewritings: Set[ComposedRewriting], axioms_no_rewritings: Set[Axiom]):
        previous: Dict[int, Set[ComposedRewriting]] = OrderedDict({1: SortedSet()})
        generated = False
        for rw in rewritings:
            if not self.already_generated(rw):
                # add all the quantifier-free axioms that share symbols, but don't have rewritings and
                # all the axioms with the same parent that share symbols, but don't have new rewritings
                similar_axioms = rw.similar_axioms.union(axioms_no_rewritings)
                group: Group or None = self.get_group(similar_axioms)
                if group is None:
                    if self.already_invalid(similar_axioms):  # we already know that a subset of these axioms
                        continue                              # do not have valid instantiations
                    group = Group(similar_axioms, self.parser.env.formula_manager)
                    self.generated_groups.add(group)
                elif not group.only_qfree_axioms and len(group.valid_instantiations) == 0:
                    # we have already generated this group, and we know it does not have valid instantiations
                    continue
                generated = True
                previous[1].add(rw)
                self.generated_rewritings.add(rw)
                yield group, rw.rewritings

        rw_len = 2
        while generated and rw_len <= len(rewritings):
            previous[rw_len] = SortedSet()
            generated = False
            for rw in previous[rw_len - 1]:
                with_distinct_qvars: Set[ComposedRewriting] = SortedSet()
                with_distinct_qvars.update([other for other in previous[1] if rw.same_subset(other)])
                for other in with_distinct_qvars:
                    composed_rewriting = rw.join(other)
                    if not self.already_generated(composed_rewriting):
                        # add all the quantifier-free axioms that share symbols, but don't have rewritings and
                        # all the axioms with the same parent that share symbols, but don't have new rewritings
                        similar_axioms = rw.similar_axioms.union(axioms_no_rewritings)
                        group: Group or None = self.get_group(similar_axioms)
                        if group is None:
                            if self.already_invalid(similar_axioms):  # we already know that a subset of these axioms
                                continue                              # do not have valid instantiations
                            group = Group(similar_axioms, self.parser.env.formula_manager)
                            self.generated_groups.add(group)
                        elif len(group.valid_instantiations) == 0:  # we have already generated this group,
                            continue                                # and we know it does not have valid instantiations
                        generated = True
                        previous[rw_len].add(composed_rewriting)
                        self.generated_rewritings.add(composed_rewriting)
                        yield group, rw.rewritings
            rw_len += 1

    def already_generated(self, composed_rewriting) -> bool:
        for r in self.generated_rewritings:
            if r.equivalent(composed_rewriting):
                return True
        return False

    def already_unsat(self, formula: Formula) -> bool:
        for unsat_formula in self.unsat_formulas:
            if unsat_formula.contained(formula):
                return True
        return False

    def already_sat(self, formula: Formula) -> bool:
        return formula in self.sat_formulas

    def test_cluster(self, axiom: Axiom, axiom_number: int, session: Session, previous_formula: SatFormula,
                     all_rewritings: Set[ComposedRewriting], axioms_no_rewritings: Set[Axiom],
                     current_sat_formulas: List[SatFormula], max_test_number: int) -> bool:

        test_number = 0
        self.generated_rewritings.clear()

        negated_axiom = previous_formula.negated_axiom

        for rw_number, (group, rws) in enumerate(self.groups_to_rws(all_rewritings, axioms_no_rewritings)):
            session.push("Rewritings No %d: processing..." % rw_number)

            triggers: Set[FNode] = set(axiom.triggers).union(group.triggers)
            qvars: Set[FNode] = group.qvars.difference(previous_formula.qvars)
            # declare all the quantified variables (we use them in instantiations, without fresh names)
            self.declare_qvars(qvars, session)

            rewritings: Set[Rewriting] = rws.union(previous_formula.rewritings)
            triggers.update(previous_formula.triggers)
            similar_axioms: Set[Axiom] = group.similar_axioms.union(previous_formula.similar_axioms)

            proto_terms = self.construct_proto_triggering_terms(triggers, group.qvars.union(previous_formula.qvars),
                                                                group.constants.union(previous_formula.constants),
                                                                rewritings)
            assert len(proto_terms.terms) > 0, "no terms were picked; nothing to generate"

            # Handle the special case in which the proto terms are concrete,
            # i.e. all ast leaves are predefined constants.
            if proto_terms.just_constants():

                msg = "Skipping the stack since the term's leaves are all constants."
                self.print_dbg(msg)
                session.log_comment(msg)
                inconsistent, term = self.handle_special_case(proto_terms, axiom.axiom_id,
                                                              self.number_of_tests[axiom.axiom_id])
                test_number += 1
                self.number_of_tests[axiom.axiom_id] += 1

                if inconsistent:
                    self.triggering_terms.append(term)
                    if not self.opt.multiple_dummies:
                        session.pop("Rewritings No %d: done [stack skipped, inconsistency found]" % rw_number)
                        return True
                    elif test_number == max_test_number:
                        session.pop("Rewritings No %d: done [stack skipped, max #tests reached]" % rw_number)
                        return len(self.triggering_terms) != 0
                    else:
                        session.pop("Rewritings No %d: done [stack skipped]" % rw_number)
                        continue
                elif test_number == max_test_number:
                    session.pop("Rewritings No %d: done [stack skipped, max #tests reached]" % rw_number)
                    return len(self.triggering_terms) != 0
                else:
                    session.pop("Rewritings No %d: done [stack skipped]" % rw_number)
                    continue

            self.add_rewritings(session, rws)

            for inst_number, inst in enumerate(self.valid_instantiations(group)):
                current_formula = Formula(negated_axiom, inst.union(previous_formula.instantiations),
                                          rewritings, triggers)

                if self.already_unsat(current_formula):
                    continue

                if self.already_sat(current_formula):
                    continue

                session.push("Instantiation No %d: processing..." % inst_number)
                session.add_constraints(smt2_list(inst), "instantiation constraints")

                result, status = self.synthesize_triggering_term(axiom_number, axiom.axiom_id, proto_terms, session,
                                                                 self.number_of_tests[axiom.axiom_id])
                inconsistent, term = result
                test_number += 1
                self.number_of_tests[axiom.axiom_id] += 1

                if status != SmtResponse.UNSAT:
                    # store the sat formulas not to test them again (for different transitivity depth)
                    self.sat_formulas.add(current_formula)
                else:
                    # store the unsat formulas to cut the search space afterwards
                    self.unsat_formulas.add(current_formula)

                if inconsistent:
                    self.triggering_terms.append(term)
                    if not self.opt.multiple_dummies:
                        session.pop("Instantiation No %d: done [inconsistency found]" % inst_number)
                        session.pop("Rewritings No %d: done [inconsistency found]" % rw_number)
                        return True

                if not inconsistent and self.depth < self.opt.max_depth:
                    if status != SmtResponse.UNSAT:
                        # store only the sat formulas which will be strengthened afterwards
                        sat_formula = SatFormula(session.simplified_log(), similar_axioms, current_formula)
                        current_sat_formulas.append(sat_formula)

                if test_number == max_test_number:
                    session.pop("Instantiation No %d: done [max #tests reached]" % inst_number)
                    session.pop("Rewritings No %d: done [max #tests reached]" % rw_number)
                    return len(self.triggering_terms) != 0

                session.pop("Instantiation No %d: done" % inst_number)
            session.pop("Rewritings No %d: done" % rw_number)
        return len(self.triggering_terms) != 0

    @staticmethod
    def valid_instantiations(group: Group):
        # we already computed them
        if len(group.valid_instantiations) > 0:
            yield from group.valid_instantiations

        else:
            all_instantiations: List[List[FNode]] = group.instantiations
            if len(all_instantiations) == 1:
                for inst in all_instantiations[0]:
                    valid_inst: Set[FNode] = {inst}
                    group.valid_instantiations.append(valid_inst)
                    yield valid_inst

            else:
                previous: Dict[int, Set[FNode]] = OrderedDict({2: SortedSet(key=lambda x: True)})

                for inst_0 in all_instantiations[0]:
                    for inst_1 in all_instantiations[1]:
                        node: FNode = And(inst_0, inst_1)
                        simple_node = node.simplify()
                        if not simple_node.is_false():  # if it's trivially false, then it is unsat by construction
                            previous[2].add(simple_node)

                valid = True
                inst_len = 3
                while valid and inst_len <= len(all_instantiations):
                    previous[inst_len] = SortedSet(key=lambda x: True)
                    valid = False
                    for prev_inst in previous[inst_len - 1]:
                        for current_inst in all_instantiations[inst_len - 1]:  # the first index is 0
                            node: FNode = And(prev_inst, current_inst)
                            simple_node = node.simplify()
                            if not simple_node.is_false():  # if it's trivially false, then it is unsat by construction
                                previous[inst_len].add(simple_node)
                                valid = True
                    inst_len += 1

                for node in previous[inst_len - 1]:
                    valid_inst: Set[FNode] = SortedSet(node.args(), key=lambda x: True)
                    group.valid_instantiations.append(valid_inst)
                    yield valid_inst

    def build_clusters_from_axiom(self, axiom: Axiom, axiom_number: int, total_formulas: int, other_axioms: Set[Axiom],
                                  session: Session, previous_formula: SatFormula) -> bool:
        negated_axiom: Axiom = previous_formula.negated_axiom
        previous_rewritings: Set[Rewriting] = previous_formula.rewritings

        current_sat_formulas: List[SatFormula] = []

        session.push("Axiom %s: processing..." % axiom.axiom_id)

        max_test_number = max(1, int(self.opt.max_formulas / total_formulas / self.depth))

        all_rewritings, axioms_no_rewritings = self.direct_rewritings(axiom, other_axioms, previous_rewritings,
                                                                      max_test_number)
        if len(all_rewritings) == 0:
            session.pop("Axiom %s: done" % axiom.axiom_id)
            return len(self.triggering_terms) != 0

        inconsistent = self.test_cluster(axiom, axiom_number, session, previous_formula, all_rewritings,
                                         axioms_no_rewritings, current_sat_formulas, max_test_number)
        if inconsistent and not self.opt.multiple_dummies:
            session.pop("Axiom %s: done [inconsistency found]" % axiom.axiom_id)
            return True

        if self.depth < self.opt.max_depth:
            # store only the formulas which will be used afterwards
            self.simpler_sat_formulas[self.depth].update({negated_axiom.axiom_id: current_sat_formulas})
        session.pop("Axiom %s: done" % axiom.axiom_id)
        return len(self.triggering_terms) != 0

    def test_negated_axiom(self, axiom: Axiom, axiom_number: int, session: Session, tactic: Tactic) -> bool:
        current_formula = Formula(axiom, set(), set(), axiom.triggers)

        if self.already_unsat(current_formula):
            return len(self.triggering_terms) != 0

        if self.already_sat(current_formula):
            return len(self.triggering_terms) != 0

        session.push("Axiom %s: processing the negation..." % axiom.axiom_id)

        # declare all the quantified variables of the negated axiom
        self.declare_qvars(axiom.quantified_variables, session)

        new_decls, neg_conditions = self.std_negate(tactic, axiom)
        session.declare_consts(new_decls, comment="constants for the negation")
        session.add_constraints(neg_conditions)

        # check if the negated axiom, by itself, is satisfiable
        proto_terms = self.construct_proto_triggering_terms(axiom.triggers, axiom.quantified_variables, axiom.constants)
        result, status = self.synthesize_triggering_term(axiom_number, axiom.axiom_id, proto_terms, session,
                                                         self.number_of_tests[axiom.axiom_id])
        self.number_of_tests[axiom.axiom_id] += 1
        inconsistent, term = result

        if status != SmtResponse.UNSAT:
            # store the sat formulas not to test them again (for different transitivity depth)
            self.sat_formulas.add(current_formula)
        else:
            # store the unsat formulas to cut the search space afterwards
            self.unsat_formulas.add(current_formula)

        if inconsistent:
            self.triggering_terms.append(term)
            if not self.opt.multiple_dummies:
                session.pop("Axiom %s: negation done [inconsistency found]" % axiom.axiom_id)
                return True
        if not inconsistent and status != SmtResponse.UNSAT and self.depth < self.opt.max_depth:
            # store only the sat formulas which will be strengthened afterwards
            sat_formula = SatFormula(session.simplified_log(), set(), current_formula)
            self.simpler_sat_formulas[self.depth].update({axiom.axiom_id: [sat_formula]})

        session.pop("Axiom %s: negation done" % axiom.axiom_id)
        return len(self.triggering_terms) != 0

    def increase_frequency(self, axiom: Axiom, frequency_diff: int, other_axioms: Set[Axiom]):
        while frequency_diff > 0:
            axiom_copy = self.duplicate_axiom(axiom)
            other_axioms.add(axiom_copy)
            frequency_diff -= 1

    def strengthen_formula(self, formula: SatFormula, formula_number: int, total_formulas: int,
                           axiom_number: int, session: Session) -> bool:
        session.push("Formula No %d: processing..." % formula_number)
        session.add_previous_commands(formula.declarations_and_assertions)

        negated_axiom = formula.negated_axiom

        if self.depth == 1:
            other_axioms: Set[Axiom] = SortedSet(key=lambda x: x.axiom_id)
            frequency_diff = self.opt.max_axiom_frequency - 1
            if frequency_diff > 0:
                self.increase_frequency(negated_axiom, frequency_diff, other_axioms)

            for an_axiom in negated_axiom.axioms_with_common_symbols:
                other_axioms.add(an_axiom)
                if frequency_diff > 0 and an_axiom.is_quantified():
                    self.increase_frequency(an_axiom, frequency_diff, other_axioms)

            if len(other_axioms) != 0:
                inconsistent = self.build_clusters_from_axiom(negated_axiom, axiom_number, total_formulas, other_axioms,
                                                              session, formula)
                if inconsistent and not self.opt.multiple_dummies:
                    # close the unbalanced push commands from formula.declarations_and_assertions
                    for index in range(0, formula.unbalanced_push_commands()):
                        session.pop("Adding missing pop [inconsistency found]")
                    session.pop("Formula No %d: done [inconsistency found]" % formula_number)
                    return True

        else:
            for similar_axiom_number, similar_axiom in enumerate(formula.similar_axioms):
                if not similar_axiom.is_quantified():
                    continue

                other_axioms: Set[Axiom] = SortedSet(key=lambda x: x.axiom_id)
                for an_axiom in similar_axiom.axioms_with_common_symbols.union({similar_axiom}):
                    axiom_frequency = self.frequency(an_axiom, formula.similar_axioms)
                    if an_axiom.axiom_id == negated_axiom.axiom_id:
                        frequency_diff = self.opt.max_axiom_frequency - 1 - axiom_frequency
                        if frequency_diff > 0:
                            self.increase_frequency(negated_axiom, frequency_diff, other_axioms)
                        continue

                    frequency_diff = self.opt.max_axiom_frequency - axiom_frequency
                    if axiom_frequency == 0:
                        other_axioms.add(an_axiom)
                        frequency_diff -= 1
                    if frequency_diff > 0 and an_axiom.is_quantified():
                        self.increase_frequency(an_axiom, frequency_diff, other_axioms)

                if len(other_axioms) != 0:
                    inconsistent = self.build_clusters_from_axiom(similar_axiom, similar_axiom_number, total_formulas,
                                                                  other_axioms, session, formula)
                    if inconsistent and not self.opt.multiple_dummies:
                        # close the unbalanced push commands from formula.declarations_and_assertions
                        for index in range(0, formula.unbalanced_push_commands()):
                            session.pop("Adding missing pop [inconsistency found]")
                        session.pop("Formula No %d: done [inconsistency found]" % formula_number)
                        return True

        # close the unbalanced push commands from formula.declarations_and_assertions
        for index in range(0, formula.unbalanced_push_commands()):
            session.pop("Adding missing pop")

        session.pop("Formula No %d: done" % formula_number)
        return len(self.triggering_terms) != 0

    @staticmethod
    def frequency(axiom: Axiom, similar_axioms: List[Axiom]) -> int:
        return sum(axiom.axiom_id == an_axiom.axiom_id for an_axiom in similar_axioms)

    # creates a copy of the axiom, using fresh names for the quantified variables and the same axiom_id as the original
    def duplicate_axiom(self, axiom: Axiom) -> Axiom:
        axiom_str = smt2(axiom.axiom_node)
        axiom_copy_node = self.parse_exp(axiom_str)

        quants: Set[FNode] = SortedSet(key=lambda x: True)
        collect(axiom_copy_node, lambda n: n.is_quantifier(), quants)
        triggers: Set[FNode] = SortedSet(key=lambda x: True)

        for quant in quants:
            alternative_triggers = quant.quantifier_patterns()
            triggers.update(collect_triggers(alternative_triggers))
        # the copy has the same axiom_id as the original
        axiom_copy = Axiom(axiom.axiom_id, axiom_copy_node, quants, triggers, axiom.skolem_functions, axiom.parent_id)
        axiom_copy.axioms_with_common_symbols = axiom.axioms_with_common_symbols
        return axiom_copy

    @staticmethod
    def declare_qvars(qvars: Set[FNode], session: Session):
        for qvar in qvars:
            session.declare_const(qvar.symbol_name(), qvar.symbol_type(), comment="declaring quantified variable")

    @staticmethod
    def add_rewritings(session, rewritings: Set[Rewriting]):
        for rewriting in rewritings:
            session.add_constraint("(= %s %s)" % (smt2(rewriting.qvar), smt2(rewriting.equivalent_expr)), "rewriting")

    def run(self) -> Tuple[bool, List[str], float]:

        with Session(name="Models Session", preamble=self.m.get_setlogic() + self.preamble,
                     logs_path=self.models_logs_path, memory_limit_mb=6000) as session:
            with PhaseSwitcher(self, "skolemize_all"):

                axioms: List[Axiom] = self.skolemize_all(session)

                self.constants.update(self.get_constant_symbols())
                self.constant_nodes.update(SortedSet([const for const_set in self.constants.values() for
                                                      const in const_set], key=lambda x: True))
                self.declarations += self.get_all_decl_commands()
                self.fresh_var_nodes.update(SortedSet([fresh_var.var_node for fresh_vars in
                                                       self.fresh_vars_pool.values() for fresh_var in fresh_vars],
                                                      key=lambda x: x.symbol_name()))
                self.timeout_per_query += session._hard_timeout

                # declare the fresh variables:
                for fresh_vars in self.fresh_vars_pool.values():
                    for fresh_var in fresh_vars:
                        session.declare_const(fresh_var.var_name, fresh_var.var_type)

            with PhaseSwitcher(self, "run"):
                result = self.__run_impl(axioms, session)
                return result

    def __similarity_impl(self, skolemized_axioms: List[Axiom]):

        similarity_index = SimilarityIndex(axioms=skolemized_axioms, threshold=self.opt.similarity_threshold)
        clusters = similarity_index.get_clusters_of_similar_axioms(transitive=False)

        # the cluster includes all the axioms
        if len(clusters) == 1:
            self.one_cluster = True

    def __run_impl(self, axioms: List[Axiom], session: Session) -> Tuple[bool, List[str], float]:

        with Tactic(name="Skolemizer", preamble=self.preamble, memory_limit_mb=1000) as tactic:

            while not self.one_cluster and self.opt.similarity_threshold >= 0:
                with PhaseSwitcher(self, "similarity(%f)" % self.opt.similarity_threshold):
                    self.__similarity_impl(axioms)
                    axioms = sorted(axioms, key=lambda a: a.cluster_id)

                    if self.depth == 0:
                        # this branch should be executed only once
                        self.simpler_sat_formulas[self.depth] = OrderedDict()
                        for axiom_number, axiom in enumerate(axioms):
                            if not axiom.is_quantified():
                                continue
                            self.number_of_tests[axiom.axiom_id] = 0
                            inconsistent = self.test_negated_axiom(axiom, axiom_number, session, tactic)
                            if inconsistent:
                                if self.opt.multiple_dummies:
                                    print(self.triggering_terms[-1])
                                else:
                                    return True, self.triggering_terms, self.opt.similarity_threshold

                    self.depth = 1
                    if self.depth > self.opt.max_depth:
                        # test all the remaining dummies in batch mode
                        inconsistent = self.test_remaining_dummies()
                        if inconsistent:
                            if self.opt.multiple_dummies:
                                print(self.triggering_terms[-1])
                            else:
                                return True, self.triggering_terms, self.opt.similarity_threshold
                        return len(self.triggering_terms) != 0, self.triggering_terms, self.opt.similarity_threshold

                    while self.depth <= self.opt.max_depth:
                        self.simpler_sat_formulas[self.depth] = OrderedDict()

                        for axiom_number, axiom in enumerate(axioms):
                            if not axiom.is_quantified() or (not axiom.new_similar_axioms and
                                                             self.opt.max_axiom_frequency == 1):
                                continue
                            if axiom.axiom_id in self.simpler_sat_formulas[self.depth - 1].keys():

                                session.push("Axiom %s depth %d: processing..." % (axiom.axiom_id, self.depth))
                                previous_formulas = self.simpler_sat_formulas[self.depth - 1][axiom.axiom_id]

                                for number, formula in enumerate(previous_formulas):
                                    inconsistent = self.strengthen_formula(formula, number, len(previous_formulas),
                                                                           axiom_number, session)
                                    if inconsistent:
                                        if self.opt.multiple_dummies:
                                            print(self.triggering_terms[-1])
                                        else:
                                            return True, self.triggering_terms, self.opt.similarity_threshold

                                    # remove the formulas which were already strengthened, except from those for
                                    # self.depth = 0, which are computed only once
                                    if self.depth != 1:
                                        self.simpler_sat_formulas[self.depth - 1][axiom.axiom_id] = \
                                            previous_formulas[number+1:]

                                session.pop("Axiom %s depth %d: done" % (axiom.axiom_id, self.depth))
                                # remove the axioms whose formulas were already strengthened, except from those for
                                # self.depth = 0, which are computed only once
                                if self.depth != 1:
                                    del self.simpler_sat_formulas[self.depth - 1][axiom.axiom_id]

                        self.depth += 1

                    # test all the remaining dummies in batch mode
                    inconsistent = self.test_remaining_dummies()
                    if inconsistent:
                        if self.opt.multiple_dummies:
                            print(self.triggering_terms[-1])
                        else:
                            return True, self.triggering_terms, self.opt.similarity_threshold

                    self.opt.similarity_threshold = round(self.opt.similarity_threshold - 0.1, 2)
                    self.depth = 1

        return len(self.triggering_terms) != 0, self.triggering_terms, max(0.0, self.opt.similarity_threshold)

    def test_remaining_dummies(self) -> bool:
        if len(self.dummies) > 0:
            inconsistent, term = self.test_dummy_batch_mode()
            self.dummies.clear()
            if inconsistent:
                self.triggering_terms.append(term)
                return True
        return False


def main():
    inconsistent, dummies, final_sigma = GroupsAxiomTester("smt-inputs/paper/figure1.smt2",
                                                           opt=Optset(type_constraints=True)).run()
    if inconsistent:
        print("Found dummies: ", "\n\n".join(dummies))
    print("Final sigma: " + str(final_sigma))


if __name__ == "__main__":
    main()
