###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2022 ETH Zurich.
###############################################################################

from typing import Set, Callable, List, Iterable

from sortedcontainers import SortedSet

from pysmt.fnode import FNode
from pysmt.formula import FormulaManager
from pysmt.shortcuts import Plus, Minus
from pysmt.typing import _FunctionType
from src.algorithms.formulas.rewritings import Rewriting
from collections import deque


def collect(node: FNode, what: Callable[[FNode], bool], result: Iterable[FNode]):
    if isinstance(result, Set) or isinstance(result, SortedSet):
        collect_in_set(node, what, result)
    elif isinstance(result, List):
        collect_in_list(node, what, result)
    else:
        raise Exception("function collect not implemented for results of types other than Set or List")


def collect_in_set(node: FNode, what: Callable[[FNode], bool], result: Set[FNode]):
    if what(node):
        result.add(node)
    for child in node.args():
        collect_in_set(child, what, result)


def collect_top_level(node: FNode, what: Callable[[FNode], bool], result: Set[FNode]):
    if what(node):
        result.add(node)
    else:
        for child in node.args():
            collect_top_level(child, what, result)


def collect_in_list(node: FNode, what: Callable[[FNode], bool], result: List[FNode]):
    if what(node):
        result.append(node)
    for child in node.args():
        collect(child, what, result)


def collect_disjuncts(node: FNode) -> List[FNode]:
    disjuncts: List[FNode] = []
    all_disjuncts(node, disjuncts)
    return disjuncts


def all_disjuncts(node: FNode, result: List[FNode]):
    if node.is_or():
        for disj in node.args():
            all_disjuncts(disj, result)
    else:
        result.append(node)


def inner_disjuncts(node: FNode) -> List[FNode]:
    if node.is_quantifier():
        disjuncts: List[FNode] = collect_disjuncts(node.arg(0))
    else:
        disjuncts: List[FNode] = collect_disjuncts(node)
    return disjuncts


def contains_and(node: FNode) -> bool:
    return '&' in str(node)


def collect_quantifiers(node: FNode) -> Set[FNode]:
    quants = []
    q = deque([node])
    while q:
        n = q.popleft()
        if n.is_quantifier():
            quants.append(n)
        for ch in n.args():
            q.append(ch)
    return SortedSet(quants, key=lambda x: True)

    # quantifiers: Set[FNode] = SortedSet(key=lambda x: True)
    # collect(node, lambda n: n.is_quantifier(), quantifiers)
    # return quantifiers


def collect_quantified_variables(quantifiers: Set[FNode], only_in_body=False):
    qvars_list = [qvar for quantifier in quantifiers for qvar in quantifier.quantifier_vars()]
    all_qvars = SortedSet(qvars_list, key=lambda a: a.symbol_name())

    assert len(qvars_list) == len(all_qvars), "the names of the quantified variables should be globally unique"

    if only_in_body:
        res = SortedSet([], key=lambda a: a.symbol_name())
        [collect_in_set(quant.arg(0), lambda x: x in all_qvars, res) for quant in quantifiers]
        return res
    else:
        return all_qvars


def collect_triggers(alternative_triggers: List[FNode]) -> Set[FNode]:
    triggers: Set[FNode] = SortedSet()
    for alternative_trigger in alternative_triggers:
        for trigger in alternative_trigger:
            triggers.add(trigger)
    return triggers


def is_function(symbol: FNode):
    return symbol._is_fun_application


def is_constant(symbol: FNode):
    typ = symbol.get_type()
    return not isinstance(typ, _FunctionType)


# remove terms that are contained in other terms
def remove_redundant_terms(terms: Set[FNode]) -> Set[FNode]:
    minimized_terms: Set[FNode] = SortedSet(key=lambda x: str(x))
    for term in terms:
        contained = False
        for other_term in terms:
            if other_term is not term and term in other_term.all_args():
                contained = True
                break
        if not contained:
            minimized_terms.add(term)
    return minimized_terms


def equivalent_functions(f: FNode, g: FNode, skolem_functions: Set[FNode], qvars: Set[FNode],
                         rewritings: List[Rewriting]) -> bool:
    if f == g:
        return True
    if not f.is_function_application() or not g.is_function_application():
        return False
    if f.get_function_name() is not g.get_function_name():
        return False
    for arg1, arg2 in zip(f.args(), g.args()):
        if not can_be_rewritten(arg1, arg2, skolem_functions, qvars, rewritings):
            if (is_function(arg1) and arg1.function_name() in skolem_functions) or \
               (is_function(arg2) and arg2.function_name() in skolem_functions):
                # this is a special case of unification, where one of the args is a skolem function
                # we do not identify rewritings, but we allow them to unify
                continue
            else:
                return False
    return True


# returns true if arg1 can be rewrite as arg2 (or vice versa)
def can_be_rewritten(arg1: FNode, arg2: FNode, skolem_functions: Set[FNode], qvars: Set[FNode],
                     rewritings: List[Rewriting]) -> bool:
    if arg1 == arg2:
        return True
    # arg1 is a quantified variable or arg2 is a quantified variable
    if arg1 in qvars or arg2 in qvars:
        record_rewritings(arg1, arg2, qvars, rewritings)
        return True
    if arg1.is_minus():
        left = arg1.arg(0)
        right = arg1.arg(1)
        if left in qvars and right not in qvars:
            # qvar - constant
            rhs = Plus(arg2, right)
            record_rewritings(left, rhs, qvars, rewritings)
            return True
    if arg1.is_plus():
        left = arg1.arg(0)
        right = arg1.arg(1)
        if left in qvars and right not in qvars:
            # qvar + constant
            rhs = Minus(arg2, right)
            record_rewritings(left, rhs, qvars, rewritings)
            return True
    if equivalent_functions(arg1, arg2, skolem_functions, qvars, rewritings):
        return True
    return False


def record_rewritings(arg1: FNode, arg2: FNode, qvars: Set[FNode], rewritings: List[Rewriting]):
    if arg1 in qvars:
        rewritings.append(Rewriting(arg1, arg2, arg2 in qvars))
    elif arg2 in qvars:
        rewritings.append(Rewriting(arg2, arg1, False))


def extract_prenex_body(node: FNode, mgr: FormulaManager) -> FNode:
    """
    This function simply removes all quantors and quantified variable bindings from the formula.
    Each quantifier variable in `node` turns into a free variable in the result.
    """

    while node.is_quantifier():
        node = node.arg(0)

    new_children = []
    for child in node.args():
        new_child = extract_prenex_body(child, mgr)
        new_children.append(new_child)

    new_node = mgr.create_node(node.node_type(), tuple(new_children), payload=node._content.payload)
    return new_node
