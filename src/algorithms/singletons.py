###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2024 ETH Zurich.
###############################################################################

import multiprocessing
import os
import io
import re
import shutil
import time
from collections import OrderedDict
from os.path import isdir, isfile, join, basename
from typing import List, Tuple, Dict, Set, Iterator

from sortedcontainers import SortedSet

from pysmt.fnode import FNode
from pysmt.shortcuts import Not, reset_env, Or, And
from pysmt.smtlib.parser import SmtLibZ3Parser, Tokenizer, PysmtTypeError, PysmtAnnotationsError, itertools
from pysmt.smtlib.script import SmtLibCommand
from pysmt.typing import PySMTType
from src.algorithms.formulas.fresh import FreshVariable
from src.algorithms.formulas.proto import ProtoTerms
from src.frontend.optset import Optset
from src.minimizer.minimizer import Minimizer
from src.algorithms.formulas.axiom import Axiom
from src.algorithms.formulas.dummy import Dummy, ComposedDummy, DummyWorker
from src.algorithms.exceptions import InferredPatternInvalid, PatternException
from src.algorithms.formulas.rewritings import Rewriting
from src.session.session import Session
from src.session.pattern_provider import PatternProvider
from src.session.tactic import Tactic
from src.utils.file import add_fname_suffix, ensure_dir_structure
from src.utils.preamble import SEED_MEMORY_OPTIONS, EMATCHING
from src.utils.smt import smt2, smt2_tup_tup, run_test
from src.utils.ast import collect, collect_triggers, collect_quantified_variables, collect_quantifiers, \
    collect_disjuncts, inner_disjuncts, contains_and, remove_redundant_terms
from src.utils.enums import SMTSolver, SmtResponse, SmtCommand
from src.utils.string import create_substituted_string
from src.algorithms.wrappers import PhaseSwitcher, AnnotationsPreserver, VariableNamesPreserver

# ATTENTION:
#  Do not remove the following imports:
#  They synchronize global variables across different modules
from src.session import debug_mode


def applyDefaultPySMTenvSettings(env):
    env.enable_infix_notation = True
    env.unique_names_guarantee = True


def resetPySMTenv():
    env = reset_env()
    applyDefaultPySMTenvSettings(env)


class IndividualAxiomTester:

    DBG_PREFIX = "IndividualAxiomTester"

    def __get_printable_str(self, msg) -> str:
        if self.phase:
            return "%s >> %s" % (self.phase, msg)
        else:
            return msg

    def print_dbg(self, msg, skip_prefix=False):
        if debug_mode.flag:
            if skip_prefix:
                print(msg)
            else:
                print("%s.DBG >> %s" % (self.DBG_PREFIX, self.__get_printable_str(msg)))

    def print(self, msg):
        print("%s >> %s" % (self.DBG_PREFIX, self.__get_printable_str(msg)))

    ALPH: str = SmtLibZ3Parser.TOKEN_ALPHABET

    USELESS_CMD_LIST = [SmtCommand.CHECK_SAT,  # We emit our own queries via sessions.
                        SmtCommand.EXIT,
                        SmtCommand.LABELS,
                        SmtCommand.GET_INFO,
                        SmtCommand.EVAL,
                        SmtCommand.ECHO]

    def cleanup(self):
        # delete temporary files
        if not debug_mode.flag:
            if isdir(self.tmp_file_dir):
                self.print("deleting directory with temporary files: %s" % self.tmp_file_dir)
                shutil.rmtree(self.tmp_file_dir)
            if isdir(self.logs_dir):
                self.print("deleting directory with temporary files: %s" % self.logs_dir)
                shutil.rmtree(self.logs_dir)

    def __init__(self, *args, **kwargs):
        self.phase = None
        with PhaseSwitcher(self, "init_singletons", next_phase_name="ready"):
            self.__init_impl(*args, **kwargs)

    def __init_impl(self, filename: str, tmp_file_dir: str or None = None, logs_dir: str or None = None,
                    opt: Optset = Optset()):

        self.start_time = time.perf_counter()
        self.filename = filename
        self.opt = opt

        # all the dummies ever generated
        self.all_dummies: Set[Dummy] = SortedSet(key=lambda x: x.name)
        # the dummies from the current batch
        self.dummies: Set[Dummy] = SortedSet(key=lambda x: x.name)

        # cache the previously generated models
        self.previous_models: Dict[ProtoTerms, List[Dict[FNode, FNode]]] = OrderedDict()

        self.solved_dir = os.path.join(os.path.dirname(filename), "solved")

        if tmp_file_dir:
            self.tmp_file_dir = tmp_file_dir
        else:
            self.tmp_file_dir = os.path.join(os.path.dirname(filename), "tmp")

        self.standardized_file_path = os.path.join(self.tmp_file_dir, os.path.basename(filename))

        m = Minimizer(filename, fout_name=self.standardized_file_path, auto_confg=False,
                      useless_commands=self.USELESS_CMD_LIST,
                      solver=SMTSolver.Z3, add_check_sat=False, validate=False,
                      remove_duplicates=(not self.opt.keep_duplicate_assertions))

        assert len(m.get_checksat_commands()) == 0

        self.num_axioms = len(m.get_assertions())
        self.m = m

        self.preamble = SEED_MEMORY_OPTIONS + \
                        m.get_declarations() + \
                        m.get_const_definitions()

        self.original_declarations: Set[str] = [decl[0] for decl in self.m.get_fun_const_decl_triplets()]

        # the following 5 fields are initialized (during) after Skolemization,
        # to include the Skolem declarations/functions
        self.declarations: List[str] = []
        # constants grouped by their type names
        self.constants: Dict[str, Set[FNode]] = OrderedDict()
        # all the constants
        self.constant_nodes: Set[FNode] = SortedSet(key=lambda x: True)
        # fresh variables grouped by their type names
        self.fresh_vars_pool: Dict[str, Set[FreshVariable]] = OrderedDict()
        # all the fresh variables (the FNodes)
        self.fresh_var_nodes: Set[FNode] = SortedSet(key=lambda x: x.symbol_name())

        self.timeout_per_query: int = 0  # will be updated with the session's timeout
        self.distributivity_level = 0

        # Make PySMT aware of the declared stuff
        resetPySMTenv()
        self._refresh_parser()

        self.suffix: str or None = None

        # To be used for saving data after the session run
        if logs_dir:
            assert isdir(logs_dir) or not isfile(logs_dir), "parameter logs_dir must be a directory"
        self.logs_dir = logs_dir

        if debug_mode.flag and logs_dir:
            self.skolems_logs_path = self.get_logs_path("skolems")
            self.patterns_logs_path = self.get_logs_path("patterns")
            self.models_logs_path = self.get_logs_path("models")
        else:
            self.skolems_logs_path = None
            self.patterns_logs_path = None
            self.models_logs_path = None

        # Internal options
        self.raise_on_axioms_with_incomplete_patterns = True  # True means that incomplete patterns will raise exception

    def get_logs_path(self, logs_for: str) -> str:
        return join(self.logs_dir, self.__basename("%s.smt2" % logs_for))

    def __basename(self, ending="") -> str:
        return add_fname_suffix(basename(self.filename), ending, delete_ext=True)

    def _register_namespace(self, ax_id: str, tnum=None, anum=None):
        self.suffix = ax_id
        if tnum is not None:
            self.suffix += "T%s" % tnum
        if anum is not None:
            self.suffix += "@%d" % anum

        if tnum is not None and anum is not None:
            self.print_dbg("Axiom %s, Test No: %d, Attempt No %d" % (ax_id, tnum, anum))
        elif anum is not None:
            self.print_dbg("Axiom %s, Attempt No %d" % (ax_id, anum))
        elif tnum is not None:
            self.print_dbg("Axiom %s, Test No: %d" % (ax_id, tnum))
        else:
            self.print_dbg("Axiom %s" % ax_id)

    def _refresh_parser(self):
        self.parser = SmtLibZ3Parser()
        with io.StringIO("\n".join(self.preamble)) as stream:
            self.preamble_script = self.parser.get_script(stream)

    def parse_cmd(self, command_str, with_annotations=True) -> SmtLibCommand:

        self.print_dbg("Parsing command: %s" % command_str)

        mgr = self.parser.env.formula_manager
        mgr.quants = set()

        with AnnotationsPreserver(self.preamble_script.annotations, with_annotations):
            with io.StringIO(command_str) as stream:
                tokenizer = Tokenizer(stream, interactive=False)
                command_gen = self.parser.get_command(tokenizer)
                command = next(command_gen)

        exp = command.args[0]
        mgr.quants_for_node[exp] = set(mgr.quants)
        mgr.quants = set()

        return command

    def parse_exp(self, expression_str, with_annotations=True) -> FNode:

        mgr = self.parser.env.formula_manager
        mgr.quants = set()

        with AnnotationsPreserver(self.preamble_script.annotations, with_annotations):
            with io.StringIO(expression_str) as stream:
                tokenizer = Tokenizer(stream, interactive=False)
                exp = self.parser.get_expression(tokenizer)

        mgr.quants_for_node[exp] = set(mgr.quants)
        mgr.quants = set()

        return exp

    def create_quant_with_patterns(self, quant: FNode, nested_qvars: Tuple[FNode], pattern_str: str) -> FNode:

        qvars = quant.quantifier_vars()
        qbody = quant.arg(0)

        with io.StringIO(pattern_str) as stream:
            tokenizer = Tokenizer(stream, interactive=False)
            try:
                [self.parser.cache.bind(var.symbol_name(), var) for var in nested_qvars]
                pats = self.parser.parse_inferred_patterns(qbody, "pattern", tokenizer)
                [self.parser.cache.unbind(var.symbol_name()) for var in nested_qvars]
            except PysmtTypeError as e:
                self.print_dbg("PySMT reported type error (%s) while parsing the pattern `%s`. "
                               "We conclude that this is due to an invalid pattern." % (str(e), pattern_str))
                if self.raise_on_axioms_with_incomplete_patterns:
                    raise InferredPatternInvalid(smt2(quant), pattern_str)
                pats = tuple()
            except NotImplementedError as e:
                self.print_dbg("PySMT reported non-implemented error (%s) while parsing the pattern `%s`. "
                               "We conclude that this is due to an invalid pattern." % (str(e), pattern_str))
                if self.raise_on_axioms_with_incomplete_patterns:
                    raise InferredPatternInvalid(smt2(quant), pattern_str)
                pats = tuple()
            except AttributeError as e:
                self.print_dbg("PySMT reported attribute error (%s) while parsing the pattern `%s`. "
                               "We conclude that this is due to a type incorrect pattern." % (str(e), pattern_str))
                if self.raise_on_axioms_with_incomplete_patterns:
                    raise InferredPatternInvalid(smt2(quant), pattern_str)
                pats = tuple()

        # Check that the pattern is valid
        valid_pats = []

        for mpat in pats:
            is_valid_mpat = True
            for pat in mpat:
                interpreted_symbols = SortedSet(key=lambda x: True)
                collect(pat, lambda n: not n.is_function_application() and len(n.args()) > 0,
                        interpreted_symbols)
                if len(interpreted_symbols) > 0:
                    is_valid_mpat = False
                    self.print_dbg("detected interpreted symbols in pattern: %s" % pattern_str)
                    if self.raise_on_axioms_with_incomplete_patterns:
                        raise InferredPatternInvalid(smt2(quant), pattern_str)
                    break
            if is_valid_mpat:
                valid_pats.append(mpat)

        pats = tuple(valid_pats)

        self.print_dbg("Parsed patterns: (%s)" % str(pats))

        new_quant = self.parser.env.formula_manager.ForAll(qvars, qbody, patterns=pats)

        ann_dict = self.parser.cache.annotations.annotations(quant)
        if ann_dict is not None:
            for ann_name in ann_dict:
                self.parser.cache.annotations.add(new_quant, ann_name, ann_dict[ann_name])

        return new_quant

    @staticmethod
    def _triplet_to_signature(trip: Tuple[str, Tuple[str], str]) -> str:
        return "%s (%s) %s" % (trip[0], " ".join(trip[1]), trip[2])

    @staticmethod
    def _triplet_to_decl(trip: Tuple[str, Tuple[str], str]) -> str:
        return "(declare-fun %s)" % IndividualAxiomTester._triplet_to_signature(trip)

    @staticmethod
    def _node_to_decl_triplet(decl: FNode) -> Tuple[str, Tuple[str], str]:
        name = decl.symbol_name()
        signature = decl.get_type()
        if signature.is_function_type():
            args = signature.param_types
            type = signature.return_type
            return smt2_tup_tup((name, tuple(args), type))
        else:
            return smt2_tup_tup((name, tuple(), signature))

    def _get_all_decls(self):
        symbols = self.parser.env.formula_manager.symbols
        res = [self._node_to_decl_triplet(decl[-1]) for _, decl in symbols.items()]
        return list(map(lambda d: self._triplet_to_decl(d), res))

    def get_all_decl_commands(self) -> List[str]:
        return self._get_all_decls()

    def get_decls(self, name: str, type: PySMTType or None = None) -> List[FNode]:
        symbols = self.parser.env.formula_manager.symbols
        if name not in symbols:
            return []
        symbols_with_name = symbols[name]
        if not type:
            return symbols_with_name
        else:
            symbols_with_name_type = [n for n in symbols_with_name if n.get_type() == type]
            return symbols_with_name_type

    def is_declared(self, name: str, typ: PySMTType or None = None) -> bool:
        decls = self.get_decls(name, typ)
        if not decls:
            return False
        else:
            return len(decls) > 0

    def get_decl(self, name: str, typ: PySMTType or None = None) -> FNode or None:
        decls = self.get_decls(name, typ)
        if not decls or len(decls) == 0:
            return None
        else:
            return decls[-1]

    # mapping from type names to sets of constants
    def get_constant_symbols(self) -> Dict[str, Set[FNode]]:
        all_symbols = [symb for symbs_for_name in self.parser.cache.keys.values() for symb in symbs_for_name]
        # Filter out types/sorts and interpreted constants (true, false, 0, 1, etc.)
        const_symbols = [symb for symb in all_symbols if isinstance(symb, FNode) and not symb.is_constant()]
        const_dict: Dict[str, Set[FNode]] = OrderedDict()
        for symb in const_symbols:
            typ = symb.get_type().name
            if typ not in const_dict.keys():
                const_dict[typ] = SortedSet({symb}, key=lambda x: True)
            else:
                const_dict[typ].add(symb)
        return const_dict

    def skolemize(self, t: Tactic, formula_node: FNode, formula_str=None, approach="nnf", ensure_qfree=False) \
        -> Tuple[List[Tuple[str, str]],
                 List[Tuple[str, Tuple[str], str]],
                 Dict[str, str],
                 List[str]]:

        # Retrieve the set of originally bound variables
        quants: Set[FNode] = self.get_quantifiers(formula_node)

        # Explanation for only_in_body=True:
        # Skolemizing e.g. (forall x, y . {Q(x,y)} P(x)) requires dropping qvar y as it's not mentioned in the body
        # Otherwise, inst_map would get messed up
        qvars = collect_quantified_variables(quants, only_in_body=True)

        self.print_dbg("target vars for Skolemization: %s" % str(qvars))

        # Get data from Skolemizer
        if not formula_str:
            formula_str = smt2(formula_node)
        sk_vars, sk_formulas = t.skolemize(formula_str, approach=approach)

        skolem_decls = SortedSet([(var, typ) for var, args, typ in sk_vars if len(args) == 0], key=lambda x: True)

        self.print_dbg("Skolem form of the original axiom is: %s" % sk_formulas)

        # We rely on Z3's naming convention for Skolem variables:
        #  'x' -> 'x!n', for some natural n
        # However, z3 may introduce new free variables via defunctionalization, e.g. 'z3name!4'

        reduced_skolem_decls = SortedSet(key=lambda x: True)
        for decl in skolem_decls:
            name = decl[0]
            sp = name.split("!")
            if len(sp) < 2:
                continue
            prefix = "!".join(sp[:-1])
            postfix = sp[-1]
            for qvar in qvars:
                if prefix != qvar.symbol_name():
                    continue
                try:
                    int(postfix)
                except ValueError:
                    continue
                else:
                    reduced_skolem_decls.add(decl)

        std_skolem_vars = SortedSet([var_name for var_name, _ in reduced_skolem_decls], key=lambda x: True)

        defun_decls = SortedSet([decl for decl in skolem_decls if decl not in reduced_skolem_decls and
                                not self.is_declared(decl[0])], key=lambda x: True)
        std_defun_decls = SortedSet([(name, typ) for name, typ in defun_decls], key=lambda x: True)

        # Represent neatly the mapping from old to new,
        # and new declarations (with types).
        instantiation_map = OrderedDict()
        for qvar, skolem_var_name in zip(sorted(qvars, key=lambda v: v.symbol_name()), sorted(std_skolem_vars)):
            original_name = qvar.symbol_name()
            instantiation_map[original_name] = skolem_var_name

        # Extract the missing declarations and add them to PySMT
        required_new_decls = sorted((decl for decl in reduced_skolem_decls.union(std_defun_decls)
                                     if not self.is_declared(decl[0])))
        for decl_name, decl_type, in required_new_decls:
            decl = self._triplet_to_decl((decl_name, tuple(), decl_type))
            self.parse_cmd(decl)

        # Treat Skolem functions with arity >0 (these should appear only if the quantifiers were not eliminated).
        skolem_fun_decls = SortedSet([(var, args, typ) for var, args, typ in sk_vars if len(args) != 0],
                                     key=lambda x: True)
        required_new_fun_decls = sorted((decl for decl in skolem_fun_decls if not self.is_declared(decl[0])),
                                        key=lambda x: True)

        for triplet in required_new_fun_decls:
            decl = self._triplet_to_decl(triplet)
            self.parse_cmd(decl)

        assert not ensure_qfree or (len(required_new_fun_decls) == 0), \
            "Skolem functions should be defunctionalized in q-free formulas."

        return required_new_decls, required_new_fun_decls, instantiation_map, sk_formulas

    def std_negate(self, sk: Tactic, axiom: Axiom) -> Tuple[List[Tuple[str, str]], List[str]]:
        with PhaseSwitcher(self, "std_negate(%s)" % axiom.axiom_id):
            return self.__std_negate_impl(sk, axiom)

    def __std_negate_impl(self, sk: Tactic, axiom: Axiom) -> Tuple[List[Tuple[str, str]], List[str]]:

        axiom_name = axiom.axiom_id
        node = Not(axiom.axiom_node)

        # propagate the quantifiers to the negated node
        self.parser.env.formula_manager.quants_for_node[node] = axiom.quantifiers

        sk_decls, _, inst_map, sk_formulas = self.skolemize(sk, node, approach="nnf", ensure_qfree=True)
        sk_axioms: List[str] = list()
        for axiom_number, sk_axiom_str in enumerate(sk_formulas):
            _, sk_axiom_str = self.rename_axiom(axiom_name, axiom_number, sk_axiom_str, sk_formulas)
            sk_axioms.append(sk_axiom_str)

        new_sk_axioms = []

        for sk_axiom in sk_axioms:
            new_sk_axiom = sk_axiom
            for var, val in inst_map.items():
                new_sk_axiom = create_substituted_string(self.ALPH, val, var, new_sk_axiom)
            new_sk_axioms.append(new_sk_axiom)

        missing_sk_decls = [(decl_name, decl_type) for decl_name, decl_type in sk_decls
                            if decl_name not in inst_map.values()]

        return missing_sk_decls, new_sk_axioms

    def construct_proto_triggering_terms(self, triggers: Set[FNode], all_qvars: Set[FNode], constants: Set[FNode],
                                         rewritings: Set[Rewriting] = set()) -> ProtoTerms:
        terms: Set[FNode] = SortedSet(key=lambda x: str(x))

        for trigger in triggers:
            self.print_dbg("retrieved triggering term %s" % str(trigger))

            term = trigger

            # Apply the rewritings
            all_rewritings = sorted(rewritings)
            for index, rewriting in enumerate(all_rewritings):
                var = rewriting.qvar
                val = rewriting.equivalent_expr
                term = term.substitute({var: val})

                # Propagate the substitution to the following rewritings
                next_rewritings = all_rewritings[index + 1:]
                next_index = index + 1
                for next_rewriting in next_rewritings:
                    next_var = next_rewriting.qvar
                    next_val = next_rewriting.equivalent_expr
                    all_rewritings[next_index] = Rewriting(next_var, next_val.substitute({var: val}),
                                                           next_rewriting.just_qvars)
                    next_index = next_index + 1

            terms.add(term)

        # Remove redundant terms
        if len(terms) > 1:
            terms = remove_redundant_terms(terms)
        qvars: Set[FNode] = SortedSet([var for term in terms for var in term.get_free_variables() if var in all_qvars],
                                      key=lambda x: x.symbol_name())
        return ProtoTerms(terms, qvars, constants)

    def construct_dummy(self, terms: Set[FNode], fresh_vars: Set[FreshVariable]) -> Dummy or None:
        filtered_terms: Set[FNode] = SortedSet(key=lambda x: str(x))

        # remove the triggering terms that contain skolem functions
        for term in terms:
            if not self.contains_skolem_functions(term):
                filtered_terms.add(term)

        if len(filtered_terms) == 0:
            return None

        dummy_fun_name = "__dummy_%s__" % self.suffix
        return Dummy(dummy_fun_name, self.suffix, filtered_terms, fresh_vars)

    def get_skolem_functions(self, node: FNode):
        skolem_functions: Set[FNode] = SortedSet(key=lambda x: True)
        collect(node, lambda n: n.is_function_application() and
                                n.get_function_name() not in self.original_declarations, skolem_functions)
        return skolem_functions

    def contains_skolem_functions(self, term: FNode):
        return len(self.get_skolem_functions(term)) != 0

    def parse_assignment_list(self, raw_assignments: str) -> Dict[FNode, FNode]:
        with io.StringIO(raw_assignments) as stream:
            pairs = self.parser.get_assignment_list(stream)
        return OrderedDict([(var, val) for var, val in pairs])

    def get_model(self, session: Session, model_vars: Set[str]) -> Dict[FNode, FNode] or None:

        raw_model = session.get_values(model_vars)
        if not raw_model:
            return None

        if raw_model.startswith('Completed Query') and 'ping' in raw_model or raw_model.startswith('ping'):
            raw_model = raw_model[raw_model.index('('):]

        model = self.parse_assignment_list(raw_model)
        assert len(model) == len(model_vars), "all variables must have some assignment in the model"

        # Declare constants of uninterpreted types (that remain as strings so far)
        std_model = OrderedDict()

        already_present = OrderedDict()

        for var, val in model.items():
            if isinstance(val, str):
                typ = var.get_type()
                if not typ.custom_type:
                    # this is a strange model, in which the solver assigned fresh values to vars of built-in types
                    return None
                # make the declarations unique
                val_name = val + "_" + self.suffix

                if val_name not in already_present:
                    decl_str = "(declare-fun %s () %s)" % (val_name, typ.name)
                    decl_cmd = self.parse_cmd(decl_str)
                    decl = decl_cmd.args[0]
                    already_present[val_name] = decl

                std_model[var] = already_present[val_name]
            else:
                std_model[var] = val

        return std_model

    @staticmethod
    def apply(proto_terms: ProtoTerms, model: Dict[FNode, FNode]) -> Set[FNode]:
        new_terms: Set[FNode] = SortedSet(key=lambda x: str(x))

        # Substitute the values from the model
        for term in proto_terms.terms:
            new_term = term.substitute(model)
            new_terms.add(new_term)

        return new_terms

    @staticmethod
    def optimize_model(model: Dict[FNode, FNode], qvars: Set[FNode]) -> Dict[FNode, FNode]:

        model_for_qvars: Dict[FNode, FNode] = OrderedDict()

        # constants or predefined fresh variables
        const_value_map: Dict[FNode, Set[FNode]] = OrderedDict()

        for var, val in model.items():
            if var in qvars:
                model_for_qvars[var] = val
            elif val not in const_value_map.keys():
                const_value_map[val]: Set[FNode] = SortedSet(key=lambda x: x.symbol_name())
                const_value_map[val].add(var)

        all_models: Dict[FNode, Set[FNode]] = OrderedDict()
        for var, val in model_for_qvars.items():
            if val in const_value_map:
                all_models[var] = const_value_map[val]  # assign constants
            else:
                all_models[var] = SortedSet({val}, key=lambda x: True)  # assign value from model

        qvars_list = list(qvars)
        for a_model in itertools.product(*all_models.values()):
            optimized_model = OrderedDict()
            assert(len(a_model) == len(qvars_list))
            for index, val in enumerate(a_model):
                optimized_model[qvars_list[index]] = val

            # NOTE: the solver returns a family of models, as it can assign the same fresh value to multiple constants
            # we return here the first one and ask the solver for a different model (otherwise we have to integrate the
            # logic for different models here)
            return optimized_model

    @staticmethod
    def declare_funs(sessions: Iterator[Session], funs: List[Tuple[str, Tuple[str], str]], comment: str) -> List[str]:
        decls = []
        for s in sessions:
            decls = s.declare_funs(funs, comment=comment)
        return decls

    @staticmethod
    def declare_consts(sessions: Iterator[Session], consts: List[Tuple[str, str]], comment: str) -> List[str]:
        decls = []
        for s in sessions:
            decls = s.declare_consts(consts, comment=comment)
        return decls

    def get_alias_classes(self, model: Dict[FNode, FNode]) -> Dict[FNode, Tuple[FNode, ...]]:
        alias_classes = OrderedDict()

        for val in model.values():
            aliases = tuple([var for var in model.keys() if model[var] == val and
                             val not in self.constant_nodes])
            if len(aliases) > 0:
                alias_classes[val] = aliases
        return alias_classes

    def ensure_fresh_model(self, session: Session, this_model: Dict[FNode, FNode]):

        assert len(this_model) > 0, "model cannot be empty"

        # Compute alias classes
        alias_classes: Dict[FNode, Tuple[FNode, ...]] = self.get_alias_classes(this_model)

        aliases = OrderedDict()
        for val, vars in alias_classes.items():
            if len(vars) > 1:
                aliases[val] = vars

        if len(aliases) > 0:
            if not self.opt.diverse_models:
                aliasing_cond = "(and %s)" % " ".join(["(= %s)" % " ".join([smt2(var) for var in
                                aliases[val]]) for val in aliases])
            else:
                aliasing_cond = "(or %s)" % " ".join(["(= %s)" % " ".join([smt2(var) for var in
                                aliases[val]]) for val in aliases])
        else:
            aliasing_cond = "true"  # true is the unit of AND

        if len(this_model) > 0:
            if not self.opt.diverse_models:
                values_cond = "(and %s)" % " ".join(["(= %s %s)" % (smt2(var), smt2(val))
                                                                        for val, var in this_model.items()])
            else:
                values_cond = "(or %s)" % " ".join(["(= %s %s)" % (smt2(var), smt2(val))
                                                                 for val, var in this_model.items()])
        else:
            values_cond = "true"  # true is the unit of AND

        if not (aliasing_cond == "true" and values_cond == "true"):
            not_this_model = "(not (and %s %s))" % (aliasing_cond, values_cond)
            session.add_soft_constraint(not_this_model, comment="ensure new values")

    def skolemize_one(self, a_num: int, a: str, *args, **kwargs) -> List[Axiom]:
        with PhaseSwitcher(self, "skolemize_all: %d/%d" % (a_num+1, self.num_axioms)):
            return self.__skolemize_one_impl(a, *args, **kwargs)

    def get_quantifiers(self, assert_node):
        return self.parser.env.formula_manager.quants_for_node[assert_node]

    def __skolemize_one_impl(self, assertion: str, session: Session, tactic: Tactic, pp: PatternProvider) -> \
            List[Axiom]:

        axiom_name = Minimizer.get_assertion_name(assertion)

        # Get the axiom node
        try:
            with VariableNamesPreserver():
                axiom_node = self.parse_cmd(assertion).args[0]
        except PysmtAnnotationsError:
            self.print_dbg("detected equivalent axioms with different names; keep only the first one")
            return []

        # Get axioms in Skolem form and new required declarations
        if self.opt.disable_preprocessor:
            quantifiers: Set[FNode] = self.get_quantifiers(axiom_node)

            self.distributivity_level = 0
            skolem_functions: Set[FNode] = self.get_skolem_functions(axiom_node)
            if contains_and(axiom_node) and self.distributivity_level < self.opt.max_distributivity:
                qvars = collect_quantified_variables(quantifiers)
                # make all the quantified variables available to the parser
                [self.parser.cache.bind(var.symbol_name(), var) for var in qvars]
                split_axioms = self.split_axiom(axiom_node, axiom_name, skolem_functions)
                [self.parser.cache.unbind(var.symbol_name()) for var in qvars]
                return split_axioms
            else:
                axiom = self.create_axiom(axiom_name, axiom_node, quantifiers, skolem_functions)
                return [axiom]

        axiom_str = self.m.get_assertion_body(assertion)
        new_const_decls, new_fun_decls, _, skolemized_axioms_str = self.skolemize(tactic, axiom_node, axiom_str)
        # Add new declarations to the session
        fun_declarations = self.declare_funs(sessions=(pp, session),
                                             funs=new_fun_decls, comment="Skolem functions")
        constant_declarations = self.declare_consts(sessions=(pp, session),
                                                    consts=new_const_decls, comment="Skolem constants")
        # Update the preamble
        for declaration in fun_declarations + constant_declarations:
            self.preamble.append(declaration)

        skolem_functions: Set[FNode] = SortedSet(key=lambda x: True)
        for fun_declaration in fun_declarations:
            skolem_function = self.parse_cmd(fun_declaration).args[0]
            skolem_functions.add(skolem_function)

        # Create Axioms in Skolem form (without splitting quantifiers and applying distributivity)
        skolemized_axioms = []
        for axiom_number, axiom_str in enumerate(skolemized_axioms_str):
            axiom_id, axiom_str = self.rename_axiom(axiom_name, axiom_number, axiom_str, skolemized_axioms_str)
            self.parser.env.keep_qvar_names = False
            axiom_node = self.parse_exp(axiom_str)
            if not self.opt.keep_duplicate_assertions and axiom_node in self.parser.env.formula_manager.formulae:
                self.print_dbg("skipping duplicate axiom %s" % smt2(axiom_node))
                continue

            axiom_str = self.m.get_assertion_body(self.m.get_assertion_without_name("(assert %s)" % axiom_str))
            axiom_node, quantifiers, patterns = self.infer_patterns(pp, axiom_id, axiom_node, axiom_str)
            if self.raise_on_axioms_with_incomplete_patterns:
                if len(patterns) < len(quantifiers):
                    raise PatternException("some quantifiers do not have patterns")

            axiom = Axiom(axiom_id, axiom_node, quantifiers, patterns, skolem_functions, axiom_name)
            skolemized_axioms.append(axiom)
            self.create_fresh_variables(axiom)

        return skolemized_axioms

    def split_quantifier(self, quant: FNode) -> Tuple[List[FNode], bool]:
        conjuncts, found_and = self.distributivity(inner_disjuncts(quant))

        new_quantifiers: List[FNode] = []
        qvars_str = ""
        for qvar in quant.quantifier_vars():
            qvars_str += "(%s %s)" % (smt2(qvar), smt2(qvar.get_type()))
        triggers = collect_triggers(quant.quantifier_patterns())
        triggers_str = ""
        for trigger in triggers:
            triggers_str += " " + smt2(trigger)
        for conjunct in conjuncts:
            quant_str = "(forall (%s)" % qvars_str + " (! " + smt2(conjunct) + " :pattern ( %s ) ))" % triggers_str
            self.parser.env.keep_qvar_names = False
            new_quant = self.parse_exp(quant_str)
            assert len(quant.quantifier_patterns()) > 0, 'the triggers were not propagated'
            new_quantifiers.append(new_quant)
        return new_quantifiers, found_and

    def full_distributivity(self, disjuncts: List[FNode]) -> List[FNode]:
        conjuncts, found_and = self.distributivity(disjuncts)
        self.distributivity_level += 1
        while found_and and self.distributivity_level < self.opt.max_distributivity:
            new_conjuncts: List[FNode] = []
            found_and = False
            for conjunct in conjuncts:
                if not contains_and(conjunct):
                    new_conjuncts.append(conjunct)
                elif conjunct.is_quantifier():
                    split_quants, found = self.split_quantifier(conjunct)
                    if found:
                        found_and = True
                    new_conjuncts += split_quants
                else:
                    inner_conjuncts, found = self.distributivity(collect_disjuncts(conjunct))
                    if found:
                        found_and = True
                    new_conjuncts += inner_conjuncts
            conjuncts = new_conjuncts
            self.distributivity_level += 1
        conjuncts_node: FNode = And(*conjuncts).simplify()
        return conjuncts_node.args()

    def distributivity(self, disjuncts: List[FNode]) -> Tuple[List[FNode], bool]:
        conjuncts: List[FNode] = []
        found_and: bool = False
        for disjunct in disjuncts:
            if disjunct.is_quantifier() and contains_and(disjunct):
                split_quants, found_and = self.split_quantifier(disjunct)
                if not found_and:
                    found_and = True
                conjuncts += split_quants
            elif not disjunct.is_and():
                # add the disjunct to all the conjuncts
                if len(conjuncts) == 0:
                    conjuncts.append(disjunct)
                else:
                    new_conjuncts: List[FNode] = []
                    for conjunct in conjuncts:
                        simple_node: FNode = Or(conjunct, disjunct).simplify()
                        if not simple_node.is_true():
                            new_conjuncts.append(simple_node)
                    conjuncts = new_conjuncts
            else:
                new_conjuncts: List[FNode] = []
                for arg in disjunct.args():
                    if not found_and and not arg.is_quantifier() and contains_and(arg):
                        found_and = True
                    # add the arg to all the conjuncts
                    if len(conjuncts) == 0:
                        new_conjuncts.append(arg)
                    else:
                        for conjunct in conjuncts:
                            simple_node: FNode = Or(conjunct, arg).simplify()
                            if not simple_node.is_true():
                                new_conjuncts.append(simple_node)
                conjuncts = new_conjuncts
        if len(conjuncts) == 0:
            return [], False
        return conjuncts, found_and

    def split_axiom(self, axiom_node: FNode, axiom_id: str, skolem_functions: Set[FNode]) -> List[Axiom]:
        split_axioms: List[Axiom] = []
        conjuncts: List[FNode] = self.full_distributivity(collect_disjuncts(axiom_node))
        for index, conjunct in enumerate(conjuncts):
            new_quants = collect_quantifiers(conjunct)
            new_axiom = self.create_axiom(axiom_id, conjunct, new_quants, skolem_functions, index)
            split_axioms.append(new_axiom)
        return split_axioms

    def create_axiom(self, axiom_id: str, new_axiom_node: FNode, new_quants: Set[FNode],
                     skolem_functions: Set[FNode], axiom_number: int=-1) -> Axiom:
        if axiom_number == -1:
            new_axiom_id = axiom_id
        else:
            new_axiom_id = axiom_id + "_" + str(axiom_number)
        new_triggers: Set[FNode] = SortedSet(key=lambda x: True)
        for quant in new_quants:
            triggers: Set[FNode] = collect_triggers(quant.quantifier_patterns())
            assert len(triggers) > 0, 'each quantifier should have triggers'
            new_triggers.update(triggers)

        new_axiom = Axiom(new_axiom_id, new_axiom_node, new_quants, new_triggers, skolem_functions, axiom_id)
        self.create_fresh_variables(new_axiom)
        return new_axiom

    def create_fresh_variables(self, axiom: Axiom):
        qvars = axiom.quantified_variables
        for qvar in qvars:
            if not qvar.get_type().custom_type:
                continue
            typ = qvar.get_type().name
            if typ not in self.fresh_vars_pool.keys():
                self.fresh_vars_pool[typ] = SortedSet(key=lambda x: x.var_name)
                vars_number = 0
            else:
                vars_number = len(self.fresh_vars_pool[typ])
            for index in range(self.opt.max_axiom_frequency):
                var_name = "fresh_%s_%s" % (typ, str(vars_number + index))
                declaration = "(declare-fun %s () %s)" % (var_name, typ)
                var_node = self.parse_cmd(declaration).args[0]
                fresh_var = FreshVariable(var_name, typ, var_node, declaration)
                self.fresh_vars_pool[typ].add(fresh_var)

    def skolemize_all(self, session: Session) -> List[Axiom]:

        with PhaseSwitcher(self, "skolemize_all"):

            with Tactic(name="Skolemize All Session", preamble=EMATCHING + self.preamble,
                        logs_path=self.skolems_logs_path, memory_limit_mb=6000) as tactic:

                with PatternProvider(name="Pattern Provider", preamble=EMATCHING + self.preamble,
                                     logs_path=self.patterns_logs_path) as pp:

                    axioms = self.__skolemize_all_impl(session, tactic, pp)

                # inform PySMT about the Skolem declarations
                self._refresh_parser()
                return axioms

    # add the Skolem declarations to the session
    def __skolemize_all_impl(self, session: Session, tactic: Tactic, pp: PatternProvider) -> List[Axiom]:

        assertions = self.m.get_assertions()
        skolemized_axioms = []
        for a_num, a in enumerate(assertions):
            sk_asserts = self.skolemize_one(a_num, a, session, tactic, pp)
            # One axiom might be Skolemized into several
            [skolemized_axioms.append(sk_assert) for sk_assert in sk_asserts]

        return skolemized_axioms

    def rename_axiom(self, axiom_name, axiom_number, axiom_str, skolemized_axioms_str):
        axiom_id = axiom_name
        if len(skolemized_axioms_str) > 1:
            # the solver split the original axiom in multiple axioms and assigned names to them
            axiom_id = axiom_id + "_" + str(axiom_number)
        # rename the axiom
        name_from_solver = Minimizer.get_assertion_name(axiom_str)
        axiom_str = create_substituted_string(self.ALPH, name_from_solver, axiom_id, axiom_str)
        return axiom_id, axiom_str

    def infer_patterns(self, pp: PatternProvider, axiom_id: str,
                       axiom_node: FNode, axiom_str: str) -> Tuple[FNode, Set[FNode], Set[FNode]]:
        with PhaseSwitcher(self, "pattern_inference(%s)" % axiom_id):
            return self.__infer_patterns_impl(pp, axiom_node, axiom_str)

    def __infer_patterns_impl(self, pp: PatternProvider, node: FNode, axiom_str: str) -> Tuple[FNode, Set[FNode], Set[FNode]]:

        self.print_dbg("Inferring patterns for %s" % str(node))

        def get_qid(n: FNode) -> str:
            return next(iter(self.parser.cache.annotations[n]["qid"]))

        # Ensure all quantifiers have known qids
        quant_bodies_with_qid = SortedSet(self.parser.cache.annotations.all_annotated_formulae("qid"),
                                          key=lambda x: True)
        qid_pool = SortedSet([get_qid(qbody) for qbody in quant_bodies_with_qid], key=lambda x: True)

        def pick_fresh_qid(prefix: str, count: int) -> str:
            nonlocal qid_pool
            new_qid = "%s_%d" % (prefix, count)
            if new_qid in qid_pool:
                new_qid = pick_fresh_qid(prefix, count=count + 1)
            qid_pool.add(new_qid)
            return new_qid

        quants = self.get_quantifiers(node)
        triggers = set()

        if len(quants) == 0:
            self.print_dbg("Skipping QF formula `%s`" % str(node))
            return node, quants, triggers

        quants_without_patterns = OrderedDict()
        for quant in quants:
            # Set missing qids
            qbody = quant.arg(0)
            if qbody not in quant_bodies_with_qid:
                qid = pick_fresh_qid("quant-with-auto-pattern", len(qid_pool))
                self.parser.cache.annotations.add(qbody, "qid", qid)
            else:
                qid = get_qid(qbody)
            alternative_triggers = quant.quantifier_patterns()
            if len(alternative_triggers) == 0:
                quants_without_patterns[qid] = quant
            else:
                triggers.update(collect_triggers(alternative_triggers))

        # Serialize
        # node_str = axiom_str -- TODO
        node_str = smt2(node)

        if len(quants_without_patterns) == 0:
            # All the quantifiers already have patterns
            self.print_dbg("All quantifiers already have patterns in the formula `%s`"
                           % node_str)
            return node, quants, triggers

        # Get the pattern annotations by calling Z3
        raw_patterns = pp.get_pattern(node_str)

        if len(raw_patterns) == 0:
            # No patterns were provided by Z3
            self.print_dbg("No patterns were provided by Z3 for the formula `%s`"
                           % node_str)
            return node, quants, triggers

        self.print_dbg("Obtained patterns `%s`" % ", ".join(raw_patterns.values()))
        self.print_dbg("  for node `%s`" % node_str)

        # Maps qid to a tuple of quantified variables of this quantifier only
        shallow_qvars = OrderedDict()
        # Maps qid to a tuple of quantified variables from all levels of nestedness, including this quantifier's level
        nested_qvars = OrderedDict()  # (respecting De Bruijn's order)

        def collect_nested_qvars(n: FNode, parents_qs: Tuple[FNode] = tuple()):
            nonlocal nested_qvars
            if n.is_quantifier():
                qid = get_qid(n.arg(0))
                shallow_qvars[qid] = tuple(n.quantifier_vars())
                my_qs = nested_qvars[qid] = tuple(reversed(n.quantifier_vars())) + parents_qs
            else:
                my_qs = parents_qs
            for c in n.args():
                collect_nested_qvars(c, my_qs)

        collect_nested_qvars(node)

        # Map the quantified variable names (:var 1), etc. from the patterns
        # to the actual names of quantified vars.
        patterns = OrderedDict()
        for qid, raw_pattern in raw_patterns.items():
            qvars_of_this_quant = shallow_qvars[qid]
            qvars = nested_qvars[qid]

            # Check that the list of De Bruijn indices maps to the list of qvars
            de_bruijn = SortedSet(map(int, re.findall("\(:var\s+(\d+)\)", raw_pattern)))

            if not set(range(len(qvars_of_this_quant))).issubset(de_bruijn):
                # This means that there are not enough De Bruijn indices to cover all quantified variables of
                # this quantifier. Hence, this is not a valid pattern.
                self.print_dbg("Not enough De Bruijn indices to cover all quantified variables of the current "
                               "quantifier with qid %s" % qid)
                if self.raise_on_axioms_with_incomplete_patterns:
                    raise InferredPatternInvalid(node_str, raw_pattern)
                else:
                    return node, quants, set()

            if not de_bruijn.issubset(set(range(len(qvars)))):
                # Some pattern holes have not been matched with qvars. This is due to Z3 producing weird things
                #  when a trigger cannot be inferred without rearranging the quantifiers.
                # Repro:
                # (declare-sort A 0)(declare-sort B 0)(declare-fun P (B A B) Bool)(declare-fun Q (A B) Bool)(declare-fun f (Int) Bool)
                # (assert (forall ((y0 A) (y1 B)) (! (and (forall ((z B)) (! (P z y0 y1) :qid SubQuant-II)) (forall ((x1 Int)) (! (f x1) :qid SubQuant-I))) :qid MainQuant)))
                # ;(smt.inferred-patterns :qid MainQuant ((P (:var 0) (:var 3) (:var 2)) (f (:var 1))))
                self.print_dbg("Some De Bruijn indices cannot be mapped to quantified variables. "
                               "This is due to Z3. No patterns inferred for quantifier with qid %s" % qid)
                if self.raise_on_axioms_with_incomplete_patterns:
                    raise InferredPatternInvalid(node_str, raw_pattern)
                else:
                    return node, quants, set()

            pattern_str = raw_pattern
            for arg_num, arg in enumerate(qvars):
                arg_name = smt2(arg)
                pattern_str = pattern_str.replace("(:var %s)" % arg_num, arg_name)

            # Check that all De Bruijn indices were substituted.
            m = re.match("\(:var\s+(\d+)\)", pattern_str)
            assert not m, "could not parse pattern inferred by Z3: %s" % raw_pattern

            patterns[qid] = pattern_str

        self.print_dbg("Instantiated auto patterns: %s" % ", ".join(patterns.values()))

        # Parse the patterns and add them as payload.
        # Substitute the old quants with the new ones along the way.
        new_node = node

        def walk_quant(n: FNode, new_quants: Set[FNode], new_triggers: Set[FNode]):
            nonlocal new_node
            if n.is_quantifier():
                quant = n
                qid = get_qid(n.arg(0))
                if qid in quants_without_patterns:
                    if qid in patterns:
                        pattern = patterns[qid]
                        new_quant = self.create_quant_with_patterns(quant, nested_qvars[qid], pattern)
                        self.print_dbg(smt2(new_quant))
                        self.print_dbg("quant: %s" % smt2(quant))
                        self.print_dbg("new_quant: %s" % smt2(new_quant))
                        self.print_dbg("new_node before sub: %s" % smt2(new_node))
                        new_node = new_node.substitute({quant: new_quant})
                        self.print_dbg("new_node after sub: %s" % smt2(new_node))
                        new_quants.add(new_quant)
                        new_triggers.update(collect_triggers(new_quant.quantifier_patterns()))
                    else:
                        self.print_dbg("Z3 did not provide any patterns for quantifier with qid %s" % qid)
                        new_quants.add(quant)
                else:
                    new_quants.add(quant)
                    new_triggers.update(collect_triggers(quant.quantifier_patterns()))

            for c in n.args():
                walk_quant(c, new_quants, new_triggers)

        new_quants: Set[FNode] = SortedSet(key=lambda x: True)
        new_triggers: Set[FNode] = SortedSet(key=lambda x: True)
        walk_quant(new_node, new_quants, new_triggers)

        return new_node, new_quants, new_triggers

    def run_experiment(self, batch_size: int, dummy_constraints: List[str],
                       fresh_vars: Set[FreshVariable]) -> Tuple[str, str, str]:
        # make a copy of the input file and add the dummy_constraints
        std_file_name = add_fname_suffix(self.standardized_file_path, 'std')
        if not self.opt.keep_duplicate_assertions:
            std_file_name = add_fname_suffix(std_file_name, 'unique')
        test_file_name = add_fname_suffix(std_file_name, 'dummy')
        outcome = run_test(std_file_name, test_file_name, fresh_vars, dummy_constraints,
                           batch_size, self.timeout_per_query)
        return outcome, std_file_name, test_file_name

    def __print_outcome(self, outcome: str, attachment: any) -> None:
        if outcome == "timeout":
            aux_text = " after %.3f seconds" % attachment
        elif outcome == "unknown":
            aux_text = " with reason %s" % attachment
        else:
            aux_text = ""
        self.print_dbg("Test outcome: %s%s" % (outcome, aux_text))

    def handle_special_case(self, proto_terms: ProtoTerms, axiom_id: str, test_number: int) -> Tuple[bool, str]:
        self._register_namespace(axiom_id, tnum=test_number)

        dummy = self.construct_dummy(proto_terms.terms, set())
        if dummy is None:
            self.print_dbg("No dummy could be constructed (it only contained Skolem functions).")
            return False, ""

        if dummy not in self.all_dummies:
            # print(dummy)
            self.dummies.add(dummy)
            self.all_dummies.add(dummy)

        if len(self.dummies) == self.opt.batch_size:
            # test the dummies synthesized so far in batch mode
            inconsistent, term = self.test_dummy_batch_mode()
            self.dummies.clear()
            if inconsistent:
                return inconsistent, term
        return False, ""

    # collect the names of the quantified variables for which the solver should find values
    # (also include the original constants of the right types from the bodies/triggers of the axioms from which the
    # proto terms were generated)
    # also return the fresh variables used to constrain the quantified variables of user-defined types
    def get_vars(self, proto_terms: ProtoTerms, session: Session) -> Tuple[Set[str], Set[FreshVariable]]:
        var_names: Set[str] = SortedSet(key=lambda x: x)
        types: Set[str] = SortedSet(key=lambda x: x)

        qvars_user_defined_types: Dict[str, Set[FNode]] = OrderedDict()

        for qvar in proto_terms.qvars:
            var_names.add(qvar.symbol_name())
            var_typ = qvar.get_type()
            typ = var_typ.name
            types.add(typ)
            if var_typ.custom_type:
                if typ not in qvars_user_defined_types.keys():
                    qvars_user_defined_types[typ] = SortedSet(key=lambda x: True)
                qvars_user_defined_types[typ].add(qvar)

        constants: Dict[str, Set[FNode]] = OrderedDict()
        for typ in types:
            if typ in self.constants.keys():
                constants[typ] = self.constants[typ].intersection(proto_terms.constants)

        filtered_constants: Set[str] = SortedSet((const.symbol_name() for consts in constants.values()
                                                  for const in consts), key=lambda x: x)

        fresh_var_names: Set[str] = SortedSet(key=lambda x: x)
        fresh_vars: Set[FreshVariable] = SortedSet(key=lambda x: x.var_name)
        if qvars_user_defined_types:
            fresh_vars = self.constrain_qvars_user_defined_types(qvars_user_defined_types, constants, session)
            fresh_var_names.update({v.var_name for v in fresh_vars})

        all_var_names: Set[str] = SortedSet(var_names.union(filtered_constants).union(fresh_var_names), key=lambda x: x)
        return all_var_names, fresh_vars

    def constrain_qvars_user_defined_types(self, qvars_user_defined_types: Dict[str, Set[FNode]],
                                           constants: Dict[str, Set[FNode]], session: Session) -> Set[FreshVariable]:
        used_fresh_vars: Set[FreshVariable] = SortedSet(key=lambda x: True)
        for typ, qvars in qvars_user_defined_types.items():
            number_of_qvars = len(qvars)
            values: Set[str] = SortedSet(key=lambda x: x)
            if typ in constants.keys():
                values.update({const.symbol_name() for const in constants[typ]})
            fresh_vars = {v for v in itertools.islice(self.fresh_vars_pool[typ], number_of_qvars)}
            values.update({v.var_name for v in fresh_vars})
            conditions: List[str] = []
            for qvar in qvars:
                condition = ""
                for val in values:
                    condition = condition + " " + "(= %s %s)" % (smt2(qvar), val)
                conditions.append("(or %s)" % condition)
            aliasing = "(and %s)" % " ".join([condition for condition in conditions])
            session.add_constraint(aliasing, comment="ensure the quantified variable has a predefined fresh value")
            used_fresh_vars.update(fresh_vars)
        return used_fresh_vars

    def synthesize_triggering_term(self, axiom_num: int, axiom_id: str, *args, **kwargs) -> \
            Tuple[Tuple[bool, str], SmtResponse]:

        with PhaseSwitcher(self, "stt(%s): %d/%d" % (axiom_id, axiom_num+1, self.num_axioms)):
            return self.__synthesize_triggering_term_impl(axiom_id, *args, **kwargs)

    def __synthesize_triggering_term_impl(self, axiom_id: str, proto_terms: ProtoTerms, session: Session, test_number:
                                          int) -> Tuple[Tuple[bool, str], SmtResponse]:
        result = False, ""

        if proto_terms not in self.previous_models.keys():
            self.previous_models[proto_terms] = []
        else:
            # avoid generating the same models multiple times
            for prev_model in self.previous_models[proto_terms]:
                self.ensure_fresh_model(session, prev_model)

        all_vars, fresh_vars = self.get_vars(proto_terms, session)

        session.push("Initialize the stack...")

        for model_num in range(self.opt.max_different_models):
            self._register_namespace(axiom_id, tnum=test_number, anum=model_num)

            status, _ = session.check_sat()

            if status == "unsat":
                self.print_dbg("The solver returned `unsat` for the synthesized formula.")
                session.pop("Remove the stack [the solver returned `unsat` for the synthesized formula].")
                return result, SmtResponse.UNSAT

            model = self.get_model(session, all_vars)

            if not model:
                self.print_dbg("No more models available for this axiom.")
                session.pop("Remove the stack [no more models available].")
                return result, SmtResponse.SAT

            # Use the constants from the axiomatization instead of the ones declared by the solver
            model_with_constants: Dict[FNode, FNode] = self.optimize_model(model, proto_terms.qvars)

            # Instantiate fresh vars with model values and construct the dummy
            terms = self.apply(proto_terms, model_with_constants)
            dummy = self.construct_dummy(terms, fresh_vars)

            if dummy is None:
                self.print_dbg("No dummy could be constructed (it only contained Skolem functions).")
                session.pop("Remove the stack [no dummy constructed].")
                return result, SmtResponse.SAT

            if dummy not in self.all_dummies:
                # print(dummy)
                self.dummies.add(dummy)
                self.all_dummies.add(dummy)

            if len(self.dummies) == self.opt.batch_size:
                # test the dummies synthesized so far in batch mode
                inconsistent, term = self.test_dummy_batch_mode()
                self.dummies.clear()
                if inconsistent:
                    session.pop("Remove the stack [inconsistency found].")
                    return (inconsistent, term), SmtResponse.SAT

            # Ensure this model will not be used again
            if model_with_constants not in self.previous_models[proto_terms]:
                self.previous_models[proto_terms].append(model_with_constants)
            self.ensure_fresh_model(session, model_with_constants)

        self.print_dbg("Max #different models reached.")
        session.pop("Remove the stack [max #different models reached].")
        return result, SmtResponse.SAT

    def test_dummy_batch_mode(self) -> Tuple[bool, str]:
        dummy_constraints: List[str] = []
        fresh_vars: Set[FreshVariable] = SortedSet(key=lambda x: True)
        # Run the test
        for dummy in self.dummies:
            dummy_constraints += dummy.dummified_constraints()
            fresh_vars.update(dummy.fresh_vars)
        outcome, file_name, file_with_dummy = self.run_experiment(len(self.dummies), dummy_constraints, fresh_vars)

        if outcome == "unsat":
            composed_dummy = ComposedDummy(self.dummies, fresh_vars, file_name, self.timeout_per_query)

            if self.opt.timeout:
                # perform the minimization, with a timeout
                current_time = time.perf_counter()
                worker = DummyWorker(composed_dummy)
                p = multiprocessing.Process(target=worker.run)
                remaining_time = self.opt.timeout - (current_time - self.start_time) - 10

                p.start()
                p.join(remaining_time)

                if p.is_alive():
                    p.terminate()
                    p.join()
                    # the minimization timed out, we only have a partial result
                    dummy_constraint = worker.partial_result.value.decode()
                else:
                    dummy_constraint = worker.result.value.decode()
            else:
                dummy_constraint = composed_dummy.minimize(expected_outcome=outcome)
            self.print_dbg("%s is inconsistent; revealed via triggering term %s" % (self.filename, dummy_constraint))
            self.print_dbg("!! SUCCESS !!")

            # store the file with the minimized dummy
            if not os.path.isdir(self.solved_dir):
                ensure_dir_structure(self.solved_dir)
            if os.path.isfile(composed_dummy.test_file_name):  # we performed at least one minimization step
                solved_file = add_fname_suffix(os.path.join(self.solved_dir, os.path.basename(self.filename)),
                                               'dummy_min')
                shutil.copy(composed_dummy.test_file_name, solved_file)
            else:
                # there was no time left for minimization
                solved_file = add_fname_suffix(os.path.join(self.solved_dir, os.path.basename(self.filename)), 'dummy')
                shutil.copy(file_with_dummy, solved_file)
                merged_dummy: Dummy = composed_dummy.merge(composed_dummy.dummies, False)
                dummy_constraint = merged_dummy.abstract_constraint(merged_dummy.terms, True)
            return True, dummy_constraint

        # if outcome == "timeout":
            # we triggered matching loops
            # self.timeout_per_query = 0.1
        #    composed_dummy = ComposedDummy(self.dummies, fresh_vars, file_name, self.timeout_per_query)
        #    res = composed_dummy.minimize(expected_outcome=outcome)
        #    print(res)

        return False, ""
