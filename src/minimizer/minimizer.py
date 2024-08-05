###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2022 ETH Zurich.
###############################################################################

import io
import re
from os.path import join, dirname, basename, isdir
from typing import List, Callable, Dict

from sortedcontainers import SortedSet

from pysmt.smtlib.parser import Tokenizer
from src.minimizer.test_runner import TestRunner
from src.utils.preamble import STANDARD_PREAMBLE, STANDARD_MBQI_PREAMBLE
from src.minimizer.exceptions import MinimizerException
from src.utils.enums import SMTSolver, SmtCommand
from src.utils.file import canonize_fname, assert_file_readable, assert_file_writable, add_fname_suffix, \
    ensure_dir_structure
from src.utils.smt import are_smt_files_equisat
from src.utils.string import create_substituted_string
from src.utils.string import is_token_in_string
from src.utils.toposort import toposort, topo_key


class Minimizer:

    DBG_PREFIX = "Minimizer >> "

    # Pre-compute stuff for speed
    SMT_COMMAND_NAMES = list(map(lambda cmd: cmd.value, list(SmtCommand)))

    # Regexes
    ASSERTION_BODY_PATTERN = r'^\(' + str(SmtCommand.ASSERT) + '\s+(.+?)\s*\)\s*$'
    TOKEN_SYMBOL_PATTERN   = """%@#a-zA-Z0-9$_!?<>=+.-"""

    # Precomputing this speeds things up dramatically and allows to avoid regexes elsewhere.
    TOKEN_ALPHABET = "%@abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789$_!?<>=+.-"
    # import exrex
    # TOKEN_ALPHABET = ''.join(exrex.generate('['+TOKEN_SYMBOL_PATTERN+']'))
    TOKEN_DELIMITER = '[^%s]' % TOKEN_SYMBOL_PATTERN

    TOKEN_PATTERN          = '[' + TOKEN_SYMBOL_PATTERN + ']+'
    EXT_TOKEN_PATTERN      = "[^|()]+"  # If it includes sumbols outside of TOKEN_ALPHABET,
                                        #  then it should be wrapped into ||
    ASSERTION_NAME_PATTERN = r'^.*:named\s+\|?(' + TOKEN_PATTERN + ')\|?.*?$'
    QUANTIFIER_ASSERTION_PATTERN = \
                             r'(?:\(!\s*)?\((?:forall|exists)\s+(.*)\)(?:\s*\))?'
    ABSTRACT_DECL_PATTERN  = r'\((?:declare|define)-\w+\s+.*?\|?(' + TOKEN_PATTERN + ')\|?'
    DECL_NAME_TYPE_PATTERN = r'^\((?:declare|define)-\w+\s+.*?((?:\|' + EXT_TOKEN_PATTERN + '\||' + TOKEN_PATTERN + '))\s*(?:(?:\((.*?)\))?\s*(.*))?\)\s*$'
    FUNCDEF_BODY_PATTERN   = r'^\(' + str(SmtCommand.DEFINE_FUN) + '\s+\|?(' + TOKEN_PATTERN + \
                              ')\|?\s+\(((?:\(.*?\)\s*)*)\)\s+(' + TOKEN_PATTERN + \
                              ')\s+(\(?.+\)?)\)\s*$'
    CONSTDEF_BODY_PATTERN  = r'^\(' + str(SmtCommand.DEFINE_CONST) + '\s+\|?(' + TOKEN_PATTERN + \
                              ')\|?\s+(' + TOKEN_PATTERN + \
                              ')\s+(\(?.+\)?)\)\s*$'
    SORTDEF_BODY_PATTERN   = r'^\(' + str(SmtCommand.DEFINE_SORT) + '\s+\|?(' + TOKEN_PATTERN + \
                              ')\|?\s+\((.*?)\)\s+(.*?)\)\s*$'
    SORTDECL_PATTERN       = r'^\(' + str(SmtCommand.DECLARE_SORT) + '\s+\|?(' + TOKEN_PATTERN + \
                              ')\|?(?:\s+(\d+))?\s*\)\s*$'
    DECLARE_DATATYPES_PATTERN = r'^\(' + str(SmtCommand.DECLARE_DATATYPES) + '\s+\(\s*(\(' + TOKEN_PATTERN + \
                                 '\s+\d+\)\s*)*\)\s*(.*?)\s*\)\s*$'

    SMT_BUILTIN_AS_PATTERN = r'(\(as (%s) (%s)\))' % (TOKEN_PATTERN, TOKEN_PATTERN)

    @classmethod
    def parse_setinfo(cls, cmd):
        m = re.match(r"\(set-info\s+:(%s)\s+\|?(.+)\|?\s*\)" % cls.TOKEN_PATTERN, cmd)
        assert m and len(m.groups()) == 2, "not a valid set-info command: %s" % cmd
        info_slot = m.group(1).strip("|")
        info_text = m.group(2).strip()
        return info_slot, info_text

    # Static helper methods
    @classmethod
    def print_dbg(cls, msg, quite=False):
        if not quite:
            print(cls.DBG_PREFIX + msg)

    @staticmethod
    def is_assertion(cmd):
        return cmd.startswith('(' + str(SmtCommand.ASSERT))

    @staticmethod
    def is_checksat(cmd):
        return cmd.startswith('(' + str(SmtCommand.CHECK_SAT))

    @staticmethod
    def is_getunsatcore(cmd):
        return cmd.startswith('(' + str(SmtCommand.GET_UNSAT_CORE))

    @staticmethod
    def is_getproof(cmd):
        return cmd.startswith('(' + str(SmtCommand.GET_PROOF))

    @staticmethod
    def is_option__produce_unsat_cores(cmd):
        return cmd.startswith('(' + str(SmtCommand.SET_OPTION) + ' :produce-unsat-cores')

    @staticmethod
    def is_option__produce_proofs(cmd):
        return cmd.startswith('(' + str(SmtCommand.SET_OPTION) + ' :produce-proofs')

    @staticmethod
    def is_option__smt_core_minimize(cmd):
        return cmd.startswith('(' + str(SmtCommand.SET_OPTION) + ' :smt.core.minimize')

    @staticmethod
    def is_setoption(cmd):
        return cmd.startswith('(' + str(SmtCommand.SET_OPTION))

    @staticmethod
    def is_setlogic(cmd):
        return cmd.startswith('(' + str(SmtCommand.SET_LOGIC))

    @staticmethod
    def is_setinfo(cmd):
        return cmd.startswith('(' + str(SmtCommand.SET_INFO))

    @staticmethod
    def is_definesort(cmd):
        return cmd.startswith('(' + str(SmtCommand.DEFINE_SORT))

    @staticmethod
    def is_definefun(cmd):
        return cmd.startswith('(' + str(SmtCommand.DEFINE_FUN))

    @staticmethod
    def is_defineconst(cmd):
        return cmd.startswith('(' + str(SmtCommand.DEFINE_CONST))

    @staticmethod
    def is_comment(cmd):
        return cmd.startswith(';')

    @staticmethod
    def is_declaresort(cmd):
        return cmd.startswith('(' + str(SmtCommand.DECLARE_SORT))

    @staticmethod
    def is_declarefunction(cmd):
        return cmd.startswith('(' + str(SmtCommand.DECLARE_FUN))

    @staticmethod
    def is_declareconstant(cmd):
        return cmd.startswith('(' + str(SmtCommand.DECLARE_CONST))

    @staticmethod
    def is_declaredatatypes(cmd):
        return cmd.startswith('(' + str(SmtCommand.DECLARE_DATATYPES))

    @staticmethod
    def is_declaration(cmd):
        return cmd.startswith('(' + str(SmtCommand.DECLARE_SORT)) or \
               cmd.startswith('(' + str(SmtCommand.DEFINE_SORT)) or \
               cmd.startswith('(' + str(SmtCommand.DECLARE_CONST)) or \
               cmd.startswith('(' + str(SmtCommand.DEFINE_CONST)) or \
               cmd.startswith('(' + str(SmtCommand.DECLARE_DATATYPES)) or \
               cmd.startswith('(' + str(SmtCommand.DECLARE_FUN)) or \
               cmd.startswith('(' + str(SmtCommand.DEFINE_FUN))

    @staticmethod
    def is_push(cmd):
        return cmd.startswith('(' + str(SmtCommand.PUSH))

    @staticmethod
    def is_pop(cmd):
        return cmd.startswith('(' + str(SmtCommand.POP))

    @staticmethod
    def is_reset(cmd):
        return cmd == '(' + str(SmtCommand.RESET) + ')'

    @staticmethod
    def is_exit(cmd):
        return cmd == '(' + str(SmtCommand.EXIT) + ')'

    @staticmethod
    def is_getinfo(cmd):
        return cmd.startswith('(' + str(SmtCommand.GET_INFO))

    @classmethod
    def extract_declared_token(cls, cmd) -> str:
        res = re.match(cls.ABSTRACT_DECL_PATTERN, cmd)
        if not res:
            raise MinimizerException("cannot find token name in declaration line: " + str(cmd))
        token = res.group(1)
        return token

    @classmethod
    def is_token_in_string(cls, tk: str, a: str) -> bool:
        return is_token_in_string(cls.TOKEN_ALPHABET, tk, a, None, None)

    @classmethod
    def is_token_in_declaration_signature(cls, tk: str, decl: str) -> bool:

        """ Check that the token is used in this abstract declaration """

        declared_token = cls.extract_declared_token(decl)
        return cls.is_token_in_string(tk, decl) and tk != declared_token

    @classmethod
    def used_options_and_declarations(cls, commands: List[str],
                                      assertions: List[str],
                                      declarations: List[str]):

        """Return all the options, all the types and only the symbols (functions and variables) that appear in at least
        one of the assertions."""

        def is_token_useful(tk):
            # Check that the token is used at least in one of the assertions
            #  or function definitions
            try:
                next(filter(lambda a: tk in a and cls.is_token_in_string(tk, a), assertions))
            except StopIteration:
                try:
                    next(filter(lambda d: cls.is_token_in_declaration_signature(tk, d), declarations))
                except StopIteration:
                    return False
            return True

        def is_used_option_or_used_declaration(cmd):
            #print('is_used_option_or_used_declaration >>> '+cmd+'\n')
            if not cls.is_declaration(str(cmd)):
                return True
            declared_token = cls.extract_declared_token(cmd)
            return is_token_useful(declared_token)

        filter_object = filter(is_used_option_or_used_declaration, commands)
        filtered_commands = list(filter_object)
        return filtered_commands

    def remove_duplicates(self):
        buffer = SortedSet(key=lambda x: True)  # First come first serve
        assertions_without_name = SortedSet(key=lambda x: True)

        for command in self.all_commands:
            if self.is_assertion(command):
                # remove assertions with the same body, but different named annotation
                unnamed_assertion = self.get_assertion_without_name(command)
                if unnamed_assertion not in assertions_without_name:
                    buffer.add(command)
                    assertions_without_name.add(unnamed_assertion)
            else:
                buffer.add(command)
        self.all_commands = list(buffer)

    def rewrite_special_tokens(self):
        # TODO: refactor using [[Minimizer.__rewrite_tokens]]

        rename_map = {
            "(RegEx String)": "RegExStr",
            "RoundingMode": "RMode"  # see https://github.com/Z3Prover/z3/issues/4380
        }

        def add_rewrite_rule(token):
            nonlocal rename_map
            new_token = token
            changed = False
            if "#" in token:
                new_token = new_token.replace("#", "@sharp@")
                changed = True
            if "'" in token:
                new_token = new_token.replace("'", "@quote@")
                changed = True
            if changed:
                rename_map[token] = new_token

        # Collect rewrite rules
        for cmd in self.all_commands:

            if self.is_declaration(cmd):
                if self.is_declaresort(cmd):
                    token, _ = self._get_sortdecl_name_arity(cmd)
                else:
                    token, _, _ = self._get_decl_name_args_type(cmd)
                add_rewrite_rule(token)
                continue

            # Not a declaration.
            with io.StringIO(cmd) as stream:
                tokens = Tokenizer(stream, interactive=False)
                while True:
                    try:
                        token = tokens.consume_maybe()
                        if token.startswith("#") and (token[1] == "b" or token[1] == "x"):
                            # it is a BitVector
                            pass
                        else:
                            # it is a variable name
                            add_rewrite_rule(token)

                    except StopIteration:
                        break

        # Apply rewrite rules
        commands = self.all_commands[:]
        for token, new_token in rename_map.items():
            for cmd_num, _ in enumerate(self.all_commands):
                old_cmd = commands[cmd_num]
                commands[cmd_num] = old_cmd.replace(token, new_token)
        self.all_commands = commands

    def drop_unused_declarations(self):
        other_commands = self.get_other_commands()
        assertions = self.get_assertions()
        declarations = self.get_declarations()

        other_commands = self.used_options_and_declarations(commands=other_commands,
                                                            assertions=assertions,
                                                            declarations=declarations)
        self.all_commands = other_commands + assertions

    def is_command_useless(self, cmd):
        return True in [cmd.startswith('(' + str(useless_cmd)) for useless_cmd in self.useless_commands]

    def __assert_valid_smt(self):
        assert len(list(filter(lambda x: not x.startswith('('), self.all_commands))) == 0, \
            'valid SMT commands must start with an opening bracket'
        assert len(list(filter(lambda x: not x.endswith(')'), self.all_commands))) == 0, \
            'valid SMT commands must end with a closing bracket'

    ###################################################################
    #
    # DEFAULT FUNCTIONALITY IS IMPLEMENTED VIA CONSTRUCTOR
    #
    ###################################################################

    def __init__(self,
                 fin_name=None,
                 in_stream=None,
                 fout_name=None,
                 solver=None,
                 solver_timeout=300,
                 target_dir='/tmp/minimizer',
                 useless_commands=frozenset([SmtCommand.EXIT, SmtCommand.LABELS, SmtCommand.GET_INFO, SmtCommand.EVAL, SmtCommand.ECHO]),
                 str_commands_to_add=tuple(),
                 validate=True,
                 standardize=True,
                 remove_duplicates=False,
                 treat_special_tokens=True,
                 write_result_to_disk=True,
                 auto_confg=True,
                 add_check_sat=True,
                 mbqi=False,
                 ensure_incremental_mode=False):
        """
        Main routine: minimization tool for SMT2 files.
        :param fin_name: name of a readable, parsable, and type-correct SMT2 file
        :param in_stream: an optional StringIO to be used instead of an actual file
        :param fout_name: name of an existing writable file or a non-existing file in a writable directory
        :param target_dir: name of a directory where you would like to keep the artifacts from one of the mining phases
        :param solver: the SMT solver used to extract the unsat core
        :param useless_commands: the collection of SMT commands that should be removed
        :param str_commands_to_add: the list of (string) SMT2 commands that should be added
        :param auto_confg: whether standard set-option commands should be ensured
        :param add_check_sat: whether check-sat command should be ensured
        :param mbqi: whether to set :smt.mbqi to true or false
        :param ensure_incremental_mode: whether to ensure that a (push) command is added at the end of the file
                                        (right before the first check-sat if one is present)
        """

        assert not in_stream or not write_result_to_disk, \
            "Minimizer: incompatible options `in_stream` and `write_result_to_disk`"
        assert not in_stream or not validate, \
            "Minimizer: incompatible options `in_stream` and `validate`"

        self.mbqi = mbqi
        self.original_outcome = None
        self.treat_special_tokens = treat_special_tokens

        if not in_stream:
            self.fin_name = canonize_fname(fin_name)
            self.target_dir = canonize_fname(target_dir)
            self.fout_name = fout_name # may be None
        self.solver = solver
        self.solver_timeout = solver_timeout
        self.all_commands = []
        self.useless_commands = useless_commands

        self.all_commands += str_commands_to_add

        if not in_stream:
            self.load_file()
        else:
            self.load_stream(in_stream)

        if auto_confg:
            self.ensure_correct_smt_configuration()
        if add_check_sat:
            self.ensure_checksat()
        if ensure_incremental_mode:
            self.__ensure_push_command()
        if standardize:
            self.standardize_smt_file()

        if remove_duplicates:
            self.remove_duplicates()
            self.__assert_valid_smt()

        if write_result_to_disk:

            suffix_list = []
            if standardize:
                suffix_list.append('std')
            if remove_duplicates:
                suffix_list.append('unique')

            if len(suffix_list) > 0:
                new_suffix = '_'.join(suffix_list)
            else:
                new_suffix = 'min'

            self.save_to_disk(new_suffix=new_suffix)

        if validate:
            self.do_validate(other_file=self.fout_name)

    def set_mbqi(self, flag: bool, save=False):
        something_changed = False
        new_suffix = None
        original_commands = self.all_commands
        new_commands = []

        for orig_cmd in original_commands:
            cmd = orig_cmd.lower()

            if cmd == '(set-option :smt.mbqi false)' and flag:
                new_commands.append('(set-option :smt.mbqi true)')
                something_changed = True
                new_suffix = 'mbqi'
                continue

            if cmd == '(set-option :smt.mbqi true)' and not flag:
                new_commands.append('(set-option :smt.mbqi false)')
                something_changed = True
                new_suffix = 'ematching'
                continue

            new_commands.append(orig_cmd)

        self.mbqi = flag

        if something_changed:
            self.all_commands = new_commands
            if save:
                assert new_suffix, "new suffix must be set at this point"
                return self.save_to_disk(new_suffix=new_suffix, quiet=True)
            else:
                return new_commands
        else:
            return None

    def do_validate(self, other_file):
        is_multi_query_file = self.is_multiquery()
        is_equisat, orig_outcome, new_outcome = are_smt_files_equisat(self.solver, self.fin_name, other_file,
                                                                      timeout=self.solver_timeout,
                                                                      use_soft_timeout=is_multi_query_file,
                                                                      multiquery=is_multi_query_file)
        if not is_equisat:
            self.print_dbg(' Mismatch in outcomes before and after transformation: \n%s\n<orig:vs:new>\n%s'
                           % (orig_outcome, new_outcome))
        else:
            self.print_dbg(' Preserved satisfiability after transformation (status: %s)' % orig_outcome)

        self.original_outcome = orig_outcome

        return is_equisat, orig_outcome, new_outcome

    def get_assertions(self, as_dict=False):
        if as_dict:
            return dict(filter(lambda elem: self.is_assertion(elem[1]), enumerate(self.all_commands)))
        return list(filter(lambda cmd: self.is_assertion(cmd), self.all_commands))

    def _get_sortdecl_name_arity(self, sdecl):
        m = re.match(self.SORTDECL_PATTERN, sdecl)
        assert len(m.groups()) == 2
        return m.group(1), m.group(2)

    def _get_decl_name_args_type(self, decl):
        if self.is_declarefunction(decl):
            m = re.match(self.DECL_NAME_TYPE_PATTERN, decl)
            assert len(m.groups()) == 3
            args_str = m.group(2)
            args = tuple(re.sub(r"\s+", " ", args_str).strip().split())
            token = m.group(1).strip("|")
            typ = m.group(3)
        else:
            assert self.is_declareconstant(decl)
            m = re.match(self.DECL_NAME_TYPE_PATTERN, decl)
            assert len(m.groups()) == 3 and m.group(2) is None
            args = tuple()
            token = m.group(1).strip("|")
            typ = m.group(3)

        return token, tuple(args), typ

    def get_declarations(self):
        gen = filter(lambda cmd: self.is_declaration(cmd), self.all_commands)
        return list(gen)

    def get_fun_const_decl_triplets(self):
        gen = filter(lambda cmd: self.is_declarefunction(cmd) or self.is_declareconstant(cmd), self.all_commands)
        return sorted(map(lambda d: self._get_decl_name_args_type(d), gen))

    def get_other_commands(self):
        return list(filter(lambda cmd: not self.is_assertion(cmd) and
                                       not self.is_command_useless(cmd), self.all_commands))

    def get_setinfo(self):
        return list(filter(lambda cmd: self.is_setinfo(cmd), self.all_commands))

    def get_setlogic(self):
        return list(filter(lambda cmd: self.is_setlogic(cmd), self.all_commands))

    def get_setoption(self):
        return list(filter(lambda cmd: self.is_setoption(cmd), self.all_commands))

    def get_sort_definitions(self):
        return list(filter(lambda cmd: self.is_definesort(cmd), self.all_commands))

    def get_sort_declarations(self):
        return list(filter(lambda cmd: self.is_declaresort(cmd), self.all_commands))

    def get_datatype_declarations(self):
        return list(filter(lambda cmd: self.is_declaredatatypes(cmd), self.all_commands))

    def get_fun_definitions(self):
        return list(filter(lambda cmd: self.is_definefun(cmd), self.all_commands))

    def get_const_definitions(self):
        return list(filter(lambda cmd: self.is_defineconst(cmd), self.all_commands))

    def get_unsatcore_commands(self):
        return list(filter(lambda cmd: self.is_getunsatcore(cmd)
                                       or self.is_option__produce_unsat_cores(cmd)
                                       or self.is_option__smt_core_minimize(cmd), self.all_commands))

    def get_proof_commands(self):
        return list(filter(lambda cmd: self.is_getproof(cmd)
                                       or self.is_option__produce_proofs(cmd), self.all_commands))

    def get_meta_commands(self):
        return list(filter(lambda cmd: self.is_setinfo(cmd) or self.is_setlogic(cmd), self.all_commands))

    def get_checksat_commands(self):
        return list(filter(lambda cmd: self.is_checksat(cmd), self.all_commands))

    def get_options_info_logic(self):
        return list(filter(lambda cmd: self.is_setoption(cmd) or self.is_setinfo(cmd) or self.is_setlogic(cmd),
                           self.all_commands))

    def get_options(self):
        return list(filter(lambda cmd: self.is_setoption(cmd), self.all_commands))

    def ensure_correct_smt_configuration(self):
        self.all_commands = list(filter(lambda c: not self.is_setoption(c), self.all_commands))

        padding = len(self.get_meta_commands())
        std_preamble = STANDARD_MBQI_PREAMBLE if self.mbqi else STANDARD_PREAMBLE
        for conf_num, conf_bit in enumerate(std_preamble):
            if conf_bit not in self.all_commands:
                self.all_commands.insert(conf_num + padding, conf_bit)

    def ensure_checksat(self):
        chsat_cmd = '(' + str(SmtCommand.CHECK_SAT) + ')'
        if chsat_cmd not in self.all_commands:
            self.all_commands.append(chsat_cmd)

        self.__assert_valid_smt()

    def add_missing_assertion_names(self):
        """
        Add name annotations to the SMT assertions, represented as strings.
        The naming schema guarantees that manually-specified names will remain unchanged,
         and that the freshly generated names will not collide with the existing ones.
        """
        named_assertions = dict()
        # map each unnamed assertion to its original position
        unnamed_assertions = dict()

        # Separate already named assertions from unnamed ones
        original_assertions = self.get_assertions()
        for a in original_assertions:
            matches = re.match(self.ASSERTION_NAME_PATTERN, a)
            if matches:
                a_name = matches.group(1)
                if a_name in named_assertions:
                    raise MinimizerException('Found two assertions with the same name:\n' + str(a) + '\n' + \
                                             ' ... and\n' + str(named_assertions[a_name]) + '\n')
                named_assertions[a_name] = a
            else:
                unnamed_assertions[a] = self.all_commands.index(a)

        assert set(unnamed_assertions.keys()).union(list(named_assertions.values())) == set(original_assertions)

        # Generate a unique prefix that is safe to use for generating assertion names
        def generate_unique_prefix(collection, prefix):
            if True in [item.startswith(prefix) for item in collection]:
                return generate_unique_prefix(collection, '_' + prefix)
            return prefix

        unique_prefix = generate_unique_prefix(named_assertions.keys(), 'A')

        # Augment previously unnamed assertions with automatically generated unique names
        for (counter, assertion) in enumerate(unnamed_assertions):
            new_name = unique_prefix + str(counter)

            # Extract assertion body
            pattern = self.ASSERTION_BODY_PATTERN
            matches = re.match(pattern, assertion)
            if not matches:
                raise MinimizerException("could not get assertion body for `" + assertion + "`" + \
                                         "using given regex `" + pattern + "`")

            # Create a new assertion, adding ` :named [unique_prefix][counter]`
            newly_named_assertion = '(' + str(SmtCommand.ASSERT) + \
                                    ' (! ' + matches.group(1) + ' :named ' + new_name + '))'
            #named_assertions[new_name] = newly_named_assertion
            place = unnamed_assertions[assertion]
            self.all_commands[place] = newly_named_assertion

    @classmethod
    def get_assertion_name(cls, assertion):
        matches = re.match(cls.ASSERTION_NAME_PATTERN, assertion)
        if matches:
            return matches.group(1)
        return None

    @classmethod
    def get_assertion_without_name(cls, assertion):
        name = cls.get_assertion_name(assertion)
        if name is not None:
            res = create_substituted_string(cls.TOKEN_ALPHABET, name, '', assertion)
            res = res[len("(assert (! "):-len(" :named ))")]
            return "(assert %s)" % res
        return assertion

    @classmethod
    def get_assertion_body(cls, assertion):
        matches = re.match(cls.ASSERTION_BODY_PATTERN, assertion)
        return matches.group(1)

    def __save_artifact(self, artifact, base_name, tag=None, quiet=False):
        if tag:
            output_file_name = join(self.target_dir, add_fname_suffix(base_name, tag))
        else:
            output_file_name = join(self.target_dir, base_name)

        self.write_lines_to_file(artifact, output_file_name, quiet=quiet)
        return output_file_name

    def run(self, solver=SMTSolver.Z3,
            timeout=1.5, with_soft_timeout=False,
            seed: int or None = 0,
            with_timing=False,
            new_suffix=None, accumulate=True,
            with_unsat_core=False,
            with_proof=False,
            get_reason_unknown=False,
            enum_inst=False,
            quiet=False, progress_token='.',
            save_artifact_lbd: Callable[[str, Dict[str, any]], bool] or None = None, art_tag=None):

        """
        A unification method for running SMT solvers or Vampire, getting the response, and archiving the artifacts.

        :param solver:             The SMT solver to run the test with or Vampire.
        :param timeout:            Tool timeout (in seconds)
        :param with_soft_timeout:  Whether to use soft timeouts (useful for multi-query files)
        :param seed                Specify the random seed for the underlying tools
        :param with_timing         Whether to also return the runtime (as last tuple component; in seconds)
        :param new_suffix:         If set to some string, used to extend the suffix of the output file; example:
                                   new_suffix=='world && 'hello.smt2' --> 'hello_world.smt2'
        :param accumulate:         Whether to extend the persistent file name `self.fout_name` with the suffix.
                                   It's useful to set this to False for performing multiple similar runs.
        :param with_unsat_core:    Whether to extract unsat core (may negatively affect the completeness of the solver)
        :param with_proof:         Whether to construct the proof
        :param enum_inst:          Whether to use enumerative instantiation when E-matching saturates (for CVC4/cvc5)
        :param get_reason_unknown: Whether to ask the solver why the status is unknown
        :param quiet:              Whether to indicate success with a single character to STDOUT or report more info.
        :param progress_token:     If `quite`, which character to use.
        :param save_artifact_lbd:  Whether to save the run file and the outcome into a separate file in `self.target_dir`
                                   The first lambda arg is the outcome:str; the seconds lambda arg is aux:Dict[str, Any]
        :param art_tag:            The suffix that will be added to the artifact file.
        :return:                   If `with_unsat_core` and     `with_timing`, then [status:str], [unsat core:str], [timing:int(seconds)]
                                   If `with_unsat_core` and not `with_timing`, then [status:str], [unsat core:str]
                                   If `with_proof` and          `with_timing`, then [status:str], [proof:str], [timing:int(seconds)]
                                   If `with_proof` and not      `with_timing`, then [status:str], [proof:str]
                                   If `get_reason_unknown` and not `with_timing`, then [status:str], [reason_unknown:str]
                                   If not `with_unsat_core` and `with_timing`, then [status:str], [timing:int(seconds)]
                                   Otherwise, [status:str] as a single string.
        """

        original_smt_commands = self.all_commands[:]

        if with_unsat_core:
            self.ensure_unsat_core_commands()

        if with_proof:
            self.ensure_proof_commands()

        if get_reason_unknown:
            self.all_commands.append('(' + str(SmtCommand.GET_INFO) + ' :reason-unknown)')

        if solver is SMTSolver.VAMPIRE:
            self.all_commands = self.__vampire_commands()
        elif solver is SMTSolver.CVC4 or solver is SMTSolver.CVC5:
            self.all_commands = self.__cvc_commands()

        actual_file_name = self.save_to_disk(new_suffix=new_suffix, accumulate=accumulate, quiet=True)

        runner = TestRunner(solver, timeout=timeout, seed=seed, enum_inst=enum_inst)
        smt_status, attachment, elapsed_time = runner.run_test_timed(actual_file_name, ignore_warnings=True,
                                                                      use_soft_timeout=with_soft_timeout, quiet=quiet)
        if quiet:
            print(progress_token, end='', flush=True)
        else:
            self.print_dbg('%s returned "%s" in %.3f seconds on %s'
                           % (str(solver), smt_status, elapsed_time, actual_file_name))

        aux = dict()
        if with_timing and elapsed_time:
            aux['time'] = elapsed_time
        if get_reason_unknown:
            aux['reason'] = ' '.join(attachment)
        if with_proof:
            aux['proof'] = attachment

        if with_unsat_core:
            unsat_core_axioms: List[str] = []
            for line in attachment:
                line = line.replace('(', '')
                line = line.replace(')', '')
                if len(line) == 0:
                    continue
                unsat_core_axioms += line.split(' ')
            attachment = unsat_core_axioms

        if not save_artifact_lbd or save_artifact_lbd(smt_status, **aux):
            if art_tag and not isinstance(art_tag, str):
                art_tag = str(art_tag(smt_status))
            artifact = self.all_commands + [';' + runner.get_last_command(), ';' + smt_status]
            if get_reason_unknown:
                artifact.append(';(%s)' % ' '.join(attachment))
            if with_unsat_core:
                artifact.append(';unsat core: %s' % ' '.join(attachment))
            if with_proof:
                artifact.append('\n')
                for line in attachment:
                    artifact.append(';%s' % line)
            if with_timing:
                artifact.append(';elapsed_time=%fs' % elapsed_time)
            artifact_finename = self.__save_artifact(artifact, basename(actual_file_name), tag=art_tag)
            if not quiet:
                self.print_dbg("Saving artifact to %s" % artifact_finename)

        self.all_commands = original_smt_commands

        if with_unsat_core or with_proof or get_reason_unknown:
            if with_timing:
                return smt_status, attachment, elapsed_time
            else:
                return smt_status, attachment
        else:
            if with_timing:
                return smt_status, elapsed_time
            else:
                return smt_status

    @classmethod
    def smt_key(cls, x):
        # The position of the current SMT command in [[SmtCommand]] is the primary sorting factor
        primary = [cmd in x for cmd in cls.SMT_COMMAND_NAMES].index(True)
        if cls.is_assertion(x):
            # Secondary sorting factor for assertions
            matches = re.match(cls.ASSERTION_NAME_PATTERN, x)
            if matches:
                # The name of the assertion is the secondary sorting factor (additional effort to keep the `natural` order)
                # For more details: http://www.codinghorror.com/blog/2007/12/sorting-for-humans-natural-sort-order.html
                convert = lambda text: int(text) if text.isdigit() else text.lower()
                alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
                secondary = alphanum_key(matches.group(1))
            else:
                # Assertion with no name
                secondary = []
        else:
            # Not an assertion (cannot have a name)
            secondary = []

        return primary, secondary

    @classmethod
    def topo_smt_key(cls, x, topo_layers=None):
        """
        A criterium for comparing SMT command lines.

        Use this function for sorting SMT files according to the order
         specified in [[cls.SMT_COMMAND_NAMES]] (main factor)
         or according to the topological order of the command dependencies.

        Note: comments are currently disallowed if this method is used.
        """
        assert not cls.is_comment(x), "sorting SMT commands with comments (`;`) is not supported"

        if topo_layers:
            topological = topo_key(topo_layers, x)
        else:
            topological = 0
        primary, secondary = cls.smt_key(x)

        command_class = 0
        if SmtCommand.SET_INFO.value in x:
            command_class = -10
        elif SmtCommand.SET_OPTION.value in x:
            command_class = -9
        elif SmtCommand.DECLARE_SORT.value in x:
            command_class = 3
        elif SmtCommand.DECLARE_DATATYPES.value in x:
            command_class = 4
        elif SmtCommand.DEFINE_FUN.value in x:
            command_class = 5
        elif SmtCommand.DECLARE_FUN.value in x:
            command_class = 6
        elif SmtCommand.DECLARE_CONST.value in x:
            command_class = 7
        elif SmtCommand.ASSERT.value in x:
            command_class = 8
        elif SmtCommand.CHECK_SAT.value in x:
            command_class = 9
        elif SmtCommand.GET_UNSAT_CORE.value in x:
            command_class = 10

        if topological >= 0:
            command_class = 0

        return command_class, topological, primary, secondary

    def sort_commands(self):

        """Sort the commands according to the sorting criterium specified in [[smt_key]]."""

        # Define dependency predicate for declarations
        def depends_on(decl, dep) -> bool:
            assert self.is_declaration(decl)
            assert self.is_declaration(dep)
            dep_token = self.extract_declared_token(dep)
            return self.is_token_in_declaration_signature(dep_token, decl)

        # Compute dependency data for declarations
        dep_data = dict()
        all_decls = set(filter(lambda cmd: self.is_declaration(cmd), self.all_commands))
        for decl in all_decls:
            dep_data[decl] = {dep for dep in all_decls if depends_on(decl, dep)}

        if len(dep_data) > 0:
            topo_layers = list(toposort(dep_data))
        else:
            topo_layers = None

        def k(c):
            res = self.topo_smt_key(c, topo_layers)
            return res

        self.all_commands.sort(key=k)

    def _commands_from_stream(self, in_stream, remove_comments):
        # Read lines as-is:
        lines = in_stream.readlines()
        commands = self.extract_commands(lines, remove_comments)
        # Save canonical commands representation
        self.all_commands = commands
        # Validate
        self.__assert_valid_smt()

    def load_stream(self, in_stream: io.StringIO, remove_comments=True):

        """Read contents from a stream, canonize the format, and store the commands into a buffer."""

        self._commands_from_stream(in_stream, remove_comments)


    def load_file(self, file_path=None, remove_comments=True):

        """Load contents of a file, canonize the format, and store the commands into a buffer."""

        if file_path is None:
            required_file_path = self.fin_name
        else:
            required_file_path = file_path

        # Check that the file exists and is readable
        assert_file_readable(required_file_path)
        self.print_dbg("Reading from %s" % required_file_path)

        with open(required_file_path, 'r') as in_file:
            self._commands_from_stream(in_file, remove_comments)

    def extract_commands(self, lines, remove_comments=True):
        # Get rid of line terminators
        lines = map(lambda line: line.strip(), lines)
        # Each comment should start with a new line
        lines = [item for sublist in map(lambda line: re.split('(;.*)', line), lines) for item in sublist]
        # Remove comments?
        if remove_comments:
            lines = list(filter(lambda line: not self.is_comment(line), lines))
        # Move the beginning of each SMT command to a new line
        for cmd in SmtCommand:
            lines = [item for sublist in
                     map(lambda line: re.split(r'(\(%s%s)' % (str(cmd), self.TOKEN_DELIMITER), line), lines) for item in
                     sublist]
        # Join multi-line commands into single-line commands
        ind = 0
        commands = ['']
        after_comment = False
        for line in lines:
            if self.is_comment(line):
                # Start of comment
                ind += 1
                commands.append(line)
                after_comment = True
            elif after_comment or True in [re.match(r'\(%s(?:%s|$)' % (cmd, self.TOKEN_DELIMITER), line) is not None
                                           for cmd in self.SMT_COMMAND_NAMES]:
                # Continuation after comment or start of SMT command
                ind += 1
                commands.append(line)
                after_comment = False
            else:
                # Continuation of SMT command
                commands[ind] += (' ' + line)
                after_comment = False
        # print(commands)
        # Filter out empty lines
        commands = list(filter(lambda line: len(line.strip()) > 0, commands))
        # Remove trailing spaces
        commands = list(map(lambda cmd: re.sub(r'\s+', ' ', cmd).strip(), commands))
        return commands

    def write_lines_to_file(self, lines, output_file_name, quiet=True):
        assert not isdir(output_file_name) # must be an existing or future file, but not a directory

        ensure_dir_structure(dirname(output_file_name))
        # Check that we can write into this file
        assert_file_writable(output_file_name)
        if not quiet:
            self.print_dbg("Writing to file: " + output_file_name)

        # Save canonical representation of SMT commands
        with open(output_file_name, 'w') as out_file:
            for item in lines:
                out_file.write('%s\n' % item)

    def __get_current_file_name(self, new_suffix=None, accumulate=True):

        # No output info provided?
        if new_suffix is None and self.fout_name is None:
            self.fout_name = result = add_fname_suffix(self.fin_name, 'min')

        elif self.fout_name is None:
            self.fout_name = result = add_fname_suffix(self.fin_name, new_suffix)

        elif new_suffix:
            result = add_fname_suffix(self.fout_name, new_suffix)
        else:
            result = self.fout_name

        if accumulate:
            self.fout_name = result

        return result

    def save_to_disk(self, new_suffix=None, quiet=True, accumulate=True,
                     as_artifact=False, as_out_file=True, for_vamp=False):
        """
        Save canonized commands into a file.
        """
        new_file_name = self.__get_current_file_name(new_suffix=new_suffix, accumulate=accumulate)
        self.print_dbg('Writing to %s' % new_file_name)
        if for_vamp:
            commands = self.__vampire_commands()
        else:
            commands = self.all_commands
        if as_out_file:
            self.write_lines_to_file(commands, new_file_name, quiet)
        if as_artifact:
            artifact_file = self.__save_artifact(commands, basename(new_file_name))
            if not as_out_file:
                return artifact_file

        return new_file_name

    ###################################################################
    #
    # NON-DEFAULT FUNCTIONALITY IS IMPLEMENTED VIA THE METHODS BELOW
    #
    ###################################################################

    def rewrite_declarations(self):

        for sort_decl in self.get_sort_declarations():

            m = re.match(self.SORTDECL_PATTERN, sort_decl)
            assert m and (len(m.groups()) == 1 or len(m.groups()) == 2)
            sort_name = m.group(1)
            sort_arity = m.group(2)
            if not sort_arity:
                new_sort_decl = '(' + str(SmtCommand.DECLARE_SORT) + ' ' + sort_name + ' 0)'
                place = self.all_commands.index(sort_decl)
                self.all_commands[place] = new_sort_decl

        # TODO: implement this for the general case
        for dt_decl in self.get_datatype_declarations():
            print(self.DECLARE_DATATYPES_PATTERN)
            print(dt_decl)

            m = re.match(self.DECLARE_DATATYPES_PATTERN, dt_decl)

            assert m and len(m.groups()) == 2
            assert m.group(1) is None, 'explicit sort signatures in declare-datatypes commands are not supported yet'

            if dt_decl == '(declare-datatypes () (( $Snap ($Snap.unit) ($Snap.combine ($Snap.first $Snap) ($Snap.second $Snap)))))':
                new_sort_decl = '(declare-sort $Snap 0)'
                new_sort_constructor_0 = '(declare-fun $Snap.unit () $Snap)'
                new_sort_constructor_1 = '(declare-fun $Snap.combine ($Snap $Snap) $Snap)'
                place = self.all_commands.index(dt_decl)
                self.all_commands[place] = new_sort_decl
                self.all_commands.insert(place + 1, new_sort_constructor_0)
                self.all_commands.insert(place + 2, new_sort_constructor_1)

            if dt_decl == '(declare-datatypes () ((Fuel (ZFuel) (SFuel (prec Fuel)))))':
                new_sort_decl = '(declare-sort Fuel 0)'
                new_sort_constructor_0 = '(declare-fun ZFuel () Fuel)'
                new_sort_constructor_1 = '(declare-fun SFuel (Fuel) Fuel)'
                place = self.all_commands.index(dt_decl)
                self.all_commands[place] = new_sort_decl
                self.all_commands.insert(place + 1, new_sort_constructor_0)
                self.all_commands.insert(place + 2, new_sort_constructor_1)

    def rewrite_definitions(self):
        """
        This method rewrites function definitions using function declarations and universally-
         quantified assertions. This is helpful, e.g., for extracting minimal unsat cores.
        """

        for const_def in self.get_const_definitions():

            m = re.match(self.CONSTDEF_BODY_PATTERN, const_def)
            assert len(m.groups()) == 3
            const_name = m.group(1)
            const_type = m.group(2)
            const_body = m.group(3)

            new_const_decl = '(' + str(SmtCommand.DECLARE_FUN) + ' ' + const_name + ' () ' + const_type + ')'
            new_const_def_assert = '(' + str(SmtCommand.ASSERT) + ' (! (= ' + const_name + ' ' + const_body + ') :named ' + '|' + const_name + '-Rewrite-Definition|' + '))'

            place = self.all_commands.index(const_def)
            self.all_commands[place] = new_const_decl
            self.all_commands.insert(place + 1, new_const_def_assert)

        for fun_def in self.get_fun_definitions():

            m = re.match(self.FUNCDEF_BODY_PATTERN, fun_def)
            assert m and len(m.groups()) == 4
            fun_name = m.group(1)
            fun_args = m.group(2)
            fun_type = m.group(3)
            fun_body = m.group(4)

            if not isinstance(fun_args, str) or fun_args == '':
                # Handle non-quantified case
                new_fun_decl = '(' + str(SmtCommand.DECLARE_FUN) + ' ' + fun_name + ' () ' + fun_type + ')'
                new_fun_def_assert = '(' + str(SmtCommand.ASSERT) + ' (! (= ' + fun_name + ' ' + fun_body + ') :named ' + '|' + fun_name + '-Rewrite-Definition|' + '))'
            else:
                # Handle quantified case
                _argelem = re.sub('\s+', ' ', re.sub('[^' + self.TOKEN_SYMBOL_PATTERN + ']', ' ', fun_args)).strip().split()
                qvar_types = ' '.join(_argelem[1::2])
                qvar_names = ' '.join(_argelem[::2])

                new_fun_decl = '(' + str(SmtCommand.DECLARE_FUN) + ' ' + fun_name + ' (' + qvar_types + ') ' + fun_type + ')'
                new_fun_def_assert = '(' + str(SmtCommand.ASSERT) + ' (! (forall (' + fun_args + ') (! (= (' + fun_name + ' ' + qvar_names + ') ' + fun_body + ')' + ' :pattern ((' + fun_name + ' ' + qvar_names + ')))) :named ' + '|' + fun_name + '-Rewrite-Definition|' + '))'

            place = self.all_commands.index(fun_def)
            self.all_commands[place] = new_fun_decl
            self.all_commands.insert(place + 1, new_fun_def_assert)

    def remove_unsat_core_commands(self):
        unsat_core_generation_commands = self.get_unsatcore_commands()
        for ucg_command in unsat_core_generation_commands:
            self.all_commands.remove(ucg_command)

    def remove_proof_commands(self):
        proof_commands = self.get_proof_commands()
        for proof_command in proof_commands:
            self.all_commands.remove(proof_command)

    def remove_useless_commands(self):
        self.all_commands = list(filter(lambda cmd: not self.is_command_useless(cmd), self.all_commands))

    def __ensure_push_command(self):
        """
        Make sure that there is a push command that enables incremental mode in Z3
        """
        assert len(self.get_checksat_commands()) <= 1, "too many check-sat commands at this point"

        push = '(push);  Enable incremental mode'

        last_cmd = self.all_commands[-1]
        if self.is_checksat(last_cmd):
            second_last_cmd = self.all_commands[-2]
            if self.is_push(second_last_cmd):
                return

            self.all_commands.insert(-1, push)
        else:
            self.all_commands.append(push)

    def ensure_unsat_core_commands(self):
        """
        Make sure that there is a get-unsat-core command and the `:produce-unsat-cores` option to true.
        """
        current_unsat_core_commands = list(filter(lambda c: self.is_getunsatcore(c), self.all_commands))
        if len(current_unsat_core_commands) == 0:
            self.all_commands.insert(0, '(' + str(SmtCommand.SET_OPTION) + ' :produce-unsat-cores true)')
            self.all_commands.append('(' + str(SmtCommand.GET_UNSAT_CORE) + ')')
        elif len(current_unsat_core_commands) < 2:
            raise MinimizerException('The number of unsat core commands should be at least 2!')

    def ensure_proof_commands(self):
        """
        Make sure that there is a get-proof command and the `:produce-proofs` option to true.
        """
        current_proof_commands = list(filter(lambda c: self.is_getproof(c), self.all_commands))
        if len(current_proof_commands) == 0:
            self.all_commands.insert(0, '(' + str(SmtCommand.SET_OPTION) + ' :produce-proofs true)')
            self.all_commands.append('(' + str(SmtCommand.GET_PROOF) + ')')
        elif len(current_proof_commands) < 2:
            raise MinimizerException('The number of proof commands should be at least 2!')

    def standardize_smt_file(self):
        self.rewrite_definitions()
        self.__assert_valid_smt()

        self.rewrite_declarations()
        self.__assert_valid_smt()

        #self.rewrite_overloaded_symbols()
        #self.__assert_valid_smt()

        self.rewrite_builtin_symbols()
        self.__assert_valid_smt()

        self.__rewrite_smt_commands()
        self.__assert_valid_smt()

        self.remove_unsat_core_commands()
        self.__assert_valid_smt()

        self.remove_useless_commands()
        self.__assert_valid_smt()

        self.add_missing_assertion_names()
        self.__assert_valid_smt()

        if self.treat_special_tokens:
            # See https://github.com/Z3Prover/z3/issues/4380
            fdp = self.__first_declaration_place()
            if '(declare-sort RMode 0)' not in self.all_commands:
                self.all_commands.insert(fdp+1, "(declare-sort RMode 0)")
            if "(declare-sort RegExStr 0)" not in self.all_commands:
                self.all_commands.insert(fdp+1, "(declare-sort RegExStr 0)")
            self.rewrite_special_tokens()
            self.__assert_valid_smt()

    def __first_assertion_place(self):
        for place, command in enumerate(self.all_commands):
            if self.is_assertion(command):
                return place
        return len(self.all_commands)

    def __last_assertion_place(self):
        last_place = 0
        for place, command in enumerate(self.all_commands):
            if self.is_assertion(command):
                last_place = place
        return last_place

    def __first_declaration_place(self):
        for place, command in enumerate(self.all_commands):
            if self.is_declaration(command):
                return place
        return len(self.all_commands)

    def __last_declaration_place(self):
        last_place = 0
        for place, command in enumerate(self.all_commands):
            if self.is_declaration(command):
                last_place = place
        return last_place

    @classmethod
    def is_quantified_assertion(cls, cmd):
        assert cls.is_assertion(cmd)
        m1 = re.match(cls.ASSERTION_BODY_PATTERN, cmd)
        assert len(m1.groups()) == 1
        assertion_body = m1.group(1)
        m2 = re.match(cls.QUANTIFIER_ASSERTION_PATTERN, assertion_body)
        return bool(m2)

    def minimize_unsat_core(self, rec_level=0, timeout=1.0):
        """
        This method runs the SMT solver on all subsets of the original assertions in order to obtain
         the Minimal Effective Unsat Core (MEUC).
        """

        if rec_level == 0:
            assert self.solver
            actual_status = self.run(new_suffix='meuc', quiet=True, timeout=timeout, progress_token='-')
            assert actual_status == 'unsat'

        original_assertions = tuple(self.get_assertions())
        something_changed = False

        for assertion in original_assertions:
            place = self.all_commands.index(assertion)
            self.all_commands.remove(assertion)
            actual_status = self.run(quiet=True, timeout=timeout, progress_token='-')
            if actual_status != 'unsat':
                self.all_commands.insert(place, assertion)
            else:
                something_changed = True

        print()

        if something_changed:
            self.minimize_unsat_core(rec_level + 1)

        self.save_to_disk(quiet=True)

    def __rewrite_tokens(self, add_rewrite_rule):
        rename_map = dict()

        # Collect rewrite rules
        for cmd in self.all_commands:
            if self.is_declaration(cmd):
                m = re.match(self.DECL_NAME_TYPE_PATTERN, cmd)
                assert m and len(m.groups()) > 0, "broken regex"
                token = m.group(1)
                add_rewrite_rule(token, rename_map)
                continue

            # Not a declaration.
            with io.StringIO(cmd) as stream:
                tokens = Tokenizer(stream, interactive=False)
                while True:
                    try:
                        token = tokens.consume_maybe()
                        add_rewrite_rule(token, rename_map)
                    except StopIteration:
                        break

        # Apply rewrite rules
        commands = self.all_commands[:]
        for token, new_token in rename_map.items():
            for cmd_num, _ in enumerate(self.all_commands):
                old_cmd = commands[cmd_num]
                commands[cmd_num] = old_cmd.replace(token, new_token)
        self.all_commands = commands

    def __remove_escape_symbols(self):
        # Vamp. distinguishes '|Token|' and 'Token'

        def add_rewrite_rule(token, rename_map):
            new_token = token
            changed = False
            if token.startswith("|") and token.endswith("|"):
                new_token = token.strip("|")
                changed = True
            if changed:
                rename_map[token] = new_token

        return self.__rewrite_tokens(add_rewrite_rule)

    def __vampire_commands(self):

        """
        This method creates SMT commands that are logically equivalent to [[self.all_commands]]
        but they can be parsed by Vampire.

        :return: A list of SMT commands that can be parsed by Vampire
        """

        rounding_mode_declaration = '(declare-sort RoundingMode 0)'
        string_sort_declaration = '(declare-sort String 0)'
        new_smt_commands = []

        self.__remove_escape_symbols()

        for smt_command in self.all_commands:
            # Vampire does not have built-in '/' but understands 'div'
            #smt_command = re.sub(r'\(\/ ', '(div ', smt_command)

            # Vampire does not support 'mod' as an user-defined function
            smt_command = re.sub(r'mod', 'mod_sharp_', smt_command)

            # Vampire does not support 'mod' as an user-defined function
            smt_command = re.sub(r'mod', 'mod_sharp_', smt_command)

            # Vampire does not support the interpreted index notation '(_ a b)';
            #  we need to rewrite these as 'a_b'
            m = re.match(r'\(_ (.*?) (.*?)\)', smt_command)
            if m is not None and len(m.groups()) == 2:
                smt_command = re.sub(m.group(0), m.group(1) + '_' + m.group(2), smt_command)

            # Vampire does not support the overloaded 'bv2int' symbol;
            #  therefore, we infer and declare the required (bit-width specific) sorts
            #  and declare the corresponding 'bv2int_BW' functions where BV is the bit width
            #  that comes from explicit type name in the variable declaration context.
            #  Example: (forall ((x BV_32)) ( (P (bv2int x)) ))
            #   inferred requirements:
            #   1. (declare-sort BV_32 0)
            #   2. (declare-fun bv2int_32 (BV_32) Int)
            #
            # Warning!!! This method will fail if
            #  a) the argument of 'bv2int' is non-atomic
            #  b) the argument of 'bv2int' is free in 'smt_command' (e.g. if it's a constant)
            #  c) there is shadowing of variable names within 'smt_command'
            rewrite_map = {}
            bv2int_args = re.findall('\(bv2int (.*?)\)', smt_command)
            for bv2int_arg in bv2int_args:
                m = re.match('\(%s (.*?)\)' % bv2int_arg, smt_command)
                assert len(m.groups()) == 1
                bv_type_name = m.group(1)
                bv_width = int(bv_type_name[bv_type_name.rindex('_')+1:])
                bv_convertion_fun_name = '(bv2int_%d)' % str(bv_width)
                rewrite_map['\(bv2int %s\)' % bv2int_arg] = bv_convertion_fun_name

                bv_sort_declaration = '(declare-sort %s 0)' % bv_type_name
                if bv_sort_declaration not in new_smt_commands:
                    new_smt_commands.append(bv_sort_declaration)
                bv_fun_declaration = '(declare-fun %s (%s) Int)' % (bv_convertion_fun_name, bv_type_name)
                if bv_fun_declaration not in new_smt_commands:
                    new_smt_commands.append(bv_fun_declaration)

            for rewrite_from, rewrite_to in rewrite_map.items():
                smt_command = re.sub(rewrite_from, rewrite_to, smt_command)

            if 'RoundingMode' in smt_command and rounding_mode_declaration not in new_smt_commands:
                new_smt_commands.append(rounding_mode_declaration)
            # Vampire does not support String as a built-in type
            if 'String' in smt_command and string_sort_declaration not in new_smt_commands:
                new_smt_commands.append(string_sort_declaration)
            new_smt_commands.append(smt_command)

        return new_smt_commands

    def __cvc_commands(self):

        """
        This method creates SMT commands that are logically equivalent to [[self.all_commands]]
        but they can be parsed by CVC4/cvc5.

        :return: A list of SMT commands that can be parsed by Vampire
        """

        new_smt_commands = []

        for smt_command in self.all_commands:
            # CVC4/cvc5 does not support 'mod' as an user-defined function
            smt_command = re.sub(r'mod', 'mod_at_sharp_at_', smt_command)

            # CVC4/cvc5 does not support symbols starting with '@'
            smt_command = re.sub(r'@', '_at_', smt_command)

            new_smt_commands.append(smt_command)

        return new_smt_commands

    def is_multiquery(self):
        return len(self.get_checksat_commands()) > 1

    def __rewrite_smt_commands(self):
        rewrite_map = {
            #'(reset)': '(reset-assertions)'
        }

        new_smt_commands = []
        for command in self.all_commands:
            if command in rewrite_map:
                new_smt_commands.append(rewrite_map[command])
            else:
                new_smt_commands.append(command)

        self.all_commands = new_smt_commands

    def rewrite_builtin_symbols(self):

        rewrite_map = {
            'iff': '=',
            'implies': '=>'
        }

        new_smt_commands = []
        for command in self.all_commands:
            # Apply simple rewrite rules
            for old_symbol, new_symbol in rewrite_map.items():
                command = create_substituted_string(self.TOKEN_ALPHABET, old_symbol, new_symbol, command)

            # Rewrite (as FUN_DECL SORT) as FUN_DECL
            m = re.findall(self.SMT_BUILTIN_AS_PATTERN, command)
            for group in m:
                assert len(group) == 3
                whole = re.escape(group[0])
                lhs = group[1]
                command = re.sub(whole, lhs, command)

            new_smt_commands.append(command)

        self.all_commands = new_smt_commands
