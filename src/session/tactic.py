###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2022 ETH Zurich.
###############################################################################

import io
from typing import Tuple, List

from src.minimizer.minimizer import Minimizer
from src.session.exceptions import SmtTimeout, SkolemizationTimeout, SimplifierTimeout
from src.session.session import Session
from src.utils.preamble import PRETTY_PRINTER_NOWRAP, STANDARD_PREAMBLE


class Tactic(Session):

    DBG_PREFIX = "Tactic"

    def __init__(self,
                 name: str,
                 preamble=STANDARD_PREAMBLE,
                 cmd=('z3', '-smt2', '-t:4000', '-in'),
                 hard_timeout_per_query=4.0,
                 memory_limit_mb=1000,
                 logs_path=None):

        preamble = list(preamble) + PRETTY_PRINTER_NOWRAP
        super().__init__(name=name, preamble=preamble, cmd=cmd, memory_limit_mb=memory_limit_mb,
                         hard_timeout_per_query=hard_timeout_per_query, logs_path=logs_path)

    def skolemize(self, formula: str, approach="nnf", post_simplify=True) -> \
            Tuple[List[Tuple[str, Tuple[str], str]], List[str]]:

        self.push("Begin Skolemization...")
        self.add_constraint(formula)

        tactics = ["(using-params %s :mode quantifiers)" % approach]

        if post_simplify:
            tactics.append("simplify")

        query = "(apply (then %s) :print false :print_benchmark true)" % " ".join(tactics)

        try:
            res = self._query(query)
        except SmtTimeout as e:
            raise SkolemizationTimeout(e.session, e.duration_sec)
        finally:
            self.pop("End Skolemization.")

        with io.StringIO(" ".join(res)) as stream:
            mzr = Minimizer(in_stream=stream, validate=False, treat_special_tokens=False,
                            write_result_to_disk=False, remove_duplicates=True)
            sk_vars = mzr.get_fun_const_decl_triplets()
            sk_formulas = sorted((mzr.get_assertion_body(a) for a in mzr.get_assertions()))

        return sk_vars, sk_formulas

    # Note: this method is currently not used, we use FNode.simplify() instead
    def simplify(self, formula: str) -> List[str]:

        self.push("Begin simplification...")
        self.add_constraint(formula)

        query = "(apply ctx-solver-simplify)"

        try:
            res = self._query(query)
        except SmtTimeout as e:
            raise SimplifierTimeout(e.session, e.duration_sec)
        finally:
            self.pop("End simplification.")

        with io.StringIO(" ".join(res)) as stream:
            mzr = Minimizer(in_stream=stream, validate=False, write_result_to_disk=False, remove_duplicates=True)
            simple_formulas = sorted((mzr.get_assertion_body(a) for a in mzr.get_assertions()))

        return simple_formulas
