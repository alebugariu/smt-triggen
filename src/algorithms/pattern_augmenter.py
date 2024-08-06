###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2024 ETH Zurich.
###############################################################################

from src.algorithms.exceptions import PatternException
from src.algorithms.formulas.axiom import Axiom
from src.frontend.optset import Optset
from src.session.session import Session
from src.algorithms.singletons import IndividualAxiomTester
from src.utils.preamble import STANDARD_PREAMBLE
from src.utils.smt import smt2

# ATTENTION:
#  Do not remove the following imports:
#  They synchronize global variables across different modules
from src.session import debug_mode


class PatternAugmenter(IndividualAxiomTester):

    DBG_PREFIX = "PatternAugmenter"

    def __init__(self, filename, target_dir, tmp_file_dir, logs_dir=None, debug=False):

        debug_mode.flag = debug

        opt = Optset(disable_preprocessor=False)
        super().__init__(filename=filename, tmp_file_dir=tmp_file_dir, logs_dir=logs_dir, opt=opt)
        self.raise_on_axioms_with_incomplete_patterns = True

        if target_dir:
            self.m.target_dir = target_dir

        self.m.all_commands.append("(check-sat)")

        self.num_original_axioms = 0
        self.num_good_axioms = 0

    def is_good_axiom(self, ax: Axiom):

        for quant in ax.quantifiers:
            for qvar in quant.quantifier_vars():
                covered_by_trigger = False

                for multi_trig in quant.quantifier_patterns():
                    for trig in multi_trig:
                        if qvar in trig.all_args():
                            covered_by_trigger = True
                            break
                    if covered_by_trigger:
                        break
                if not covered_by_trigger:
                    self.print_dbg("Removing bad axiom `%s`" % smt2(ax))
                    self.print_dbg("  qvar %s is not covered by any trigger" % smt2(qvar))
                    return False

        return True

    def run(self):
        """
        Skolemize and augment with missing patterns (if such patterns can be provided by Z3)
        """

        orig_preamble = STANDARD_PREAMBLE + \
                        self.m.get_setinfo() + \
                        self.m.get_setlogic()

        with Session(name="Session", preamble=self.m.get_setlogic() + self.preamble,
                        logs_path=self.models_logs_path) as session:

            try:
                augmented_axioms = self.skolemize_all(session)
                self.num_original_axioms = len(augmented_axioms)
            except PatternException:
                self.print_dbg(" skipping `%s` (some axioms could not be augmented with patterns)" % self.filename)
                return False

            augmented_declarations = list(filter(lambda x: x.startswith("(declare-") or x.startswith("(define-"),
                                                 session.get_z3_log()))

        # Filter the axioms and update Minimizer representation
        def ax_to_str(ax: Axiom) -> str:
            axiom_str = "(assert (! %s :named %s))" % (smt2(ax.axiom_node), ax.axiom_id)
            if self.is_good_axiom(ax):
                self.num_good_axioms += 1
                return axiom_str
            else:
                return "; no suitable triggers found for %s" % axiom_str

        # Update assertions
        assertions = list(map(ax_to_str, augmented_axioms))

        if self.num_original_axioms == self.num_good_axioms:
            preamble = orig_preamble
        else:
            # Update the preamble, in particular, the ground truth
            status = "unknown"
            preamble = []
            for cmd in orig_preamble:
                if self.m.is_setinfo(cmd):
                    slot, text = self.m.parse_setinfo(cmd)
                    if slot == "status":
                        # some axioms were removed; ground truth should be reset to UNKNOWN
                        status = "unknown"
                    preamble.append("(set-info :status %s)" % status)
                else:
                    preamble.append(cmd)

        self.m.all_commands = preamble + augmented_declarations + assertions

        # Save artifact
        if self.num_original_axioms == self.num_good_axioms:
            fname_tag = "aug-gt_%s-full" % "unsat"
        else:
            fname_tag = "aug-gt_%s-extr" % "unknown"
        new_file = self.m.save_to_disk(new_suffix=fname_tag, as_artifact=True, as_out_file=False)
        self.print_dbg(" $$$$$$$$$$ SUCCESS! Original file: `%s`" % self.filename)
        self.print_dbg(" $$$$$$$$$$ file augmented with patterns: `%s`" % new_file)
        self.print_dbg(" $$$$$$$$$$ %1f%% axioms survived (%d/%d)"
                       % (100.0 * self.num_good_axioms/self.num_original_axioms,
                          self.num_good_axioms, self.num_original_axioms))

        return self.num_original_axioms == self.num_good_axioms
