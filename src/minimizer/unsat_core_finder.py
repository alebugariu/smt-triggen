###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2024 ETH Zurich.
###############################################################################

import re
from os.path import join, basename

from src.minimizer.minimizer import Minimizer
from src.utils.enums import SMTSolver
from src.utils.file import add_fname_suffix, ensure_dir_structure, delete_tmp_files

"""
    Extracts the unsat core using an external SMT solver and a configurable algorithm (MBQI or E-matching).
"""


class UnsatCoreFinder(Minimizer):

    DBG_PREFIX = "UnsatCoreFinder >> "

    def __init__(self, fin_name, target_dir, solver, timeout_per_file, mbqi, enum_inst=False, debug_mode=False):

        tmp_dir = join(target_dir, 'tmp')
        ensure_dir_structure(tmp_dir)
        file_name = basename(fin_name)
        output_file_name = join(tmp_dir, file_name)

        super().__init__(fin_name, fout_name=output_file_name, target_dir=target_dir, solver=solver, validate=False,
                         standardize=False, remove_duplicates=False, mbqi=mbqi, write_result_to_disk=False)

        self.remove_useless_commands()  # remove (get-info :reason-unknown)

        actual_status, unsat_core = self.run(new_suffix='ucore', solver=solver, enum_inst=enum_inst,
                                             with_unsat_core=True, timeout=timeout_per_file)

        if actual_status != 'unsat':
            self.print_dbg("$$$ The solver returned %s, skipping this example." % actual_status)
            # delete tmp files
            if not debug_mode:
                delete_tmp_files(tmp_dir)
            return
        elif len(unsat_core) == 0:
            self.print_dbg("$$$ The solver could not construct the unsat core, skipping this example.")
            # delete tmp files
            if not debug_mode:
                delete_tmp_files(tmp_dir)
            return

        self.print_dbg('Extracted unsat core: (%s)' % ', '.join(unsat_core))

        # Remove assertions that are not in the minimal unsat core
        to_be_deleted = []
        unsat_core_axioms = []
        original_assertions = self.get_assertions()

        if len(unsat_core) == len(original_assertions):  # nothing has to be deleted
            unsat_core_axioms = original_assertions
        else:
            for cmd in original_assertions:
                matches = re.match(self.ASSERTION_NAME_PATTERN, cmd)
                if matches:
                    # This must be a named assertion
                    name = matches.group(1)
                    cvc = solver is SMTSolver.CVC4 or solver is SMTSolver.CVC5
                    if name in unsat_core or (cvc and re.sub(r'@', '_at_', name) in unsat_core):
                        unsat_core_axioms.append(cmd)
                    else:
                        to_be_deleted.append(cmd)

        assert len(unsat_core_axioms) == len(unsat_core)
        assert set(unsat_core_axioms + to_be_deleted) == set(original_assertions)

        for cmd in to_be_deleted:
            self.all_commands.remove(cmd)

        self.all_commands = self.get_options_info_logic() + self.get_declarations() + unsat_core_axioms
        self.remove_unsat_core_commands()

        if mbqi:  # the unsat core was extracted with MBQI, but we turn it off in the resulting file
            self.set_mbqi(False)

        unsat_core_file = add_fname_suffix(join(target_dir, file_name), 'ucore')
        self.write_lines_to_file(self.all_commands, unsat_core_file)

        # delete tmp files
        if not debug_mode:
            delete_tmp_files(tmp_dir)
