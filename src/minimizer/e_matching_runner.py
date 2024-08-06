###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2024 ETH Zurich.
###############################################################################

from os.path import join, basename
from src.minimizer.minimizer import Minimizer
from src.utils.file import ensure_dir_structure, delete_tmp_files


class EmatchingRunner:

    DBG_PREFIX = "EMatchingRunner >> "

    def __init__(self, fin_name, solver, target_dir, timeout_per_file, debug_mode=False):

        tmp_dir = join(target_dir, 'tmp')
        ensure_dir_structure(tmp_dir)
        file_name = basename(fin_name)
        output_file_name = join(tmp_dir, file_name)

        self.m = Minimizer(fin_name, solver=solver, fout_name=output_file_name, target_dir=target_dir,
                           validate=False, standardize=False, remove_duplicates=False, mbqi=False,
                           ensure_incremental_mode=False, write_result_to_disk=False)

        self.m.run(timeout=timeout_per_file, seed=None, with_timing=False, quiet=False, get_reason_unknown=True,
                   save_artifact_lbd=self.smt_status_handler, art_tag='incomplete-quant')

        # delete tmp files
        if not debug_mode:
            delete_tmp_files(tmp_dir)

    def smt_status_handler(self, status, reason):
        assert self.m is not None, "the Minimizer should be initialized at this point"
        if status == 'unsat':
            self.m.print_dbg("$$$ Refuted via E-matching, skipping this example.")
            return False
        if status == 'sat':
            self.m.print_dbg("$$$ E-matching returned sat, skipping this example.")
            return False
        if status == "unknown":
            if "incomplete quantifiers" in reason:
                return True
            self.m.print_dbg("$$$ E-matching returned unknown for a different reason (%s), skipping this example." %
                             reason)
            return False
        self.m.print_dbg('E-matching gave up.')
        return False
