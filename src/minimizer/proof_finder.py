###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2024 ETH Zurich.
###############################################################################

from os.path import join, basename

from src.minimizer.minimizer import Minimizer
from src.utils.file import add_fname_suffix, ensure_dir_structure, delete_tmp_files

"""
    Construct the proof using an external SMT solver and a configurable algorithm (MBQI or E-matching).
"""


class ProofFinder(Minimizer):

    DBG_PREFIX = "ProofFinder >> "

    def __init__(self, fin_name, target_dir, solver, timeout_per_file, mbqi, enum_inst=False, debug_mode=False):

        tmp_dir = join(target_dir, 'tmp')
        ensure_dir_structure(tmp_dir)
        file_name = basename(fin_name)
        output_file_name = join(tmp_dir, file_name)

        super().__init__(fin_name, fout_name=output_file_name, target_dir=target_dir, solver=solver, validate=False,
                         standardize=False, remove_duplicates=False, mbqi=mbqi, write_result_to_disk=False)

        self.remove_useless_commands()  # remove (get-info :reason-unknown)

        actual_status, proof = self.run(new_suffix='proof', solver=solver, enum_inst=enum_inst,
                                        with_proof=True, save_artifact_lbd=self.save_artifact,
                                        timeout=timeout_per_file)

        if actual_status != 'unsat':
            self.print_dbg("$$$ The solver returned %s, skipping this example." % actual_status)
            # delete tmp files
            if not debug_mode:
                delete_tmp_files(tmp_dir)
            return
        elif len(proof) == 0:
            self.print_dbg("$$$ The solver could not construct the proof, skipping this example.")
            # delete tmp files
            if not debug_mode:
                delete_tmp_files(tmp_dir)
            return

        self.print_dbg('Proof: (%s)' % ', '.join(proof))
        self.remove_proof_commands()

        if mbqi:  # the proof was extracted with MBQI, but we turn it off in the resulting file
            self.set_mbqi(False)

        self.all_commands.append('\n')
        for line in proof:
            self.all_commands.append(';%s' % line)
        file_with_proof = add_fname_suffix(join(target_dir, file_name), 'proof')
        self.write_lines_to_file(self.all_commands, file_with_proof)

        # delete tmp files
        if not debug_mode:
            delete_tmp_files(tmp_dir)

    @staticmethod
    def save_artifact(status, proof):
        return status == 'unsat' and len(proof) > 0
