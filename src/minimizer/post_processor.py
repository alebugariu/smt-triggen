###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2022 ETH Zurich.
###############################################################################

import shutil
from collections import OrderedDict
from os.path import join, basename

from src.minimizer.minimizer import Minimizer
from src.utils.file import add_fname_suffix, ensure_dir_structure, delete_tmp_files


class PostProcessor(Minimizer):

    DBG_PREFIX = "PostProcessor >> "

    RANDOM_SEED = OrderedDict({
        "R.924-981-8":
            ["(set-option :sat.random_seed 924)", "(set-option :smt.random_seed 981)", "(set-option :nlsat.seed 8)"],
        "R.405-849-277":
            ["(set-option :sat.random_seed 405)", "(set-option :smt.random_seed 849)", "(set-option :nlsat.seed 277)"],
        "R.395-482-414":
            ["(set-option :sat.random_seed 395)", "(set-option :smt.random_seed 482)", "(set-option :nlsat.seed 414)"],
        "R.604-787-142":
            ["(set-option :sat.random_seed 604)", "(set-option :smt.random_seed 787)", "(set-option :nlsat.seed 142)"],
        "R.44-170-122":
            ["(set-option :sat.random_seed 44)", "(set-option :smt.random_seed 170)", "(set-option :nlsat.seed 122)"]
    })

    def __init__(self, fin_name, solver, target_dir, timeout_per_file, debug_mode=False):

        tmp_dir = join(target_dir, 'tmp')
        ensure_dir_structure(tmp_dir)
        file_name = basename(fin_name)
        output_file_name = join(tmp_dir, file_name)

        super().__init__(fin_name, solver=solver, fout_name=output_file_name, target_dir=target_dir, validate=False,
                         standardize=False, remove_duplicates=False, write_result_to_disk=False)

        found_random_seed_cmd = False
        self.commands_before_random_seeds = []
        self.commands_after_random_seeds = []
        for cmd in self.all_commands:
            if 'set-option' in cmd and 'seed' in cmd:
                found_random_seed_cmd = True
                continue
            if not found_random_seed_cmd:
                self.commands_before_random_seeds.append(cmd)
            else:
                self.commands_after_random_seeds.append(cmd)

        stable = self.run_with_different_seeds(timeout_per_file)
        if stable:  # if the input passed all tests, save it as artifact
            new_file = add_fname_suffix(file_name, "post_stable")
            shutil.copy(fin_name, join(target_dir, new_file))
            self.print_dbg(" $$$$$$$$$ SUCCESS: File `%s` has passed all post processing tests $$$$$$$$$ " % fin_name)

        # delete tmp files
        if not debug_mode:
            delete_tmp_files(tmp_dir)

    def run_with_different_seeds(self, timeout_per_file) -> bool:
        for label, options_random_seeds in self.RANDOM_SEED.items():
            test_name = add_fname_suffix(self.fout_name, label)
            minimizer = Minimizer(self.fin_name, solver=self.solver, fout_name=test_name,
                                  target_dir=self.target_dir, validate=False, standardize=False, mbqi=False,
                                  remove_duplicates=False, ensure_incremental_mode=False, write_result_to_disk=False)
            minimizer.all_commands = self.commands_before_random_seeds + options_random_seeds + \
                                     self.commands_after_random_seeds

            status = minimizer.run(timeout=timeout_per_file, seed=None, with_timing=False, quiet=False,
                                   save_artifact_lbd=lambda x: False)
            if status != "unsat":
                self.print_dbg("%s FAILED: outcome changed from unsat to %s" % (test_name, status))
                return False
        return True
