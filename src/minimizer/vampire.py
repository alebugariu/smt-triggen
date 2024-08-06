###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2024 ETH Zurich.
###############################################################################

from os.path import join, basename

from src.minimizer.minimizer import Minimizer
from src.utils.enums import SMTSolver
from src.utils.file import ensure_dir_structure, delete_tmp_files


class VampireRunner(Minimizer):

    DBG_PREFIX = "VampireRunner >> "

    def __init__(self, fin_name, target_dir, debug_mode=True):

        tmp_dir = join(target_dir, 'tmp')
        ensure_dir_structure(tmp_dir)
        file_name = basename(fin_name)
        output_file_name = join(tmp_dir, file_name)

        self.print_dbg('Saving temporary files into %s' % tmp_dir, quite=not debug_mode)

        super().__init__(fin_name, solver=SMTSolver.VAMPIRE, fout_name=output_file_name,
                         target_dir=target_dir, write_result_to_disk=debug_mode, validate=False)

        self.print_dbg('Saving Vampire-ready files into %s' % target_dir, quite=not debug_mode)

        self.run(solver=SMTSolver.VAMPIRE, timeout=600,
                 new_suffix='vamp', accumulate=False,
                 art_tag=lambda x: x)

        # delete tmp files
        if not debug_mode:
            delete_tmp_files(tmp_dir)
