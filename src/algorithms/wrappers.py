###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2022 ETH Zurich.
###############################################################################

from pysmt.shortcuts import get_env
from src.session.session import Session


class AnnotationsPreserver:
    """
    This class wraps the save/ restore operations for recovering annotations
     after, e.g., parsing them, rolling the possible side effects back.
    """
    def __init__(self, annotations: 'Annotations', keep_changes):
        self.annotation = annotations
        self.keep_changes = keep_changes
        self.original_annotations = annotations._annotations.copy()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.keep_changes:
            self.annotation._annotations = self.original_annotations


class PhaseSwitcher:
    def __init__(self, algo: 'IndividualAxiomTester', new_phase_name: str, next_phase_name=None):
        if next_phase_name:
            self.saved_phase_name = next_phase_name
        else:
            self.saved_phase_name = algo.phase
        algo.phase = new_phase_name
        self.algo = algo

    def __enter__(self):
        self.algo.print_dbg("Starting phase `%s`..." % self.algo.phase)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.algo.print_dbg("Finished phase `%s`." % self.algo.phase)
        self.algo.phase = self.saved_phase_name


class StackFrame:
    def __init__(self, session: Session, comment: str):
        self.session = session
        self.comment = comment

    def __enter__(self):
        self.session.push(comment=self.comment)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.pop(comment=self.comment)


class VariableNamesPreserver:
    def __init__(self):
        self.env = get_env()
        self._old_state = self.env.unique_names_guarantee

    def __enter__(self):
        self.env.unique_names_guarantee = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.env.unique_names_guarantee = self._old_state
