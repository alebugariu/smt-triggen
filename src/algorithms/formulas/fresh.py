###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2022 ETH Zurich.
###############################################################################

from pysmt.fnode import FNode


class FreshVariable:

    def __init__(self, var_name: str, var_type: str, var_node: FNode, declaration: str):

        self.var_name = var_name
        self.var_type = var_type
        self.var_node = var_node
        self.declaration = declaration

    def __str__(self):
        return self.var_name

