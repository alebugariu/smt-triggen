###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2024 ETH Zurich.
###############################################################################

STANDARD_PREAMBLE = [
    '(set-option :smt.auto-config false)',
    '(set-option :smt.mbqi false)',
    '(set-option :sat.random_seed 488)',  # 100% random
    '(set-option :smt.random_seed 599)',
    '(set-option :nlsat.seed 611)',
    '(set-option :memory_max_size 6000)',  # In MB
]

STANDARD_MBQI_PREAMBLE = [
    '(set-option :smt.auto-config false)',
    '(set-option :smt.mbqi true)',
    '(set-option :sat.random_seed 488)',  # 100% random
    '(set-option :smt.random_seed 599)',
    '(set-option :nlsat.seed 611)',
    '(set-option :memory_max_size 6000)',  # In MB
]

EMATCHING = [
    '(set-option :smt.auto-config false)',
    '(set-option :smt.mbqi false)',
]

SEED_MEMORY_OPTIONS = [
    '(set-option :sat.random_seed 488)',  # 100% random
    '(set-option :smt.random_seed 599)',
    '(set-option :nlsat.seed 611)',
    '(set-option :memory_max_size 6000)',  # In MB
]

PRETTY_PRINTER_NOWRAP = [
    '(set-option :pp.max_depth 500000)',
    '(set-option :pp.max_ribbon 500000)',
    '(set-option :pp.max_width 500000)',
    '(set-option :pp.single_line true)',
    '(set-option :pp.min_alias_size 500000)',
]