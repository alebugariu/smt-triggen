###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2022 ETH Zurich.
###############################################################################

Algorithms = ["PatternAugmenter",
              "GroupsAxiomTester"]


class Optset(object):

    def __init__(self, algorithm="GroupsAxiomTester",
                 debug=False,
                 max_different_models: int = 4,
                 similarity_threshold=0.3,
                 max_depth=2,
                 max_axiom_frequency=1,
                 max_distributivity=1,
                 max_formulas: int = 100,
                 multiple_dummies=False,
                 batch_size=64,
                 type_constraints=False,
                 top_level=True,
                 diverse_models=True,
                 disable_preprocessor=True,
                 keep_duplicate_assertions=False,
                 timeout: int or None = 600):

        self.algorithm = algorithm
        self.debug = debug
        self.max_different_models = max_different_models
        self.similarity_threshold = similarity_threshold
        self.max_depth = max_depth
        self.max_axiom_frequency = max_axiom_frequency
        self.max_distributivity = max_distributivity
        self.max_formulas = max_formulas
        self.multiple_dummies = multiple_dummies
        self.batch_size = batch_size
        self.type_constraints = type_constraints
        self.top_level = top_level
        self.diverse_models = diverse_models
        self.disable_preprocessor = disable_preprocessor
        self.keep_duplicate_assertions = keep_duplicate_assertions
        self.timeout = timeout

    def set(self, args_dict):
        for key, val in args_dict.items():
            if key != "location":
                setattr(self, key, val)

        return self

    def __repr__(self):
        return " ".join(["%s=%s" % ("--" + str(key) if key != "algorithm" else str(key), str(val))
                                          for key, val in vars(self).items()])

    def __str__(self):
        return self.__repr__()

