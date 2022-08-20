###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2022 ETH Zurich.
###############################################################################

import unittest


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class BaseUnitTest(unittest.TestCase):
    currentResult = None

    def run(self, result=None):
        self.currentResult = result
        unittest.TestCase.run(self, result)

    def setUp(self):
        print('- %s: %s' % (self._testMethodName, self._testMethodDoc), end=' ')

    def tearDown(self):
        if self.currentResult.wasSuccessful():
            print(bcolors.OKGREEN + '(passed)' + bcolors.ENDC)
        else:
            print(bcolors.FAIL + '(failed)' + bcolors.ENDC)

    @classmethod
    def setUpClass(cls):
        print(bcolors.BOLD + cls.__name__ + bcolors.ENDC)
