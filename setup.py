###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2024 ETH Zurich.
###############################################################################

from setuptools import setup, find_packages

setup(name='axioms-explosion',
      version='0.1',
      description='Synthesizing triggering terms for E-matching',
      author='Ale & Arshavir',
      install_requires=[
            'scikit-learn',
            'sortedcontainers',
            'datasketch',
            'six',
            'timeout-decorator==0.4.1',
            'func_timeout',
            'psutil'
      ],
      packages=find_packages(),
)
