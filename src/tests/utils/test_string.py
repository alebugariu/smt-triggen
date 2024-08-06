###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2024 ETH Zurich.
###############################################################################

import exrex

from src.minimizer.minimizer import Minimizer
from src.tests.base_unit_test import BaseUnitTest

alph = ''.join(exrex.generate('[' + Minimizer.TOKEN_SYMBOL_PATTERN + ']'))


class TestStringUtils(BaseUnitTest):

    test_pool = [
        (False, 'ab', 'abc'),
        (True,  'ab', 'ab c'),
        (True,  'ab', '|ab|c'),
        (False, 'ab', 'qabc'),
        (True,  'ab', 'q|ab c'),
        (True,  'ab', 'q ab c'),
        (False, 'ab', 'qab abc'),
        (True,  'ab', 'qab abc |ab| wqe'),
        (True,  '@query', '(! (assert true) :named @query)'),
    ]

    def test__is_token_in_string(self):
        """
        Test that it works as expected.
        """

        from src.utils.string import is_token_in_string

        for answer, tk, string in self.test_pool:
            self.assertTrue(is_token_in_string(alph, tk=tk, a=string) == answer,
                            msg='found unexpected result for token ' + tk + ' in ' + string)

    def test_alternative_implementations_of__is_token_in_string(self):
        """
        Test that the Regex implementation of `is_token_in_string` behaves the same way as the fast implementation.
        """

        import re
        from src.utils.string import is_token_in_string

        def is_token_in_string__regex(alphabet, tk, a):
            separator = '[^' + alphabet + ']'
            escaped_tk = re.escape(tk)  # allows matching declarations with names, e.g., `|Hello|`.
            return bool(re.match(r'(^|.*' + separator + ')(' + escaped_tk + ')(' + separator + '|$).*', a))

        for _, tk, string in self.test_pool:
            fast_result = is_token_in_string(alph, tk=tk, a=string)
            regex_result = is_token_in_string__regex(alph, tk=tk, a=string)
            self.assertEqual(fast_result, regex_result,
                             msg='different results from is_token_in_string and is_token_in_string__regex for token ' +
                                 tk + ' in ' + string)

    def test__create_substituted_string(self):
        """
        Test that it works as expected based on the presence of valid substitutions.
        """

        import re
        from src.utils.string import create_substituted_string

        replacement_token = 'HELLO'
        for answer, tk, string in self.test_pool:
            assert replacement_token not in string # sanity check for the test itself
            new_string = create_substituted_string(alph, tk, replacement_token, string)
            if answer:
                # The substitution should have happened
                found_instances = re.finditer(replacement_token, new_string)
                self.assertTrue(len(list(found_instances)) > 0)
            else:
                # No substitutions should have happened
                self.assertTrue(new_string, string)

    def test__create_substituted_string__deep(self):
        """
        Test that it works as expected based on question/answer pairs.
        """

        from src.utils.string import create_substituted_string

        substitutions_oracle = [
            ('Hello', 'x42', '(assert Hello () World)', '(assert x42 () World)'), # SMT-like string
            ('cd', 'HELLO', 'abcdefg', 'abcdefg'),                                # no change ('cd' is not a token)
            ('bcdef', '_', 'a bcdef g', 'a _ g'),                                 # white spaces as token delimiters
            ('cc', '333', 'ccbb|cc|ddee cc gg', 'ccbb|333|ddee 333 gg'),          # mixed delimiters
        ]

        for tk, new_tk, string, answer in substitutions_oracle:
            new_string = create_substituted_string(alph, tk, new_tk, string)
            self.assertEqual(answer, new_string)
