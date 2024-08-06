###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2024 ETH Zurich.
###############################################################################

if 'DEBUG_MODE' not in globals():
    OPTIMIZE_TAIL_CALLS = False
else:
    OPTIMIZE_TAIL_CALLS = not DEBUG_MODE

from src.utils.optimizations import tail_call_opt


def assert_is_token_valid(alphabet: str, tk: str):
    assert len(tk) > 0 and set(tk).issubset(set(alphabet)), \
           'Token `%s` is not valid in alphabet `%s`' % (tk, alphabet)


# TODO: this method is not practical, for many examples it causes:
# TODO: "RecursionError: maximum recursion depth exceeded while calling a Python object"
@tail_call_opt(condition=OPTIMIZE_TAIL_CALLS)
def is_token_in_string(alphabet: str, tk: str, a: str, start=None, end=None) -> bool:
    """ Check that the token is used in this string """

    assert_is_token_valid(alphabet, tk)

    find_res = a.find(tk, start, end)
    if find_res == -1:
        # This means the token is not a substring of a[start:end]
        return False
    else:
        # This means the token is an infix of a
        def is_good_delimiter(d) -> bool:
            if not d:
                return True
            else:
                return d not in alphabet

        left_delimiter, right_delimiter = None, None
        if find_res > 0:
            left_delimiter = a[find_res - 1]
        if find_res + len(tk) < len(a):
            right_delimiter = a[find_res + len(tk)]
        if is_good_delimiter(left_delimiter) and is_good_delimiter(right_delimiter):
            # Valid delimiters: the token is well-separated
            return True
        else:
            # Not a valid delimiter: search in the remainder of the string:
            #  a = ['a','b','c','a','b','c','d']
            # tk = ['a','b','c']
            #  a'=         ['c','a','b','c','d']
            middle_of_leftover = int((len(a) - find_res + len(tk) - 1) / 2)  # optimization
            while middle_of_leftover < len(a) and not is_good_delimiter(a[middle_of_leftover]):
                middle_of_leftover += 1  # Avoid splitting tokens between two sub-strings

            return is_token_in_string(alphabet, tk, a[find_res + len(tk) - 1:middle_of_leftover], start=1) or \
                   is_token_in_string(alphabet, tk, a[middle_of_leftover:], start=1)


@tail_call_opt(condition=OPTIMIZE_TAIL_CALLS)
def create_substituted_string(alphabet: str, tk: str, new_term: str, a: str, start=None, end=None) -> str:
    """
    Create a new string by substituting all occurrences of `tk` with `new_term`
     while respecting the delimiters of `alphabet`.
    """

    assert_is_token_valid(alphabet, tk)

    find_res = a.find(tk, start, end)
    if find_res == -1:
        # This means the token is not a substring of a[start:end]
        return a
    else:
        # This means the token is an infix of a
        def is_good_delimiter(d) -> bool:
            if not d:
                return True
            else:
                return d not in alphabet

        left_delimiter, right_delimiter = None, None
        if find_res > 0:
            left_delimiter = a[find_res - 1]
        if find_res + len(tk) < len(a):
            right_delimiter = a[find_res + len(tk)]
        if is_good_delimiter(left_delimiter) and is_good_delimiter(right_delimiter):
            # Valid delimiters: the token is well-separated
            new_a = a[:find_res] + new_term + a[find_res + len(tk):]
            return create_substituted_string(alphabet, tk, new_term, new_a, start=find_res+len(new_term))
        else:
            # Not a valid delimiter: search in the remainder of the string:
            #  a = ['a','b','c','a','b','c','d']
            # tk = ['a','b','c']
            #  a'=         ['c','a','b','c','d']
            return create_substituted_string(alphabet, tk, new_term, a, start=find_res+len(tk))


def strminus(orig: str, sub: str) -> str:
    find_res = orig.find(sub)
    if find_res == -1:
        # This means the token is not a substring of a[start:end]
        return orig
    else:
        return orig[:find_res] + orig[find_res+len(sub):]
