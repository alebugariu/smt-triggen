###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2022 ETH Zurich.
###############################################################################

import sys

from src.session.session import Session
from src.session.tactic import Tactic


def steps(s: Session):
    s.declare_const("A", "Int")
    s.declare_const("B", "Int")
    s.check_sat()
    s.check_sat()
    s.check_sat()
    res2 = s.get_values(['A', 'B'])
    s.push("test")
    s.add_constraint("(= B -42)")
    res1, attach = s.check_sat()
    if res1 == "sat":
        res2 = s.get_values(['A', 'B'])
    else:
        s.pop("test")

    return res1, res2


def skolemize(s: Tactic):
    s.declare_fun('f', ('Int',), 'Bool')
    s.declare_fun('g', ('Int', 'Int'), 'Bool')
    axiom = '(not (forall ((x Int) (y Int)) (! (or (g x y) (forall ((z Int)) (! (or (g z x) (g y z) (f z)) :pattern ((f z)) ) )) :pattern ((g x y)) )))'
    decls, asserts = s.skolemize(axiom)
    return decls, asserts


def test_session():
    with Session(name="Test Session") as s:
        res = steps(s)
        print("res: %s" % str(res))
        print("session: \n%s" % str(s))


def test_skolem():
    with Tactic(name="Test Skolemizer") as s:
        res1, res2 = skolemize(s)
        res1 = list(map(lambda x: "(fun:%s, args:%s, typ:%s" % (x[0], x[1], x[2]), res1))
        res = "\n".join(res1+res2)
        print("res: \n%s" % res)
        print("session: \n%s" % str(s))


def main(args):
    test_session()
    test_skolem()


if __name__ == "__main__":
    main(sys.argv)
