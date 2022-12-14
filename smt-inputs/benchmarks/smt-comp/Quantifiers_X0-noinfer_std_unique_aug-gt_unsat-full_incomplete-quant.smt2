(set-info :smt-lib-version 2.6)
(set-info :source | Boogie/Spec@sharp@ benchmarks. This benchmark was translated by Michal Moskal. |)
(set-info :category "industrial")
(set-info :status unsat)
(set-logic UFLIA)
(set-option :smt.auto-config false)
(set-option :smt.mbqi false)
(set-option :sat.random_seed 488)
(set-option :smt.random_seed 599)
(set-option :nlsat.seed 611)
(set-option :memory_max_size 6000)
(declare-fun InRange (Int Int) Bool)
(declare-sort RegExStr 0)
(declare-sort RMode 0)
(declare-fun o () Int)
(declare-fun isA (Int Int) Int)
(declare-fun int_18446744073709551615 () Int)
(declare-fun isB (Int Int) Int)
(declare-fun Smt.false () Int)
(declare-fun anyEqual (Int Int) Int)
(declare-fun select1 (Int Int) Int)
(declare-fun select2 (Int Int Int) Int)
(declare-fun CONCVARSYM (Int) Int)
(declare-fun divides (Int Int) Int)
(declare-fun intAtMost (Int Int) Int)
(declare-fun subtypes (Int Int) Bool)
(declare-fun store1 (Int Int Int) Int)
(declare-fun store2 (Int Int Int Int) Int)
(declare-fun ka (Int) Int)
(declare-fun kb (Int) Int)
(declare-fun intAtLeast (Int Int) Int)
(declare-fun int_2147483647 () Int)
(declare-fun boolOr (Int Int) Int)
(declare-fun int_m9223372036854775808 () Int)
(declare-fun Smt.true () Int)
(declare-fun int_4294967295 () Int)
(declare-fun start_correct () Int)
(declare-fun boolAnd (Int Int) Int)
(declare-fun boolNot (Int) Int)
(declare-fun T_ () Int)
(declare-fun intLess (Int Int) Int)
(declare-fun intGreater (Int Int) Int)
(declare-fun anyNeq (Int Int) Int)
(declare-fun is (Int Int) Int)
(declare-fun e () Int)
(declare-fun f (Int Int) Int)
(declare-fun int_m2147483648 () Int)
(declare-fun g (Int Int) Int)
(declare-fun h (Int Int) Int)
(declare-fun modulo (Int Int) Int)
(declare-fun boolImplies (Int Int) Int)
(declare-fun boolIff (Int Int) Int)
(declare-fun int_9223372036854775807 () Int)
(assert (! (forall ((?xpy0 Int)(?ypy0 Int)) (! (not (<= (+ ?xpy0 ?ypy0) (f ?xpy0 ?ypy0))) :pattern ((f ?xpy0 ?ypy0)) )) :named A1))
(assert (! (forall ((?xpy1 Int)) (! (= (f ?xpy1 10) 3) :pattern ((f ?xpy1 10)) )) :named A2))
(assert (! (forall ((?xpy2 Int)(?ypy1 Int)) (! (= (g ?xpy2 ?ypy1) 3) :pattern ((g ?xpy2 10) (g ?xpy2 ?ypy1)) )) :named A3))
(assert (! (forall ((?ypy2 Int)) (! (= (h ?ypy2 (h ?ypy2 ?ypy2)) ?ypy2) :pattern ((g ?ypy2 ?ypy2)) :pattern ((h ?ypy2 (h ?ypy2 10))) )) :named A4))
(assert (! (forall ((?opy0 Int)) (! (not (= (isA ?opy0 T_) Smt.true)) :pattern ((isA ?opy0 T_)) )) :named A5))
(assert (! (forall ((?opy1 Int)) (! (not (= (isB ?opy1 T_) Smt.true)) :pattern ((isB ?opy1 T_)) )) :named A6))
(assert (! (forall ((?Apy0 Int)(?ipy0 Int)(?vpy0 Int)) (! (= (select1 (store1 ?Apy0 ?ipy0 ?vpy0) ?ipy0) ?vpy0) :pattern ((store1 ?Apy0 ?ipy0 ?vpy0)) )) :named A7))
(assert (! (forall ((?Apy1 Int)(?ipy1 Int)(?jpy0 Int)(?vpy1 Int)) (! (or (= ?ipy1 ?jpy0) (= (select1 (store1 ?Apy1 ?ipy1 ?vpy1) ?jpy0) (select1 ?Apy1 ?jpy0))) :pattern ((select1 (store1 ?Apy1 ?ipy1 ?vpy1) ?jpy0)) )) :named A8))
(assert (! (forall ((?Apy2 Int)(?opy2 Int)(?fpy0 Int)(?vpy2 Int)) (! (= (select2 (store2 ?Apy2 ?opy2 ?fpy0 ?vpy2) ?opy2 ?fpy0) ?vpy2) :pattern ((store2 ?Apy2 ?opy2 ?fpy0 ?vpy2)) )) :named A9))
(assert (! (forall ((?Apy3 Int)(?opy3 Int)(?fpy1 Int)(?ppy0 Int)(?gpy0 Int)(?vpy3 Int)) (! (or (= ?opy3 ?ppy0) (= (select2 (store2 ?Apy3 ?opy3 ?fpy1 ?vpy3) ?ppy0 ?gpy0) (select2 ?Apy3 ?ppy0 ?gpy0))) :pattern ((select2 (store2 ?Apy3 ?opy3 ?fpy1 ?vpy3) ?ppy0 ?gpy0)) )) :named A10))
(assert (! (forall ((?Apy4 Int)(?opy4 Int)(?fpy2 Int)(?ppy1 Int)(?gpy1 Int)(?vpy4 Int)) (! (or (= ?fpy2 ?gpy1) (= (select2 (store2 ?Apy4 ?opy4 ?fpy2 ?vpy4) ?ppy1 ?gpy1) (select2 ?Apy4 ?ppy1 ?gpy1))) :pattern ((select2 (store2 ?Apy4 ?opy4 ?fpy2 ?vpy4) ?ppy1 ?gpy1)) )) :named A11))
(assert (! (forall ((?xpy3 Int)(?ypy3 Int)) (! (and (or (not (= (boolIff ?xpy3 ?ypy3) Smt.true)) (and (or (not (= ?xpy3 Smt.true)) (= ?ypy3 Smt.true)) (or (= ?xpy3 Smt.true) (not (= ?ypy3 Smt.true))))) (or (= (boolIff ?xpy3 ?ypy3) Smt.true) (and (or (= ?xpy3 Smt.true) (= ?ypy3 Smt.true)) (or (not (= ?xpy3 Smt.true)) (not (= ?ypy3 Smt.true)))))) :pattern ((boolIff ?xpy3 ?ypy3)) )) :named A12))
(assert (! (forall ((?xpy4 Int)(?ypy4 Int)) (! (and (or (= ?ypy4 Smt.true) (not (= ?xpy4 Smt.true)) (not (= (boolImplies ?xpy4 ?ypy4) Smt.true))) (or (= (boolImplies ?xpy4 ?ypy4) Smt.true) (and (= ?xpy4 Smt.true) (not (= ?ypy4 Smt.true))))) :pattern ((boolImplies ?xpy4 ?ypy4)) )) :named A13))
(assert (! (forall ((?xpy5 Int)(?ypy5 Int)) (! (and (or (not (= (boolAnd ?xpy5 ?ypy5) Smt.true)) (and (= ?xpy5 Smt.true) (= ?ypy5 Smt.true))) (or (not (= ?xpy5 Smt.true)) (= (boolAnd ?xpy5 ?ypy5) Smt.true) (not (= ?ypy5 Smt.true)))) :pattern ((boolAnd ?xpy5 ?ypy5)) )) :named A14))
(assert (! (forall ((?xpy6 Int)(?ypy6 Int)) (! (and (or (= ?xpy6 Smt.true) (= ?ypy6 Smt.true) (not (= (boolOr ?xpy6 ?ypy6) Smt.true))) (or (= (boolOr ?xpy6 ?ypy6) Smt.true) (and (not (= ?xpy6 Smt.true)) (not (= ?ypy6 Smt.true))))) :pattern ((boolOr ?xpy6 ?ypy6)) )) :named A15))
(assert (! (forall ((?xpy7 Int)) (! (and (or (not (= ?xpy7 Smt.true)) (not (= (boolNot ?xpy7) Smt.true))) (or (= ?xpy7 Smt.true) (= (boolNot ?xpy7) Smt.true))) :pattern ((boolNot ?xpy7)) )) :named A16))
(assert (! (forall ((?xpy8 Int)(?ypy7 Int)) (! (and (or (not (= (anyEqual ?xpy8 ?ypy7) Smt.true)) (= ?xpy8 ?ypy7)) (or (not (= ?xpy8 ?ypy7)) (= (anyEqual ?xpy8 ?ypy7) Smt.true))) :pattern ((anyEqual ?xpy8 ?ypy7)) )) :named A17))
(assert (! (forall ((?xpy9 Int)(?ypy8 Int)) (! (and (or (not (= (anyNeq ?xpy9 ?ypy8) Smt.true)) (not (= ?xpy9 ?ypy8))) (or (= (anyNeq ?xpy9 ?ypy8) Smt.true) (= ?xpy9 ?ypy8))) :pattern ((anyNeq ?xpy9 ?ypy8)) )) :named A18))
(assert (! (forall ((?xpy10 Int)(?ypy9 Int)) (! (and (or (not (= (intLess ?xpy10 ?ypy9) Smt.true)) (not (<= ?ypy9 ?xpy10))) (or (= (intLess ?xpy10 ?ypy9) Smt.true) (<= ?ypy9 ?xpy10))) :pattern ((intLess ?xpy10 ?ypy9)) )) :named A19))
(assert (! (forall ((?xpy11 Int)(?ypy10 Int)) (! (and (or (not (= (intAtMost ?xpy11 ?ypy10) Smt.true)) (<= ?xpy11 ?ypy10)) (or (= (intAtMost ?xpy11 ?ypy10) Smt.true) (not (<= ?xpy11 ?ypy10)))) :pattern ((intAtMost ?xpy11 ?ypy10)) )) :named A20))
(assert (! (forall ((?xpy12 Int)(?ypy11 Int)) (! (and (or (not (= (intAtLeast ?xpy12 ?ypy11) Smt.true)) (<= ?ypy11 ?xpy12)) (or (= (intAtLeast ?xpy12 ?ypy11) Smt.true) (not (<= ?ypy11 ?xpy12)))) :pattern ((intAtLeast ?xpy12 ?ypy11)) )) :named A21))
(assert (! (forall ((?xpy13 Int)(?ypy12 Int)) (! (and (or (not (= (intGreater ?xpy13 ?ypy12) Smt.true)) (not (<= ?xpy13 ?ypy12))) (or (<= ?xpy13 ?ypy12) (= (intGreater ?xpy13 ?ypy12) Smt.true))) :pattern ((intGreater ?xpy13 ?ypy12)) )) :named A22))
(assert (! (not (= Smt.false Smt.true)) :named A23))
(assert (! (forall ((?tpy0 Int)) (! (subtypes ?tpy0 ?tpy0) :pattern ((subtypes ?tpy0 ?tpy0)) )) :named A24))
(assert (! (forall ((?tpy1 Int)(?upy0 Int)(?vpy5 Int)) (! (or (not (subtypes ?tpy1 ?upy0)) (not (subtypes ?upy0 ?vpy5)) (subtypes ?tpy1 ?vpy5)) :pattern ((subtypes ?tpy1 ?upy0) (subtypes ?upy0 ?vpy5)) )) :named A25))
(assert (! (forall ((?tpy2 Int)(?upy1 Int)) (! (or (= ?tpy2 ?upy1) (not (subtypes ?tpy2 ?upy1)) (not (subtypes ?upy1 ?tpy2))) :pattern ((subtypes ?tpy2 ?upy1) (subtypes ?upy1 ?tpy2)) )) :named A26))
(assert (! (not (= start_correct Smt.true)) :named A27_0))
(assert (! (or (= start_correct Smt.true) (and (= (isA o T_) Smt.true) (<= e 20))) :named A27_1))
(check-sat)
(get-info :reason-unknown)
;z3 -T:600 group_018/ematching/tmp/Quantifiers_X0-noinfer_std_unique_aug-gt_unsat-full.smt2
;unknown
;((:reason-unknown "smt tactic failed to show goal to be sat/unsat (incomplete quantifiers)"))
