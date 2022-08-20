(set-option :AUTO_CONFIG false)
(set-option :smt.MBQI false)

(declare-fun f (Int) Int)
(declare-fun g (Int) Int)

(assert (! (forall ((x Int)) (! (not (= (f x) (f 7))) :pattern ((g x)))) :named A0))

(check-sat)

;Solved by: Optset(max_axiom_frequency=2)