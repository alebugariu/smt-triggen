(set-option :AUTO_CONFIG false)
(set-option :smt.MBQI false)

(declare-sort U 0)
(declare-fun f (U) Int)
(declare-fun g (U) Int)
(declare-fun P (U) Bool)

(assert (! (forall ((x0 U)) (! (not (P x0)) :pattern ( (f x0)))) :named A0))
(assert (! (forall ((x1 U)) (! (P x1) :pattern ( (g x1)))) :named A1))

;(declare-fun __dummy__ (Int Int) Bool)
;(declare-const x U)
;(assert (__dummy__ (f x) (g x))) 

;(check-sat)
