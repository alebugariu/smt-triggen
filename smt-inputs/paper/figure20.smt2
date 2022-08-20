(set-option :AUTO_CONFIG false)
(set-option :smt.MBQI false)

(declare-sort U 0)
(declare-fun f (U) U)
(declare-fun g (U) U)
(declare-fun h (U) U)
(declare-fun i (U) U)

(assert (! (forall ((x0 U)) (! (= (f x0) (h x0)) :pattern ( (g x0)))) :named A0))
(assert (! (forall ((x1 U)) (! (not (= (i (f x1)) (i (h x1)))) :pattern ( (i (f x1)) (i (h x1))))) :named A1))

;(declare-fun x () U)
;(declare-fun __dummy__ (U U U) Bool)
;(assert (__dummy__ (i (f x)) (i (h x)) (g x)))

(check-sat)