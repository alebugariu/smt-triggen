(set-option :AUTO_CONFIG false)
(set-option :smt.MBQI false)

(declare-sort U 0)
(declare-fun f (U U) Int)

(declare-fun a () U)
(declare-fun b () U)

(assert (! (forall ((x0 U) (y0 U)) (! (or (= x0 y0) (= (f x0 y0) 0)) :pattern ( (f x0 y0)))) :named A0))
(assert (! (forall ((x1 U) (y1 U)) (! (or (not (= x1 a)) (not (= y1 b)) (= (f x1 y1) 1)) :pattern ( (f x1 y1)))) :named A1))
(assert (! (not (= a b)) :named A2))

;(declare-fun __dummy__ (Int) Bool)
;(assert (__dummy__ (f a b)))

(check-sat)