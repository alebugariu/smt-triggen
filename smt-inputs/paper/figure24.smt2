(set-option :AUTO_CONFIG false)
(set-option :smt.MBQI false)

(declare-fun f (Int Int) Bool)
(declare-fun a () Int)

(assert (! (forall ((x0 Int) (y0 Int)) (! (or (not (> x0 a)) (= x0 y0)) :pattern ( (f x0 y0)))) :named A0))
(assert (! (forall ((x1 Int) (y1 Int)) (! (or (not (<= x1 a)) (= x1 (* (- 1) y1))) :pattern ( (f x1 y1)))) :named A1))

;(declare-fun __dummy__ (Bool) Bool)
;(assert (__dummy__ (f (+ a 1) 0)))
;(assert (__dummy__ (f a 0)))

;(assert (__dummy__ (f 0 3)))

(check-sat)
