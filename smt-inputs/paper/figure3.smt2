(set-option :AUTO_CONFIG false)
(set-option :smt.MBQI false)

; A0: x0: Int :: {f(x0)} f(x0) > 7
; A1: x1: Int :: {f(x1)} f(x1) > 6

(declare-fun f (Int) Int)

(assert (! (forall ((x0 Int)) (! (> (f x0) 7) :pattern ((f x0)) )) :named A0))
(assert (! (forall ((x1 Int)) (! (> (f x1) 6) :pattern ((f x1)) )) :named A1))

;(check-sat)