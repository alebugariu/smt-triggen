(set-option :AUTO_CONFIG false)
(set-option :smt.MBQI false)

(declare-fun _div (Int Int) Int)

(assert (! (forall ((x Int)(y Int)) (! (= (_div x y) (/ x y)) :pattern ((_div x y)))) :named A0))

;(declare-fun __dummy__ (Int) Bool)
;(assert (__dummy__ (_div -3 4)))

;(check-sat)