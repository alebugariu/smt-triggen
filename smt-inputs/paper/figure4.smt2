(set-option :AUTO_CONFIG false)
(set-option :smt.MBQI false)

(declare-fun $div (Int Int) Int)
(declare-fun $sign (Int) Int)

(assert (! (forall ((a0 Int)(b0 Int)) (! (or (not (or (<= 0 a0) (= (mod a0 b0) 0))) (= ($div a0 b0) (+ (/ a0 b0) 0))) :pattern (($div a0 b0)))) :named A0))
(assert (! (forall ((a1 Int)(b1 Int)) (! (or (or (<= 0 a1) (= (mod a1 b1) 0)) (= ($div a1 b1) (+ (/ a1 b1) ($sign b1)))) :pattern (($div a1 b1)))) :named A1))

;(declare-fun __dummy__ (Int) Bool)
;(assert (__dummy__ ($div -3 -2)))

;(check-sat)