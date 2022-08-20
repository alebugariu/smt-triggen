(set-option :AUTO_CONFIG false)
(set-option :smt.MBQI false)

(declare-fun f (Int) Int)

(assert (! (forall ((x0 Int)) (! (> (f x0) 7) :pattern ( (f x0)))) :named A0))
(assert (! (forall ((x1 Int)) (! (= (f x1) 6) :pattern ( (f x1)))) :named A1))

;(declare-fun __dummy__ (Int) Bool)
;(assert (__dummy__ (f 0))) 

;(check-sat)