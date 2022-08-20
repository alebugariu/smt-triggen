(set-option :AUTO_CONFIG false)
(set-option :smt.MBQI false)

(declare-fun f (Bool Bool) Bool)

(assert (! (forall ((x0 Bool) (y0 Bool)) (! x0 :pattern ( (f x0 y0)))) :named A0))
(assert (! (forall ((x1 Bool) (y1 Bool)) (! (f x1 y1) :pattern ( (f x1 y1)))) :named A1))

;(declare-fun __dummy__ (Bool) Bool)
;(assert (__dummy__ (f false true)))

(check-sat)
