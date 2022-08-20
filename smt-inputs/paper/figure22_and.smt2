(set-option :AUTO_CONFIG false)
(set-option :smt.MBQI false)

(declare-fun f (Bool Bool) Bool)

(assert (! (forall ((x0 Bool) (y0 Bool)) (! (and x0 (f x0 y0)) :pattern ( (f x0 y0)))) :named A0))

;(declare-fun __dummy__ (Bool) Bool)
;(assert (__dummy__ (f false true)))

(check-sat)
