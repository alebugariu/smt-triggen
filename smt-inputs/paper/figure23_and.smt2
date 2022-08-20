(set-option :AUTO_CONFIG false)
(set-option :smt.MBQI false)

(declare-fun f (Bool Bool) Bool)

(assert (! (forall ((x0 Bool) (y0 Bool)) (! (or (not (f x0 y0)) (forall ((z0 Bool) (t0 Bool)) (! (and z0 (f z0 t0)) :pattern ( (f z0 t0) ))))
            :pattern ( (f x0 y0)))) :named A0))
(assert (! (= true (f true true)) :named A1))

;(declare-fun __dummy__ (Bool Bool) Bool)
;(assert (__dummy__ (f false true) (f true true)))

(check-sat)
