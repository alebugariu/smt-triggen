(set-option :AUTO_CONFIG false)
(set-option :smt.MBQI false)

(declare-fun P (Int) Bool)
(declare-fun Q (Int) Bool)
(declare-fun k () Int)

(assert (! (forall ((x0 Int)) (! (P x0) :pattern ( (Q x0)))) :named A0))
(assert (! (not (P k)) :named A1))

;(declare-fun __dummy__ (Bool) Bool)
;(assert (__dummy__ (Q k)))

;(check-sat)