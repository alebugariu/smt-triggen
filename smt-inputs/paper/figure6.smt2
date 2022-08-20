(set-option :AUTO_CONFIG false)
(set-option :smt.MBQI false)

(declare-fun f (Int) Int)
(declare-fun g (Int) Int)

(assert (! (forall ((x0 Int)) (! (not (= (f x0) 7)) :pattern ( (f x0)))) :named A0))
(assert (! (forall ((x1 Int)) (! (= (f (g x1)) x1) :pattern ( (f (g x1))))) :named A1))

;(declare-fun __dummy__ (Int) Bool)
;(assert (__dummy__ (f (g 7)))) 

;(check-sat)