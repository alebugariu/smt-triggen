(set-option :AUTO_CONFIG false)
(set-option :smt.MBQI false)

(declare-fun f (Int) Int)
(declare-fun g (Int) Int)

(declare-const x2 Int)

(assert (! (forall ((x0 Int)) (! (not (= (f x0) x2)) :pattern ( (f x0)))) :named A0))
(assert (! (forall ((x1 Int)) (! (= (f (g x1)) x1) :pattern ( (f (g x1))))) :named A1))
(assert (! (> x2 6) :named A2))
(assert (! (< x2 8) :named A3))


;(declare-fun __dummy__ (Int) Bool)
;(assert (__dummy__ (f (g 7)))) 

;(check-sat)
