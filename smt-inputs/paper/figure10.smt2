(set-option :AUTO_CONFIG false)
(set-option :smt.MBQI false)

(declare-sort B 0)
(declare-fun f (Int) Int)
(declare-fun g (B) Bool)

(assert (! (forall ((x0 Int)) (! (not (= (f x0) 7)) :pattern ( (f x0)))) :named A0))
(assert (! (forall ((b1 B) (x1 Int)) (! (or (g b1) (= (f x1) x1)) :pattern ( (g b1) (f x1)))) :named A1))
(assert (! (forall ((b2 B)) (! (not (g b2)) :pattern ( (g b2)))) :named A2))

;(declare-const b B)
;(declare-fun __dummy__ (Int Bool) Bool)
;(assert (__dummy__ (f 7) (g b))) 

;(check-sat)
