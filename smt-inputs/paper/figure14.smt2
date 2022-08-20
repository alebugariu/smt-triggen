(set-option :smt.auto-config false)
(set-option :smt.mbqi false)

(declare-fun len (Int) Int)
(declare-fun next (Int) Int)

(assert (! (forall ((x0 Int)) (! (> (len x0) 0) :pattern ((len (next x0))))) :named A0))
(assert (! (forall ((x1 Int)) (! (or (= (next x1) x1) (= (len x1) (+ (len (next x1)) 1))) :pattern ((len (next x1))))) :named A1))
(assert (! (forall ((x2 Int)) (! (or (not (= (next x2) x2)) (= (len x2) 1)) :pattern ((len (next x2))))) :named A2))
(assert (! (not (> (len 7) 0)) :named A3))

;(declare-fun __dummy__ (Int) Bool)
;(assert (__dummy__ (len (next 7))))

;(check-sat)