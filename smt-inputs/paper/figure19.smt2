(set-option :AUTO_CONFIG false)
(set-option :smt.MBQI false)

(declare-fun f (Int) Int)
(declare-fun g (Int) Int)

(assert (! (forall ((x0 Int)) (! (= (f (g x0)) x0) :pattern ( (f (g x0))))) :named A0))
(assert (! (= (g 2020) (g 2021)) :named A1))

;(declare-fun __dummy__ (Int Int) Bool)
;(assert (__dummy__ (f (g 2020)) (f (g 2021))))

;(declare-fun __dummy__ (Int) Bool)
;(assert (__dummy__ (f (g 2020)))) ; works as well
                                   ; (the solver applies f on both sides and triggers another instantiation)

;(check-sat)
;Solved by: Optset(top_level=False)